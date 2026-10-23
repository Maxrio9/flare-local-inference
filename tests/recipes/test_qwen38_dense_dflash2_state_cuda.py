"""Production snapshot/restore preserves complete DFlash2 windows across wrap."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.paged_batch import BatchInputs
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState, RequestAdmission
from ignition.components.state_transfer import StateCommands
from ignition.recipes.qwen38_dense.config import ModelConfig
from ignition.recipes.qwen38_dense.dflash2_config import DFlash2Config
from ignition.recipes.qwen38_dense.pool import ContinuationPool

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires admitted CUDA')


@pytest.fixture(scope='module')
def state_model():
    snapshot = Path(os.environ.get('FLARE_QWEN38_DENSE_SNAPSHOT', '/data/models/Qwen3.8-27B-NVFP4'))
    if not snapshot.is_dir():
        pytest.skip('requires target structural config')
    config = ModelConfig.load(snapshot)
    raw = json.loads((Path(__file__).parent/'fixtures/qwen38_dflash2_config.json').read_text())
    config = replace(config, draft=DFlash2Config.from_dict(raw, config))
    pool = ContinuationPool.for_model(config, 4, 40, 'cuda', speculative_tokens=7)
    requests = DeviceRequestState.allocate('cuda', vocab_size=config.vocabulary, capacity=1)
    inputs = BatchInputs.allocate(512, 64, 'cuda', capacity=1)
    commands = StateCommands.allocate('cuda', capacity=1)
    admission = AdmissionBuffers.allocate('cuda', vocab_size=config.vocabulary, capacity=1)
    completion = CompletionOutputs.allocate((2,1,24), device='cuda')
    frame, sequence = torch.zeros(1,device='cuda',dtype=torch.int32), torch.zeros(1,device='cuda',dtype=torch.int64)
    forwards = {mode: PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
        runtime=None, coordinator=None, session=None, name=mode, capacity=1,
        admission_buffers=admission, draft_state=True) for mode in ('snapshot','restore')}
    graphs = {}
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for f in forwards.values():
            f()
    stream.synchronize()
    for mode,f in forwards.items():
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):f()
        graphs[mode]=graph
    yield SimpleNamespace(pool=pool, requests=requests, inputs=inputs, commands=commands,
        admission=admission, graphs=graphs)
    stream.synchronize()
    for graph in graphs.values():graph.reset()


@pytest.mark.parametrize('prefix',(1,2,127,128,129,2047,2048,2049,4095,4096,4097))
@torch.no_grad()
def test_captured_restore_admits_new_generation_and_preserves_full_window(state_model,prefix):
    m=state_model;p,r,i,c,a=m.pool,m.requests,m.inputs,m.commands,m.admission
    # Arbitrary bytes expose dropped fields, partial windows and stale aliases.
    p.backing.copy_(torch.randint(0,256,p.backing.shape,dtype=torch.uint8,device='cuda'))
    p.region('consumed_tokens')[1]=prefix
    prompt=[42]*prefix
    epoch=prefix*100+1
    r.admit(0,RequestAdmission(generation=epoch,seed=123,max_new_tokens=64),prompt,consumed_tokens=prefix)
    i.request_state_rows.zero_();i.request_state_rows[0,0]=1;i.row_slot_ids[0]=0;i.row_generations[0]=epoch;i.live_rows.fill_(1)
    i.page_tables.fill_(-1);i.page_tables[0,:40]=torch.arange(40,device='cuda',dtype=torch.int32)
    c.request_slots[0]=0;c.source_slots[0]=1;c.destination_slots[0]=2;c.operation[0]=0;c.expected_consumed[0]=prefix;c.num_ops.fill_(1)
    m.graphs['snapshot'].replay();torch.cuda.synchronize()
    assert c.errors.item()==0
    snapshot=p.backing[2].clone();expected=snapshot;null=p.backing[0].clone();other=p.backing[3].clone()
    # Snapshot rows replace the active row's dormant partial-page fields with
    # cache bytes. Compare restored declared fields to that serialized state,
    # excluding alignment padding, which is not part of the state contract.
    page, tail = prefix//128, prefix%128
    partial = {name: value[page,:tail*(value.shape[1]//128)].clone() for name,value in p.pages.items()}
    if tail:
        for value in p.pages.values():value[page].fill_(255)
    # Poison every ring entry, not only the boundary slot. Full continuation
    # restore must recover all 2048 positions in all five layers.
    p.backing[1].fill_(255)
    a.stage(0,0,RequestAdmission(generation=epoch+1,seed=123,max_new_tokens=64),prompt);a.admission_count.fill_(1)
    c.source_slots[0]=2;c.destination_slots[0]=1
    m.graphs['restore'].replay();torch.cuda.synchronize()
    assert c.errors.item()==0
    differences = [region.name for region in p.layout.regions
        if not torch.equal(p.backing[1,region.offset:region.offset+region.nbytes],
                           expected[region.offset:region.offset+region.nbytes])]
    assert not differences, differences
    assert torch.equal(p.backing[2],snapshot)
    assert torch.equal(p.backing[0],null) and torch.equal(p.backing[3],other)
    for name,value in p.pages.items():
        assert torch.equal(value[page,:tail*(value.shape[1]//128)],partial[name])
    assert r.generation.item()==epoch+1 and r.consumed_tokens.item()==prefix
    assert r.generated_tokens.item()==0 and r.pending_token.item()==-1


@pytest.fixture(scope='module')
def context_writer(state_model):
    from b12x.preparation import PreparationSession
    from ignition.components.dflash import PreparedContext
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.scratch import GraphBufferPool,ScratchArena
    from ignition.recipes.qwen38_dense.dflash2_weights import DFlash2Weights
    snapshot=Path(os.environ.get('FLARE_DFLASH2_SNAPSHOT','/data/models/Qwen3.8-27B-DFlash2'))
    if not snapshot.is_dir():pytest.skip('requires BF16 DFlash2 checkpoint')
    raw=json.loads((Path(__file__).parent/'fixtures/qwen38_dflash2_config.json').read_text())
    target=SimpleNamespace(hidden=5120,vocabulary=248320,layers=tuple(range(64)))
    config=DFlash2Config.from_dict(raw,target)
    weights=SimpleNamespace(config=SimpleNamespace(draft=config,hidden=5120),draft=DFlash2Weights.load(snapshot,config,'cuda'))
    metadata=Metadata.allocate(8,1,64,'cuda',samples=8)
    auxiliary=torch.randn((8,5*5120),device='cuda',dtype=torch.bfloat16)*.1
    counts=torch.zeros(1,device='cuda',dtype=torch.int64)
    buffers=GraphBufferPool('cuda')
    with PreparationSession(device='cuda',autotune=False) as session:
        arena=ScratchArena('cuda')
        forward=PreparedContext(weights,auxiliary,metadata,state_model.pool,session=session,name='context',
            workspace=arena,buffers=buffers.cursor(),retain_anchor=False)
        sink_pair = _prepare_sink_pair(session, arena)
        arena.finalize()
        forward(auxiliary,counts)
        sink_pair[0](); sink_pair[1]()
        session.freeze()
        graph=torch.cuda.CUDAGraph()
        with session.capture(),torch.cuda.graph(graph):forward(auxiliary,counts)
        yield metadata,auxiliary,counts,graph,weights,sink_pair
        torch.cuda.synchronize();graph.reset()


@pytest.mark.parametrize('start',(2047,2048,2049,4095,4096,4097))
@pytest.mark.parametrize('accepted',range(9))
@torch.no_grad()
def test_rejected_rows_never_clobber_wrapped_context(state_model,context_writer,start,accepted):
    metadata,auxiliary,counts,graph,weights,_=context_writer
    pool=state_model.pool
    metadata.set(torch.tensor([start],device='cuda'),torch.tensor([8],device='cuda',dtype=torch.int32),
        torch.tensor([1],device='cuda'),torch.zeros((1,64),device='cuda',dtype=torch.int32),
        torch.zeros((1,8),device='cuda',dtype=torch.int64),torch.ones(1,device='cuda',dtype=torch.int32))
    counts.fill_(accepted)
    regions=[pool.reserved_region(f'draft{layer}.{field}') for layer in range(5) for field in ('key','value')]
    for value in regions:value.view(torch.uint8).fill_(255)
    graph.replay();torch.cuda.synchronize()
    written=torch.zeros((4,2048),device='cuda',dtype=torch.bool)
    written[1,(start+torch.arange(accepted,device='cuda'))%2048]=True
    for value in regions:
        # The accepted slots become finite; every other byte, including null
        # and unrelated rows and all rejected candidates, retains poison.
        assert torch.isfinite(value[written]).all()
        assert (value.view(torch.uint8)[~written]==255).all()


@pytest.mark.parametrize('region_name',('draft0.key','layer3.key'))
@torch.no_grad()
def test_restore_oracle_detects_corrupted_snapshot_field(state_model,region_name):
    # Exercise the production restore graph with a deliberately damaged saved
    # ring or partial-page byte. The original-state equality check must fail.
    m=state_model;p,r,i,c,a=m.pool,m.requests,m.inputs,m.commands,m.admission
    prefix,epoch=129,900001
    r.admit(0,RequestAdmission(generation=epoch,seed=3,max_new_tokens=8),[42]*prefix,consumed_tokens=prefix)
    p.backing.zero_();p.region('consumed_tokens')[1]=prefix
    i.request_state_rows[0,0]=1;i.row_slot_ids[0]=0;i.row_generations[0]=epoch;i.live_rows.fill_(1)
    i.page_tables[0,:40]=torch.arange(40,device='cuda',dtype=torch.int32)
    c.request_slots[0]=0;c.source_slots[0]=1;c.destination_slots[0]=2;c.operation[0]=0;c.expected_consumed[0]=prefix;c.num_ops.fill_(1)
    m.graphs['snapshot'].replay();torch.cuda.synchronize();assert c.errors.item()==0
    field=p.layout.region(region_name);original=p.backing[2,field.offset:field.offset+field.nbytes].clone()
    p.backing[2,field.offset]^=1
    a.stage(0,0,RequestAdmission(generation=epoch+1,seed=3,max_new_tokens=8),[42]*prefix);a.admission_count.fill_(1)
    c.source_slots[0]=2;c.destination_slots[0]=1
    m.graphs['restore'].replay();torch.cuda.synchronize();assert c.errors.item()==0
    restored=p.backing[1,field.offset:field.offset+field.nbytes]
    assert not torch.equal(restored,original)
    assert restored[0].item()==(original[0].item() ^ 1)


def _prepare_sink_pair(session, arena):
    from ignition.components.paged_attention import PreparedPagedAttention
    q=torch.zeros((8,32,128),device='cuda',dtype=torch.bfloat16)
    k=torch.zeros((1,128,8,128),device='cuda',dtype=torch.bfloat16)
    v=torch.ones_like(k)
    pages=torch.zeros((1,1),device='cuda',dtype=torch.int32)
    lengths=torch.tensor([8],device='cuda',dtype=torch.int32)
    cu=torch.tensor([0,8],device='cuda',dtype=torch.int32)
    no_sink,zero_sink=torch.empty_like(q),torch.empty_like(q)
    sink=torch.zeros(32,device='cuda',dtype=torch.float32)
    plain=PreparedPagedAttention(q,k,v,pages,lengths,cu,no_sink,mode='verify',width=8,
        causal=False,window=2047,sink=None,session=session,workspace=arena,name='absent')
    extra=PreparedPagedAttention(q,k,v,pages,lengths,cu,zero_sink,mode='verify',width=8,
        causal=False,window=2047,sink=sink,session=session,workspace=arena,name='zero')
    return plain,extra,no_sink,zero_sink


@torch.no_grad()
def test_absent_sink_differs_from_zero_logit_sink(context_writer):
    # All plans belong to the fixture's preparation batch, before freezing.
    plain,extra,no_sink,zero_sink = context_writer[-1]
    plain();extra();torch.cuda.synchronize()
    torch.testing.assert_close(no_sink,torch.ones_like(no_sink),rtol=0,atol=0)
    torch.testing.assert_close(zero_sink,torch.full_like(zero_sink,8/9),rtol=0,atol=0)
    assert not torch.equal(no_sink,zero_sink)

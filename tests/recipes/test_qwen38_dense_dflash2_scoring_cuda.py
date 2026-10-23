"""Scored-prefill absorption preserves dense target scores and full draft state."""

import os
from pathlib import Path

import pytest
import torch

pytestmark=pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS')!='1',reason='requires admitted CUDA')


@torch.no_grad()
def test_multichunk_scored_prefill_preserves_target_scores_and_draft_window():
    from ignition.components.paged_batch import BatchInputs
    from ignition.components.prompt_logprobs import PromptScoring
    from ignition.components.sampling import DeviceRequestState,RequestAdmission
    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig,prepare
    from ignition.recipes.qwen38_dense.pool import ContinuationPool
    from tokenizers import Tokenizer

    target=Path(os.environ.get('FLARE_QWEN38_DENSE_SNAPSHOT','/data/models/Qwen3.8-27B-NVFP4'))
    draft=Path(os.environ.get('FLARE_DFLASH2_SNAPSHOT','/data/models/Qwen3.8-27B-DFlash2'))
    if not target.is_dir() or not draft.is_dir():pytest.skip('requires pinned target and DFlash2 checkpoints')
    prompt=Tokenizer.from_file(str(target/'tokenizer.json')).encode('\n'.join(
        f'Record {i}: owner team-{i%17}; timeout {250+i*13} milliseconds; retries {i%4}.'
        for i in range(16)),add_special_tokens=False).ids
    assert 256<len(prompt)<384
    with PackageStorage() as storage:
        model=prepare(target,PreparationConfig(mtp=True,speculative_proposal_source='dflash2',
            draft_checkpoint=draft,prompt_logprobs=True,speculative_widths=(1,8),
            max_active=1,context=2048,prefill_chunk_size=128,snapshot_budget_bytes=400<<20,
            headroom_bytes=2<<30,compile_decoder=True,compile_workers=4),package_storage=storage)
    try:
        requests=next(x for x in model.resources if isinstance(x,DeviceRequestState))
        inputs=next(x for x in model.resources if isinstance(x,BatchInputs))
        pool=next(x for x in model.resources if isinstance(x,ContinuationPool))
        scoring=next(x for x in model.resources if isinstance(x,PromptScoring))
        results={}
        for generation,kind in enumerate(('prefill128-spec','scored_prefill128','scored_prefill128-spec'),1):
            pool.clear_for_preparation()
            requests.admit(0,RequestAdmission(generation=generation,seed=17,max_new_tokens=128),prompt)
            inputs.request_state_rows.zero_();inputs.request_state_rows[0,0]=1
            inputs.row_slot_ids[0]=0;inputs.row_generations[0]=generation
            inputs.live_rows.fill_(1);inputs.sample_enabled.zero_();inputs.page_tables.fill_(-1)
            pages=int(pool.memory_capacity[1]);inputs.page_tables[0,:pages]=torch.arange(pages,device='cuda',dtype=torch.int32)
            scoring.frame.zero_();scores=[]
            for start in range(0,len(prompt),128):
                chunk=prompt[start:start+128]
                inputs.query_lengths.fill_(len(chunk));inputs.token_ids[:len(chunk)]=torch.tensor(chunk,device='cuda')
                scoring.outputs.targets.fill_(-1)
                targets=prompt[start+1:start+len(chunk)+1]
                scoring.outputs.targets[:len(targets)]=torch.tensor(targets,device='cuda',dtype=torch.int32)
                model.graphs[kind].replay();torch.cuda.synchronize()
                assert requests.error_code.item()==0
                if kind.startswith('scored_'):scores.append(scoring.outputs.values[0,:len(targets)].clone())
            state={region.name:pool.region(region.name)[1].clone() for region in pool.layout.regions
                if region.name!='consumed_tokens'}
            state.update({f'page.{name}':value[:(len(prompt)+127)//128].clone() for name,value in pool.pages.items()})
            assert requests.consumed_tokens.item()==len(prompt) and requests.generated_tokens.item()==0
            results[kind]=(state,torch.cat(scores) if scores else None)
        normal=results['prefill128-spec'][0]
        actual=results['scored_prefill128-spec'][0]
        assert all(torch.equal(normal[k].view(torch.uint8),actual[k].view(torch.uint8)) for k in normal)
        plain=results['scored_prefill128']
        assert plain[1].shape[0]==len(prompt)-1
        assert torch.equal(plain[1],results['scored_prefill128-spec'][1])
        for name,value in plain[0].items():
            if not name.startswith('draft'):
                assert torch.equal(value.view(torch.uint8),actual[name].view(torch.uint8)),name
    finally:model.close()

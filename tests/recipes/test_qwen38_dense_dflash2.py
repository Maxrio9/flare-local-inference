"""DFlash2 structural validation and predecessor-conditioned proposal semantics."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.qwen38_dense.dflash2_config import DFlash2Config
from ignition.recipes.qwen38_dense.dflash2_selector import edge_scores


def config():
    raw = json.loads((Path(__file__).parent / 'fixtures/qwen38_dflash2_config.json').read_text())
    target = SimpleNamespace(hidden=5120, vocabulary=248320, layers=tuple(range(64)))
    return raw, target


def test_checkpoint_config():
    raw, target = config()
    c = DFlash2Config.from_dict(raw, target)
    assert (c.block_size, c.layers, c.noise_token, c.target_layers) == (8, 5, 248070, (5, 19, 33, 47, 61))


@pytest.mark.parametrize('field,value', [('is_causal', True), ('attention_sink_bias', True),
    ('num_key_value_heads', 4), ('num_target_layers', 63), ('quantization_config', {'quant_algo': 'FP8'})])
def test_rejects_changed_model_semantics(field, value):
    raw, target = config()
    raw[field] = value
    with pytest.raises(ValueError):
        DFlash2Config.from_dict(raw, target)


@pytest.mark.parametrize('field,value', [('block_size', 16), ('target_layer_ids', [6,20,34,48,62]),
    ('mask_token_id', 248320), ('selector_top_k', 8)])
def test_rejects_changed_proposal_geometry(field, value):
    raw, target = config()
    raw['dflash_config'][field] = value
    with pytest.raises(ValueError):
        DFlash2Config.from_dict(raw, target)


def test_selector_scores_realized_predecessor_and_exact_conditional_q():
    torch.manual_seed(45)
    b, slots, k, vocab, rank = 2, 7, 4, 41, 8
    a, z = torch.randn(vocab, rank), torch.randn(vocab, rank)
    h = torch.randn(b, slots, rank)
    ids = torch.stack([torch.stack([torch.randperm(vocab)[:k] for _ in range(slots)]) for _ in range(b)])
    unary, anchors = torch.randn(b, slots, k), torch.tensor([39,40])
    actual = edge_scores(a, z, h, ids, unary, anchors)
    expected = torch.empty_like(actual)
    for r in range(b):
        for s in range(slots):
            for p in range(k):
                predecessor = anchors[r] if s == 0 else ids[r,s-1,p]
                for c in range(k):
                    expected[r,s,p,c] = unary[r,s,c] + (a[predecessor] * h[r,s] * z[ids[r,s,c]]).sum()
    torch.testing.assert_close(actual, expected)
    # Every possible predecessor selects its own normalized q; it is never the
    # distribution averaged over predecessors or the uncorrected head logits.
    for p in range(k):
        dense = torch.full((b, vocab), -torch.inf).scatter(1, ids[:,3], actual[:,3,p])
        q = dense.softmax(-1)
        torch.testing.assert_close(q.gather(1, ids[:,3]), expected[:,3,p].softmax(-1))
        assert (q.sum(-1) == 1).all() or torch.allclose(q.sum(-1), torch.ones(b))
        assert (q.gather(1, ids[:,3]) > 0).all()


def test_selector_position_and_invalid_predecessor():
    from ignition.recipes.qwen38_dense.dflash2_selector import PreparedSelector
    selector = object.__new__(PreparedSelector)
    selector.ids = torch.tensor([[[3,5],[7,9]]])
    selector.scores = torch.tensor([[[[1.,2.],[1.,2.]],[[3.,4.],[5.,6.]]]])
    selector.rows = torch.arange(1)
    logits = torch.zeros((1,12))
    selector(logits, torch.tensor([11]), 1)
    assert logits[0,3].item() == 1 and logits[0,5].item() == 2
    selector(logits, torch.tensor([5]), 2)
    assert logits[0,7].item() == 5 and logits[0,9].item() == 6
    selector(logits, torch.tensor([4]), 2)
    assert logits.isneginf().all()


def test_explicit_recipe_alias_preserves_existing_dense_default(tmp_path):
    from ignition.recipes import resolve
    assert resolve('qwen3.8-27b',tmp_path).recipe.name=='qwen38_dense'
    draft=resolve('qwen3.8-27b-dflash2',tmp_path)
    assert draft.recipe.name=='qwen38_dense_dflash2'
    assert draft.recipe.ranks[1].draft_page_bytes==0
    assert draft.checkpoint.draft_bytes==3_953_666_560


def test_calibrated_nvfp4_preserves_independent_projection_declarations():
    _,target=config()
    raw=json.loads((Path(__file__).parent/'fixtures/qwen38_dflash2_nvfp4_config.json').read_text())
    draft=DFlash2Config.from_dict(raw,target)
    assert len(draft.quantization)==35
    assert 'fc' not in draft.quantization
    assert 'layers.0.attention_conv.kernel_projection' not in draft.quantization
    assert set(draft.quantization['layers.0.self_attn.q_proj'])=={'quant_algo','group_size'}
    raw['quantization_config']['ignore']=[]
    with pytest.raises(ValueError):DFlash2Config.from_dict(raw,target)


@pytest.mark.parametrize('mask',range(16))
def test_pending_anchor_follows_logical_requests_through_packed_inactive_rows(mask):
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.qwen38_dense.dflash2 import draft_anchors
    m=Metadata.allocate(32,4,1,'cpu',samples=32)
    active=torch.tensor([bool(mask&(1<<r)) for r in range(4)])
    ids=torch.tensor([[11+r]+[99]*7 for r in range(4)])
    m.set(torch.tensor([10,20,30,40]),active.int()*8,torch.arange(1,5),
        torch.zeros((4,1),dtype=torch.int32),ids,torch.tensor([4],dtype=torch.int32))
    torch.testing.assert_close(draft_anchors(m,8)[active],ids[:,0][active],rtol=0,atol=0)


def test_inactive_selector_nonfinite_scores_do_not_cross_request_rows():
    from ignition.recipes.qwen38_dense.dflash2_selector import PreparedSelector
    selector = object.__new__(PreparedSelector)
    selector.ids = torch.tensor([[[3,5],[7,9]]]*4)
    selector.scores = torch.arange(4*2*2*2,dtype=torch.float32).view(4,2,2,2)
    selector.scores[1].fill_(torch.nan)
    selector.scores[3].fill_(torch.inf)
    selector.rows = torch.arange(4)
    logits = torch.zeros((4,12))
    selector(logits,torch.tensor([5,5,3,5]),2)
    for row,previous in ((0,1),(2,0)):
        torch.testing.assert_close(logits[row,[7,9]],selector.scores[row,1,previous],rtol=0,atol=0)
        assert torch.isfinite(logits[row,[7,9]]).all()
    assert logits[1,[7,9]].isnan().all()
    assert logits[3,[7,9]].isposinf().all()


def test_nvfp4_checkpoint_retains_bf16_lineage_and_shared_calibration_scales():
    import os
    from safetensors import safe_open
    original=os.environ.get('FLARE_DFLASH2_BF16_CHECKPOINT')
    quantized=os.environ.get('FLARE_DFLASH2_NVFP4_CHECKPOINT')
    if not original or not quantized:
        pytest.skip('requires both pinned DFlash2 checkpoint files; CPU read only')
    with safe_open(original,framework='pt',device='cpu') as base, safe_open(quantized,framework='pt',device='cpu') as packed:
        unquantized=[key for key in packed.keys() if packed.get_slice(key).get_dtype()=='BF16']
        assert len(unquantized)==46
        for key in unquantized:
            assert torch.equal(base.get_tensor(key).view(torch.uint8),packed.get_tensor(key).view(torch.uint8)),key
        for layer in range(5):
            for names in (('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'),('mlp.gate_proj','mlp.up_proj')):
                for field in ('input_scale','weight_scale_2'):
                    values=[packed.get_tensor(f'layers.{layer}.{name}.{field}') for name in names]
                    assert all(torch.isfinite(v).all() and (v>0).all() for v in values)
                    assert all(torch.equal(values[0],v) for v in values[1:])


@pytest.mark.parametrize('text,theta', [
    ({'head_dim':128,'rope_theta':10000.,'rope_parameters':None},10000.),
    ({'head_dim':128,'rope_parameters':{'rope_theta':10000000.}},10000000.),
])
def test_shared_rotary_preserves_root_and_nested_checkpoint_conventions(text,theta):
    from ignition.components.dflash import Rotary
    rotary=Rotary(SimpleNamespace(text=text),'cpu')
    expected=1/theta**(torch.arange(0,128,2,dtype=torch.float32)/128)
    torch.testing.assert_close(rotary.inverse,expected,rtol=0,atol=0)


def test_default_dflash2_catalogs_fit_the_32gb_card_and_nvfp4_returns_cache_capacity():
    from ignition import planning
    from ignition.recipes.qwen38_dense.dflash2_recipe import RECIPE,NVFP4_RECIPE
    card=planning.Gpu(0,'NVIDIA RTX PRO 4500 Blackwell','GPU-planning-fixture',33759756288,
        int(31.4*(1<<30)),(12,0))
    plans=[planning.plan(recipe,recipe.checkpoints[0],(card,),selected=(0,),concurrency=1,
        reserve_bytes=0,speculation=recipe.speculation) for recipe in (RECIPE,NVFP4_RECIPE)]
    assert all(p.fits and p.context==32768 and p.prefill_chunk==512 for p in plans)
    assert plans[1].pages>plans[0].pages

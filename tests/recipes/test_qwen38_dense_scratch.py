"""Dense graph variants share bounded intermediates with disjoint live fields."""

from types import SimpleNamespace

import torch

from ignition.recipes.qwen38_dense.decoder import Scratch


def test_graph_capacity_views_share_mlp_storage_without_overlapping_live_fields():
    config = SimpleNamespace(hidden=5120, vocabulary=16, text={"intermediate_size": 17408,
        "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256})
    scratch = Scratch.allocate(8, 4, config, "cpu")
    small = scratch.rows(3, 1)
    fields = ("mlp_gate", "mlp_up", "mlp_down", "mlp_input")
    tensors = [getattr(scratch, key) for key in fields]
    assert len({value.untyped_storage().data_ptr() for value in tensors}) == 4
    assert sum(value.nbytes for value in tensors) == 8 * (3 * 17408 + 5120) * 2
    assert all(getattr(small, key).data_ptr() == getattr(scratch, key).data_ptr() for key in fields)
    for index, key in enumerate(fields, 1):
        getattr(small, key).fill_(index)
    for index, key in enumerate(fields, 1):
        value = getattr(scratch, key)
        assert value[:3].eq(index).all() and value[3:].eq(0).all()
        assert getattr(small, key).is_contiguous()
    assert scratch.activations.eq(0).all() and scratch.hidden.eq(0).all()


def test_attention_capacity_views_preserve_disjoint_live_fields_and_row_tails():
    config = SimpleNamespace(hidden=5120, vocabulary=16, text={"intermediate_size": 17408,
        "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256})
    scratch = Scratch.allocate(8, 4, config, "cpu")
    small = scratch.rows(3, 1)
    assert {key: value.shape[1] for key, value in scratch.attention.items()} == dict(
        qg=12288, k=1024, v=1024, q=6144, key=1024, attended=6144, output=5120)
    assert set(scratch.quantized) == {5120, 6144}
    fields = list(scratch.attention.values()) + list(scratch.quantized.values())
    other = [scratch.activations, scratch.hidden, scratch.logits, scratch.mlp_gate,
             scratch.mlp_up, scratch.mlp_down, scratch.mlp_input]
    assert len({t.untyped_storage().data_ptr() for t in fields + other}) == len(fields + other)
    for original, view in [(scratch.attention, small.attention), (scratch.quantized, small.quantized)]:
        for index, key in enumerate(original, 1):
            assert view[key].data_ptr() == original[key].data_ptr()
            assert view[key].shape == (3, original[key].shape[1]) and view[key].is_contiguous()
            view[key].fill_(index)
            assert original[key][:3].float().eq(index).all()
            assert original[key][3:].float().eq(0).all()
    assert all(t.float().eq(0).all() for t in other)

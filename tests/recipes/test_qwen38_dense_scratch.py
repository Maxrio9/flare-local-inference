"""Dense graph variants share bounded intermediates with disjoint live fields."""

from types import SimpleNamespace

import torch

from ignition.recipes.qwen38_dense.decoder import Scratch


def test_graph_capacity_views_share_mlp_storage_without_overlapping_live_fields():
    config = SimpleNamespace(hidden=5120, vocabulary=16, text={"intermediate_size": 17408})
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

"""Accepted MTP pairs remain causal when packed rows leave a padded suffix."""

import os

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.qwen38_dense.mtp import absorb_feedback


class FeedbackForward:
    def __init__(self, device):
        self.metadata = Metadata.allocate(32, 4, 1, device, max_query_tokens=8)
        self.previous = -32 - torch.arange(
            32, device=device, dtype=torch.bfloat16
        ).view(4, 8)
        self.target = torch.empty((32, 8), device=device, dtype=torch.bfloat16)
        self.target_begins = torch.zeros(4, device=device, dtype=torch.int32)
        self.output = torch.empty_like(self.target)
        self.set_case((8, 8, 8, 8), (8, 8, 8, 8))

    def set_case(self, queried, accepted):
        assert all(0 <= c <= q <= 8 for q, c in zip(queried, accepted))
        device = self.target.device
        begins = [sum(queried[:row]) for row in range(4)]
        self.target_begins.copy_(torch.tensor(begins, device=device))
        self.target.copy_(
            torch.arange(256, device=device, dtype=torch.bfloat16).view(32, 8)
        )
        self.target[sum(queried) :].fill_(torch.nan)
        self.output.fill_(123)
        self.metadata.set(
            torch.tensor([126, 382, 0, 65534], device=device),
            torch.tensor(accepted, device=device),
            torch.arange(1, 5, device=device),
            torch.zeros((4, 1), device=device, dtype=torch.int32),
            torch.zeros((4, 8), device=device, dtype=torch.int64),
            torch.tensor([4], device=device, dtype=torch.int32),
        )

    def __call__(self):
        absorb_feedback(
            self.previous, self.target, self.target_begins, self.metadata, self.output
        )


def expected_pairs(forward, queried, accepted):
    previous, target = forward.previous.cpu(), forward.target.cpu()
    expected, cursor = torch.zeros((32, 8), dtype=torch.bfloat16), 0
    for row, count in enumerate(accepted):
        for column in range(count):
            expected[cursor] = (
                previous[row]
                if column == 0
                else target[sum(queried[:row]) + column - 1]
            )
            cursor += 1
    return expected


CASES = [
    ((8, 8, 8, 8), (8, 8, 8, 8)),
    ((8, 8, 8, 8), (1, 1, 1, 1)),
    ((8, 8, 8, 0), (2, 3, 1, 0)),
    ((1, 4, 7, 0), (1, 1, 4, 0)),
    ((8, 0, 8, 0), (1, 0, 7, 0)),
    ((0, 0, 0, 0), (0, 0, 0, 0)),
    *[((8, 8, 8, 8), (prefix, 8, 5, 2)) for prefix in range(1, 9)],
]


@torch.no_grad()
def test_ragged_accepted_prefix_feedback_and_padding():
    forward = FeedbackForward("cpu")
    for queried, accepted in CASES:
        forward.set_case(queried, accepted)
        forward()
        torch.testing.assert_close(
            forward.output, expected_pairs(forward, queried, accepted), rtol=0, atol=0
        )


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires admitted CUDA"
)
@torch.no_grad()
def test_compiled_captured_feedback_masks_padding_before_target_indexing():
    from ignition.preparation.aot import aot_forward

    forward = FeedbackForward("cuda")
    compiled = aot_forward(forward)
    compiled(forward)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(forward)
    for queried, accepted in CASES:
        forward.set_case(queried, accepted)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            forward.output.cpu(),
            expected_pairs(forward, queried, accepted),
            rtol=0,
            atol=0,
        )
    graph.reset()

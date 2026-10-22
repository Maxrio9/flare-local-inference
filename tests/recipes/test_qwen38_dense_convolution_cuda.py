"""Accepted-prefix GDN convolution history across verification/decode transitions."""

import os

import pytest
import torch
import triton

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an admitted CUDA device"
)


@torch.no_grad()
def test_gdn_convolution_commit_preserves_accepted_prefix_and_inactive_rows():
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.sequence import short_conv
    from ignition.components.recurrent import DeltaMetadata, _commit_convolution
    from ignition.components.state import StateLayout, StateStorage

    device, channels, horizon = torch.device("cuda:0"), 10240, 10
    torch.manual_seed(38127)
    layout = StateLayout.from_specs(
        (("convolution", (channels, horizon), torch.bfloat16), ("sentinel", (256,), torch.int32))
    )
    pool = StateStorage.allocate(layout, 3, device)
    history = pool.region("convolution")
    history.normal_(std=0.2)
    pool.region("sentinel").fill_(31)
    original = pool.backing.clone()
    source = torch.randn((8, channels), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(source)
    weights = torch.randn((channels, 4), dtype=torch.bfloat16, device=device)
    metadata = DeltaMetadata.allocate(1, device, verification_columns=8)
    arguments = dict(
        x=source,
        conv_weight=weights,
        query_start_loc=metadata.query_start_loc,
        state_slot_ids=metadata.convolution_slots,
        state_is_fresh=metadata.fresh,
        num_accepted_tokens=metadata.accepted_tokens,
        num_seqs=metadata.num_seqs,
        num_tokens=metadata.num_tokens,
        conv_state=history,
        out=output,
    )
    caps = short_conv.Caps(
        device=device,
        mode="decode",
        channels=channels,
        kernel_size=4,
        max_tokens=8,
        max_seqs=1,
        max_state_slots=3,
        max_speculative_tokens=7,
    )
    plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**arguments))
    selected = {}

    def prepare(state):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in plan.scratch_specs()
        )
        binding = state.bind(plan=plan, scratch=scratch, **arguments)
        selected["owners"] = scratch, binding
        return PreparedCall(
            run=lambda: state.run(binding), restore=lambda: pool.backing.copy_(original)
        )

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare((plan.request(name="dense-gdn-history", prepare_call=prepare),))
    scratch = selected["owners"][0]
    binding = short_conv.bind(plan, scratch=scratch, **arguments)
    offsets = torch.tensor(
        [layout.region("convolution").offset // 2], dtype=torch.int64, device=device
    )
    rows = torch.ones(1, dtype=torch.int64, device=device)
    counts = torch.ones(1, dtype=torch.int32, device=device)

    def forward():
        short_conv.run(binding)
        torch.library.wrap_triton(_commit_convolution)[(1, 1, triton.cdiv(channels, 128))](
            pool.backing.view(torch.bfloat16),
            offsets,
            rows,
            counts,
            layout.row_bytes // 2,
            channels,
            horizon,
            128,
        )

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for width, accepted in ((5, 1), (5, 3), (5, 5), (8, 1), (8, 3), (8, 8)):
        pool.backing.copy_(original)
        expected = history[1, :, :3].clone()
        # Exercise W -> 1 -> W without resetting the live history between rounds.
        for length, count in ((width, accepted), (1, 1), (width, accepted)):
            metadata.query_start_loc[1] = length
            metadata.num_tokens.fill_(length)
            metadata.num_seqs.fill_(1)
            metadata.convolution_slots.fill_(1)
            counts.fill_(count)
            graph.replay()
            expected = torch.cat((expected, source[:count].T), dim=1)[:, -3:]
            torch.testing.assert_close(history[1, :, :3], expected, rtol=0, atol=0)
            assert torch.equal(pool.backing[0], original[0])
            assert torch.equal(pool.backing[2], original[2])
            assert pool.region("sentinel").eq(31).all()
        before = pool.backing.clone()
        metadata.query_start_loc.zero_()
        metadata.num_tokens.zero_()
        metadata.convolution_slots.fill_(-1)
        rows.zero_()
        counts.zero_()
        graph.replay()
        assert torch.equal(pool.backing, before)
        rows.fill_(1)

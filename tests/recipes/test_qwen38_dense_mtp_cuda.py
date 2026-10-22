"""Complete dense MTP history across shifted-page restore and width transitions."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.paged_batch import BatchInputs
from ignition.components.state_transfer import StateCommands
from ignition.preparation.package_storage import PackageStorage
from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig, prepare
from ignition.recipes.qwen38_dense.pool import ContinuationPool

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an admitted CUDA device",
)


@pytest.fixture(scope="module")
def dense_model():
    snapshot = Path(
        os.environ.get(
            "FLARE_QWEN38_DENSE_SNAPSHOT", "/data/models/Qwen3.8-27B-NVFP4-all"
        )
    )
    if not snapshot.is_dir():
        pytest.skip("requires a dense Qwen checkpoint retaining its native MTP layer")
    storage = PackageStorage()
    with storage:
        model = prepare(
            snapshot,
            PreparationConfig(
                mtp=True,
                speculative_proposal_source="mtp",
                # One maximum-width catalog: the allocator initializes once per process.
                speculative_widths=tuple(range(1, 9)),
                context=1024,
                prefill_chunk_size=512,
                max_active=1,
                snapshot_budget_bytes=190 << 20,
                headroom_bytes=2 << 30,
                compile_decoder=False,
                compile_workers=1,
            ),
            package_storage=storage,
        )
    try:
        import json

        if destination := os.environ.get("FLARE_QUALIFICATION_OUTPUT"):
            Path(destination).mkdir(parents=True, exist_ok=True)
            (Path(destination) / "memory-report-drafts7.json").write_text(
                json.dumps(model.memory_report, indent=2, default=str) + "\n"
            )
        yield model, snapshot
    finally:
        model.close()
        for region in storage.host_regions:
            region.close()


def clone_request(req):
    return {field: getattr(req, field).clone() for field in req.__dataclass_fields__}


def restore_request(req, values):
    for field, value in values.items():
        getattr(req, field).copy_(value)


def draft_prefix(pool, tables, tokens, model_config):
    pages = (tokens + 127) // 128
    return (
        tuple(
            torch.cat(
                [cache[page].view(torch.uint8) for page in tables[0, :pages].tolist()]
            )[:tokens].clone()
            for cache in pool.draft_attention(model_config)
        )
        if tokens
        else ()
    )


@pytest.mark.parametrize(
    "prefix", (1, 2, 127, 128, 129, 130, 255, 256, 257, 258, 383, 384, 385)
)
@torch.no_grad()
def test_native_mtp_restores_complete_prefix_and_repeats_width_transitions(
    dense_model, prefix
):
    from tokenizers import Tokenizer

    model, snapshot = dense_model
    req = next(x for x in model.resources if isinstance(x, DeviceRequestState))
    inp = next(x for x in model.resources if isinstance(x, BatchInputs))
    pool = next(x for x in model.resources if isinstance(x, ContinuationPool))
    commands = next(x for x in model.resources if isinstance(x, StateCommands))
    round = model.forwards["speculative1w4"]
    owner = round.owner
    model_config = owner.weights.config
    transfer = model.forwards["restore1-spec"].state.transfer
    original_ids = (
        Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        .encode(
            "A telescope collects photons and a spectrometer separates wavelengths. A transaction makes related changes visible together. Software tests preserve failed examples and measure numerical differences. Engineers inspect page ownership, pending tokens, recurrent histories, causal masks, and stochastic sampling. "
            * 20
        )
        .ids
    )
    pool.clear_for_preparation()
    commands.num_ops.zero_()
    for tensor in vars(owner.rng).values():
        tensor.zero_()
    owner.draft_generation.zero_()
    owner.draft_offset.zero_()
    prompt = original_ids[:prefix]
    req.admit(
        0, RequestAdmission(generation=prefix + 1, seed=42, max_new_tokens=128), prompt
    )
    inp.request_state_rows.zero_()
    inp.request_state_rows[0, 0] = 1
    inp.row_slot_ids[0] = 0
    inp.row_generations[0] = prefix + 1
    inp.live_rows.fill_(1)
    inp.query_lengths.fill_(prefix)
    inp.sample_enabled.fill_(1)
    inp.page_tables.fill_(-1)
    inp.page_tables[0, :5] = torch.arange(5, device="cuda", dtype=torch.int32)
    owner.draft_tables.fill_(-1)
    owner.draft_tables[0, :4] = torch.arange(1, 5, device="cuda", dtype=torch.int32)
    inp.token_ids[:prefix] = torch.tensor(prompt, device="cuda")
    model.graphs["prefill512-spec"].replay()
    torch.cuda.synchronize()
    assert req.error_code.item() == 0 and req.consumed_tokens.item() == prefix
    assert pool.region("consumed_tokens")[1].item() == prefix
    saved_req = clone_request(req)
    saved_rng = {name: value.clone() for name, value in vars(owner.rng).items()}
    saved_draft = (owner.draft_generation.clone(), owner.draft_offset.clone())
    expected_cache = draft_prefix(pool, owner.draft_tables, prefix - 1, model_config)
    commands.request_slots[0] = 0
    commands.source_slots[0] = 1
    commands.destination_slots[0] = 2
    commands.operation[0] = 0
    commands.expected_consumed[0] = prefix
    commands.num_ops.fill_(1)
    model.graphs["snapshot1-spec"].replay()
    torch.cuda.synchronize()
    assert commands.errors.item() == 0
    snapshot_bytes = pool.backing[2].clone()
    expected_regions = {
        r.name: pool.region(r.name)[1].clone()
        for r in pool.layout.regions
        if r.dtype != torch.uint8
    }

    def sequence():
        captures = []
        null_state = pool.backing[0].clone()
        for width in (owner.width, 1, 3, 2, owner.width):
            inp.query_lengths.fill_(width)
            model.graphs[f"speculative1w{width}"].replay()
            torch.cuda.synchronize()
            op = model.forwards[f"speculative1w{width}"]
            assert req.error_code.item() == 0
            assert torch.equal(pool.backing[0], null_state), (
                "null state mutated",
                width,
            )
            captures.append(
                (
                    op.probabilities.clone(),
                    req.pending_token.clone(),
                    req.consumed_tokens.clone(),
                    op.operations[-1].emitted_count.clone(),
                )
            )
        return captures

    if prefix == 129:
        # Both arms consume the same pending input. Sampling uses different RNG
        # domains, so compare the target scores and histories before that choice.
        inp.query_lengths.fill_(1)
        model.graphs["decode1"].replay()
        torch.cuda.synchronize()
        ordinary = model.forwards["decode1"].decoder.scratch.logits.clone()
        ordinary_state = {
            r.name: pool.region(r.name)[1].clone()
            for r in pool.layout.regions
            if r.dtype != torch.uint8 and r.name != "consumed_tokens"
        }
        commands.source_slots[0], commands.destination_slots[0] = 2, 1
        transfer.validate()
        transfer.apply()
        torch.cuda.synchronize()
        assert commands.errors.item() == 0
        restore_request(req, saved_req)
        model.graphs["speculative1w1"].replay()
        torch.cuda.synchronize()
        assert req.error_code.item() == 0
        torch.testing.assert_close(
            model.forwards["speculative1w1"].decoder.scratch.logits,
            ordinary,
            rtol=0,
            atol=0,
        )
        for name, value in ordinary_state.items():
            torch.testing.assert_close(pool.region(name)[1], value, rtol=0, atol=0)
        transfer.validate()
        transfer.apply()
        torch.cuda.synchronize()
        assert commands.errors.item() == 0
        restore_request(req, saved_req)
        for name, value in saved_rng.items():
            getattr(owner.rng, name).copy_(value)
        owner.draft_generation.copy_(saved_draft[0])
        owner.draft_offset.copy_(saved_draft[1])

    cold = sequence()
    assert torch.equal(pool.backing[2], snapshot_bytes), "immutable checkpoint mutated"
    # Retain shared pages and replace the private target/draft suffix, as Helios does.
    boundary = prefix // 128
    branch_pages = list(range(boundary)) + [
        page for page in (4, 5, 2, 3, 0, 1) if page >= boundary
    ]
    inp.page_tables.fill_(-1)
    inp.page_tables[0, :6] = torch.tensor(
        branch_pages, device="cuda", dtype=torch.int32
    )
    owner.draft_tables.fill_(-1)
    owner.draft_tables[0, :5].copy_(inp.page_tables[0, 1:6])
    commands.source_slots[0] = 2
    commands.destination_slots[0] = 1
    commands.expected_consumed[0] = prefix
    transfer.validate()
    transfer.apply()
    torch.cuda.synchronize()
    assert commands.errors.item() == 0
    for name, value in expected_regions.items():
        assert torch.equal(pool.region(name)[1], value), ("state restore", prefix, name)
    restored_cache = draft_prefix(pool, owner.draft_tables, prefix - 1, model_config)
    assert all(torch.equal(a, b) for a, b in zip(expected_cache, restored_cache)), (
        "draft KV restore",
        prefix,
    )
    restore_request(req, saved_req)
    for name, value in saved_rng.items():
        getattr(owner.rng, name).copy_(value)
    owner.draft_generation.copy_(saved_draft[0])
    owner.draft_offset.copy_(saved_draft[1])
    restored = sequence()
    for index, (a, b) in enumerate(zip(cold, restored)):
        for field, (x, y) in enumerate(zip(a, b)):
            assert torch.equal(x, y), ("round mismatch", prefix, index, field)
    assert torch.equal(pool.backing[2], snapshot_bytes), (
        "restoration mutated immutable checkpoint"
    )


@pytest.mark.parametrize("budget", (1, 2, 3, 4, 5))
@torch.no_grad()
def test_native_mtp_clips_consumed_state_at_output_budget_and_leaves_terminal_rows_unchanged(
    dense_model, budget
):
    model, _ = dense_model
    requests = next(x for x in model.resources if isinstance(x, DeviceRequestState))
    inputs = next(x for x in model.resources if isinstance(x, BatchInputs))
    pool = next(x for x in model.resources if isinstance(x, ContinuationPool))
    owner = model.forwards["speculative1w4"].owner
    pool.clear_for_preparation()
    prefix, generation = 127, 1000 + budget
    # Token identity does not affect the clipping invariant; force a page crossing.
    prompt = [42] * prefix
    requests.admit(
        0,
        RequestAdmission(generation=generation, seed=42, max_new_tokens=budget),
        prompt,
    )
    inputs.request_state_rows.zero_()
    inputs.request_state_rows[0, 0] = 1
    inputs.row_slot_ids[0] = 0
    inputs.row_generations[0] = generation
    inputs.live_rows.fill_(1)
    inputs.query_lengths.fill_(prefix)
    inputs.sample_enabled.fill_(1)
    inputs.token_ids[:prefix].fill_(42)
    inputs.page_tables.fill_(-1)
    inputs.page_tables[0, :5] = torch.arange(5, device="cuda", dtype=torch.int32)
    owner.draft_tables.fill_(-1)
    owner.draft_tables[0, :4] = torch.arange(1, 5, device="cuda", dtype=torch.int32)
    model.graphs["prefill128-spec"].replay()
    torch.cuda.synchronize()
    for _ in range(budget):
        if requests.status.item() == 2:
            break
        inputs.query_lengths.fill_(4)
        model.graphs["speculative1w4"].replay()
        torch.cuda.synchronize()
        assert requests.error_code.item() == 0
        assert requests.generated_tokens.item() <= budget
        assert (
            requests.consumed_tokens.item()
            == prefix + requests.generated_tokens.item() - 1
        )
    assert requests.status.item() == 2
    assert requests.generated_tokens.item() == budget
    assert requests.consumed_tokens.item() == prefix + budget - 1
    assert pool.region("consumed_tokens")[1].item() == requests.consumed_tokens.item()
    before = pool.backing.clone()
    saved_requests = clone_request(requests)
    saved_rng = {name: value.clone() for name, value in vars(owner.rng).items()}
    saved_draft = (owner.draft_generation.clone(), owner.draft_offset.clone())
    pages = int(pool.memory_capacity[1].item())
    caches = {name: tensor[:pages].clone() for name, tensor in pool.pages.items()}
    for width in (4, 1, 3):
        inputs.query_lengths.fill_(width)
        model.graphs[f"speculative1w{width}"].replay()
        torch.cuda.synchronize()
        assert torch.equal(pool.backing, before)
        for field, value in saved_requests.items():
            assert torch.equal(getattr(requests, field), value), field
        for name, value in saved_rng.items():
            assert torch.equal(getattr(owner.rng, name), value), name
        assert torch.equal(owner.draft_generation, saved_draft[0])
        assert torch.equal(owner.draft_offset, saved_draft[1])
        for name, value in caches.items():
            assert torch.equal(pool.pages[name][:pages], value), name


@pytest.mark.parametrize("prompt_kind", ("pattern", "prose", "code"))
@torch.no_grad()
def test_target_width_repeatability_diagnostics(dense_model, tmp_path, prompt_kind):
    """Measure cross-width rounding separately from repeatability at each width."""
    import json

    model, snapshot = dense_model
    requests = next(x for x in model.resources if isinstance(x, DeviceRequestState))
    inputs = next(x for x in model.resources if isinstance(x, BatchInputs))
    pool = next(x for x in model.resources if isinstance(x, ContinuationPool))
    owner = model.forwards["speculative1w4"].owner
    pool.clear_for_preparation()
    prefix, generation = 129, 2001
    prompt = [42, 100, 1000] * 43
    if prompt_kind != "pattern":
        from tokenizers import Tokenizer

        text = (
            "A GPU benchmark measures completed tokens, latency, context length, and memory. "
            "A database transaction preserves related state across rollback and replay. "
            if prompt_kind == "prose"
            else "def lookup(cache: dict, key: str):\n    if key not in cache:\n        raise KeyError(key)\n    return cache[key]\n"
        )
        prompt = (
            Tokenizer.from_file(str(snapshot / "tokenizer.json"))
            .encode(text * 20)
            .ids[:prefix]
        )
    requests.admit(
        0, RequestAdmission(generation=generation, seed=42, max_new_tokens=128), prompt
    )
    inputs.request_state_rows.zero_()
    inputs.request_state_rows[0, 0] = 1
    inputs.row_slot_ids[0] = 0
    inputs.row_generations[0] = generation
    inputs.live_rows.fill_(1)
    inputs.query_lengths.fill_(prefix)
    inputs.sample_enabled.fill_(1)
    inputs.token_ids[:prefix] = torch.tensor(prompt, device="cuda")
    inputs.page_tables.fill_(-1)
    inputs.page_tables[0, :5] = torch.arange(5, device="cuda", dtype=torch.int32)
    owner.draft_tables.fill_(-1)
    owner.draft_tables[0, :4] = torch.arange(1, 5, device="cuda", dtype=torch.int32)
    model.graphs["prefill256-spec"].replay()
    torch.cuda.synchronize()
    assert requests.error_code.item() == 0
    saved_state = pool.backing[1].clone()
    pages = int(pool.memory_capacity[1].item())
    saved_pages = {name: tensor[:pages].clone() for name, tensor in pool.pages.items()}
    candidates = torch.tensor(
        [[requests.pending_token.item(), *([42, 100, 1000] * 3)[: owner.width - 1]]],
        device="cuda",
    )

    def trace(width):
        pool.backing[1].copy_(saved_state)
        for name, value in saved_pages.items():
            pool.pages[name][:pages].copy_(value)
        decoder = model.forwards[f"speculative1w{width}"].decoder
        decoder.metadata.set(
            torch.tensor([prefix], device="cuda"),
            torch.tensor([width], device="cuda"),
            inputs.request_state_rows[:1, 0],
            inputs.page_tables[:1],
            candidates[:, :width],
            inputs.live_rows,
        )
        captured, handles = {}, []

        def hook(index, kind):
            def capture(module, arguments, output):
                value = output[0] if isinstance(output, tuple) else output
                captured[f"{index}.{kind}"] = value[0].clone()

            return capture

        for index, layer in enumerate(decoder.layers):
            handles.append(layer.register_forward_hook(hook(index, "residual")))
            attention = (
                layer.linear_attn
                if layer.block_type == "linear_attention"
                else layer.self_attn
            )
            handles.append(
                attention.register_forward_hook(hook(index, layer.block_type))
            )
        try:
            decoder()
            torch.cuda.synchronize()
            captured["logits"] = decoder.scratch.logits[0].clone()
        finally:
            for handle in handles:
                handle.remove()
        return captured

    baseline, report = None, []
    for width in range(1, owner.width + 1):
        first, repeated = trace(width), trace(width)
        assert first.keys() == repeated.keys()
        for name, value in first.items():
            assert torch.isfinite(value).all(), (width, name)
            assert torch.equal(value, repeated[name]), ("repeatability", width, name)
        if baseline is None:
            baseline = first
        differences = {
            name: float(
                (value.float() - baseline[name].float()).norm()
                / baseline[name].float().norm().clamp_min(1e-12)
            )
            for name, value in first.items()
        }
        p, q = (
            baseline["logits"].float().log_softmax(-1),
            first["logits"].float().log_softmax(-1),
        )
        report.append(
            dict(
                width=width,
                repeat_bit_exact=True,
                relative_l2=differences,
                kl_from_width_one=float((p.exp() * (p - q)).sum()),
                top_one_matches_width_one=bool(
                    first["logits"].argmax() == baseline["logits"].argmax()
                ),
            )
        )
    destination = Path(os.environ.get("FLARE_QUALIFICATION_OUTPUT", str(tmp_path)))
    destination.mkdir(parents=True, exist_ok=True)
    (
        destination
        / f"target-width-repeatability-{prompt_kind}-drafts{owner.width - 1}.json"
    ).write_text(json.dumps(report, indent=2) + "\n")

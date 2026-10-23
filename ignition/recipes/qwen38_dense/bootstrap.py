"""Prepare a single-GPU dense Qwen text graph catalog with device-resident state."""

import contextlib
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.components.capacity import (batch_capacities, prefill_capacities, request_capacity,
    speculative_shapes, speculative_token_capacity)
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import PAGE_TOKENS
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights


@dataclass(frozen=True)
class PreparationConfig:
    draft_checkpoint: Path | None = None
    history_drafts: bool = False
    mtp: bool = False
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    speculative_proposal_source: str = "mtp"
    prompt_logprobs: bool = False
    compile_workers: int | None = None
    tuning_replica: TuningReplica | None = None
    device: int = 0
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None

    def __post_init__(self):
        request_capacity(self.max_active)
        if self.speculative_proposal_source not in ("mtp", "dflash2"):
            raise ValueError("dense speculative decoding requires MTP or DFlash2")
        if (self.speculative_proposal_source == "dflash2") != (self.draft_checkpoint is not None):
            raise ValueError("DFlash2 requires an explicit draft checkpoint")
        if self.history_drafts:
            raise ValueError("dense learned proposals do not yet combine history drafts")
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed width must belong to the prepared widths")
        if (not self.speculative_widths or self.speculative_widths[0] != 1
                or tuple(sorted(set(self.speculative_widths))) != self.speculative_widths
                or self.speculative_widths[-1] > 8):
            raise ValueError("speculative widths must be increasing, include one and fit eight columns")
        if not PAGE_TOKENS <= self.context <= 262144 or self.context % PAGE_TOKENS:
            raise ValueError("Qwen dense context must be 128-aligned in [128,262144]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Qwen dense prefill chunk must be 128-aligned and fit the context")
        if self.mtp and self.speculative_proposal_source == "mtp" and self.prefill_chunk_size > 4096:
            raise ValueError("dense MTP prefill chunks must fit the budgeted 4096-token workspace")
        speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")


@torch.no_grad()
def prepare(snapshot, config, *, progress=None, package_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    device = torch.device("cuda", config.device)
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, progress=progress)
    release_unused()
    drafts = max(config.speculative_widths) - 1 if config.mtp else 0
    learned = config.mtp and config.speculative_proposal_source == "mtp"
    parallel = config.mtp and config.speculative_proposal_source == "dflash2"
    draft_weights = None
    if learned:
        from .mtp_weights import MTPWeights
        draft_weights = MTPWeights.load(manifest, weights.config, device, progress=progress)
    if parallel:
        from dataclasses import replace
        from .dflash2_config import DFlash2Config
        from .dflash2_weights import DFlash2Weights
        weights = replace(weights, config=replace(weights.config, draft=DFlash2Config.load(config.draft_checkpoint, weights.config)))
        draft_weights = DFlash2Weights.load(config.draft_checkpoint, weights.config.draft, device, progress=progress)
        weights = replace(weights, draft=draft_weights)
    layout = continuation_layout(weights.config, speculative_tokens=drafts, draft=learned)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Qwen dense continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    state_rows = slots + snapshots
    pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    total_memory = torch.cuda.mem_get_info(device)[1]
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, state_rows + 1, pages, device,
            speculative_tokens=drafts, draft=learned,
            storage=package_storage, maximum_pages=total_memory // page_bytes(weights.config, draft=learned),
            maximum_rows=total_memory // layout.row_bytes, backed_slots=min(state_rows + 1, 1 + 2 * slots))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max(config.prefill_chunk_size, slots), blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24 if config.mtp else 8), device=device)
        coordinator = None
        identity = hashlib.sha256(json.dumps({
            "config": weights.config.text, "headers": [s.header_sha256 for s in manifest.shards],
            **({"draft_config": weights.config.draft.text, "draft_payload_sha256": draft_weights.checkpoint_sha256} if parallel else {}),
            "contract": "qwen38-dense.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, tuning_replica=config.tuning_replica,
            device=device, checkpoint=identity,
            workload_schema="qwen38-dense", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device)
        session.register_scratch(arena)
        prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, weights.config, device)
        decode_tokens = max((b * w for b, w in speculative_shapes(config.max_active, config.speculative_widths, config.speculative_token_capacity)), default=slots) if config.mtp else slots
        decode_storage = Scratch.allocate(decode_tokens, decode_tokens, weights.config, device)
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
    ordinary_weights = weights
    if parallel:
        ordinary_weights = replace(weights, draft=None)
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame)
    forwards, signatures, metadata = {}, {}, {}
    absorptions = []
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    speculation = None
    if config.mtp:
        from .speculative import SpeculativePreparation
        if parallel:
            from .dflash2 import DFlashPreparation as SpeculativePreparation
        speculation = SpeculativePreparation(weights, pool, requests, inputs, decode_storage,
            completion, frame, sequence, width=drafts + 1, session=session, workspace=arena, draft_weights=draft_weights)
        if learned:
            buffers["draft_page_tables"] = speculation.draft_tables
    count = prefill_graph_count(prefill_sizes, scoring) + 5 * len(batch_sizes)
    if config.mtp:
        count += (prefill_graph_count(prefill_sizes, scoring) + 4 * len(batch_sizes)
                  + len(speculative_shapes(config.max_active, config.speculative_widths, config.speculative_token_capacity)))
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=False):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens,
                (*tuple(fields), *(("draft_page_tables",) if learned and enabled else ())),
                speculation_enabled=enabled)
            stage.update(advance=1)

        for kind, mode, sizes in prefill_modes(prefill_sizes, reversed(batch_sizes), scoring):
            for capacity in sizes:
                rows = 1 if mode == "prefill" else capacity
                name = f"{kind}{capacity}"
                m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                decoder = PreparedDecoder(ordinary_weights, scratch, m, pool, mode=mode,
                    scoring=scoring if kind == "scored_prefill" else None,
                    vocabulary_runtime=None, coordinator=coordinator, session=session, name=name, workspace=arena)
                batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool)
                forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=None,
                    coordinator=coordinator, session=session, name=name + ".generation", workspace=arena)
                add(name, forward, kind, rows, capacity if mode == "prefill" else 1,
                    (*inputs.buffers(), "sampling_grammar_mask"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for capacity in reversed(batch_sizes):
            name = f"resume{capacity}"
            forward = PreparedResume(ordinary_weights, requests, pool, inputs, completion, frame, sequence,
                rows=capacity, runtime=None, vocabulary_runtime=None, coordinator=coordinator,
                session=session, name=name, activation_mode="auto")
            add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"))
            for mode in ("init", "restore", "snapshot"):
                name = f"{mode}{capacity}"
                forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                    runtime=None, coordinator=coordinator, session=session, name=name, capacity=capacity,
                    admission_buffers=admission, draft_state=False)
                fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()))
        if speculation is not None:
            for name in tuple(forwards):
                signature = signatures[name]
                if signature.kind in ("prefill", "scored_prefill", "resume"):
                    from dataclasses import replace
                    forward = forwards[name]
                    if (learned or parallel) and signature.kind in ("prefill", "scored_prefill"):
                        from .mtp import PrefillAbsorption
                        if parallel:
                            from .dflash2 import PrefillContext as PrefillAbsorption
                        generation = forward.forward if signature.kind == "scored_prefill" else forward
                        absorption = PrefillAbsorption(speculation, generation.decoder, name)
                        absorptions.append(absorption)
                        forward = PreparedGeneration(absorption,
                            generation.batch, completion, frame, sequence, runtime=None,
                            coordinator=coordinator, session=session, name=name + ".spec_generation", workspace=arena)
                        if signature.kind == "scored_prefill":
                            forward = PreparedScoredPrefill(forward, requests, inputs)
                    forwards[name + "-spec"] = forward
                    signatures[name + "-spec"] = replace(signature, speculation_enabled=True,
                        inputs=(*signature.inputs, *(("draft_page_tables",) if learned else ())))
                    stage.update(advance=1)
            for batch in reversed(batch_sizes):
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{batch}-spec"
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=None, coordinator=coordinator, session=session, name=name, capacity=batch,
                        admission_buffers=admission, draft_state=learned or parallel,
                        draft_tables=speculation.draft_tables if learned else None)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, batch, 1, (*fields, "page_tables", *commands.buffers()), enabled=True)
            for batch, width in speculative_shapes(config.max_active, config.speculative_widths, config.speculative_token_capacity):
                name = f"speculative{batch}w{width}"
                forward = speculation.round(batch, width, name)
                forward.prepare()
                forwards[name] = forward
                signatures[name] = GraphSignature("speculative", batch, width,
                    (*inputs.buffers(), *(("draft_page_tables",) if learned else ()), "sampling_grammar_mask"), speculation_enabled=True)
                stage.update(advance=1)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
        stream=stream, pool=graph_pool, collective=None, coordinator=coordinator,
        compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
        retain_graphs=config.retain_graphs, materialize=pool.clear_for_preparation, warmup_storage=package_storage,
        compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative"))
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    assets = {role: {"path": str(snapshot / file), "sha256": hashlib.sha256((snapshot / file).read_bytes()).hexdigest()}
              for role, file in (("tokenizer", "tokenizer.json"), ("chat_template", "chat_template.jinja"))}
    eos = weights.config.text["eos_token_id"]
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text())
    eos_text = tokenizer["eos_token"]
    if isinstance(eos_text, dict):
        eos_text = eos_text["content"]
    from tokenizers import Tokenizer
    tokenizer_eos = Tokenizer.from_file(str(snapshot / "tokenizer.json")).token_to_id(eos_text)
    eos_ids = list(dict.fromkeys([*(eos if isinstance(eos, list) else [eos]), tokenizer_eos]))
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=prepare_frontend(snapshot, eos_token_ids=eos_ids),
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary, draft_lag=0 if parallel else None))
    if speculation is not None:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width,
            speculative_proposal_source="mtp" if parallel else config.speculative_proposal_source, history_drafts=config.history_drafts)
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=0, world_size=1, device=config.device, weights_bytes=weights.allocated_bytes + (0 if draft_weights is None else draft_weights.allocated_bytes),
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config, draft=learned), page_tokens=PAGE_TOKENS,
        kv_bytes=pages * page_bytes(weights.config, draft=learned), completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    if speculation is not None:
        if parallel:
            memory.update(speculative_round_scratch_bytes=speculation.storage.nbytes + speculation.layer_buffers.nbytes,
                draft_context_scratch_bytes=speculation.context_storage.nbytes)
        else:
            memory.update(speculative_sampling_scratch_bytes=speculation.storage.nbytes,
                mtp_prefill_scratch_bytes=sum(a.scratch_bytes for a in absorptions))
        memory["gdn_verification_record_bytes"] = sum(t.nbytes for t in
            (speculation.records.delta, speculation.records.keys, speculation.records.decay))
    if scoring is not None:
        memory.update(scoring.memory_report())
    return PreparedModel(forwards={name: compiled.get(name, forward) for name, forward in forwards.items()},
        rebuild_weights={"weights": weights, **({"mtp" if learned else "draft": draft_weights} if learned or parallel else {})}, device=config.device, graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        close_resources=(() if scoring is None else (scoring,)),
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges([prefill_storage, decode_storage, arena.tensor, metadata,
                               *((speculation.storage,) if speculation is not None else ()),
                               *((speculation.layer_buffers, speculation.context_storage) if parallel else ())]),
                           *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage,
                   decode_storage, metadata, compiled, stream, graph_pool, session, coordinator, speculation))

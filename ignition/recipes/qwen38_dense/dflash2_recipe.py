"""Qwen dense preparation with pinned parallel DFlash2 drafters."""

from dataclasses import replace
from functools import partial
from types import MappingProxyType

from ignition.recipes.declaration import RankCosts, Speculation
from .recipe import RECIPE as TARGET_RECIPE, usage
from .dflash2_config import NVFP4_REPOSITORY, NVFP4_REVISION


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None, draft_repository=None, draft_revision=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("dense DFlash2 preparation requires one local GPU")
    # Bound the retained private context projections to the declared cost model.
    if plan.prefill_chunk > 4096:
        raise ValueError("dense DFlash2 currently prepares prefill chunks through 4096 tokens")
    drafting = {}
    if plan.speculation is not None:
        from huggingface_hub import snapshot_download
        from .dflash2_config import REPOSITORY, REVISION

        draft = snapshot_download(draft_repository or REPOSITORY, revision=draft_revision or REVISION, cache_dir=plan.hub_cache,
            allow_patterns=("config.json", "model.safetensors"))
        speculation = plan.speculation
        drafting = dict(mtp=True, draft_checkpoint=draft, speculative_proposal_source="dflash2",
            speculative_widths=speculation.widths, speculative_fixed_width=speculation.fixed_width,
            speculative_token_capacity=plan.speculative_token_capacity, history_drafts=speculation.history_drafts)
    config = PreparationConfig(**drafting, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, tuning_replica=plan.tuning_replica,
        max_active=plan.concurrency, context=plan.context, prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache)
    return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)


# Checkpoint BF16 tensors plus the retained stacked context-K/V projection.
_DRAFT_BYTES = 3_848_808_960 + 104_857_600
# One largest-capacity context cursor, shared by serialized prefill/scored graphs.
# Includes chunk rounding beyond the 4096-token taps, projections and rotated KV.
_CONTEXT_BYTES = 4096 * (9 * 5120 * 2 + 4 * 1024 + 8) + (16 << 20)
_STATE_BYTES = 158_148_864 + 7 * 983_040 + 40 * (1 << 20)

RECIPE = replace(TARGET_RECIPE, name="qwen38_dense_dflash2", prepare=prepare, usage=usage,
    checkpoints=tuple(replace(c, alias="qwen3.8-27b-dflash2",
        device_bytes=c.device_bytes - c.draft_bytes + _DRAFT_BYTES, draft_bytes=_DRAFT_BYTES)
        for c in TARGET_RECIPE.checkpoints),
    context=32768, concurrency=1, max_concurrency=1, prefill_chunk=512,
    speculation=Speculation(draft_tokens=7, adaptive=False),
    ranks=MappingProxyType({1: RankCosts(_STATE_BYTES, 4_194_304,
        (4 << 30) + _CONTEXT_BYTES, request_scratch_bytes=128 << 20)}))


# The calibrated community checkpoint is a separate, explicit precision choice.
# Its 35 NVFP4 projections remain independent; no scales or weights are merged.
_NVFP4_DRAFT_BYTES = 1_550_134_040 + (2 << 20)
NVFP4_RECIPE = replace(RECIPE, name="qwen38_dense_dflash2_nvfp4",
    prepare=partial(prepare, draft_repository=NVFP4_REPOSITORY, draft_revision=NVFP4_REVISION),
    checkpoints=tuple(replace(c, alias="qwen3.8-27b-dflash2-nvfp4",
        device_bytes=c.device_bytes - c.draft_bytes + _NVFP4_DRAFT_BYTES,
        draft_bytes=_NVFP4_DRAFT_BYTES) for c in RECIPE.checkpoints))

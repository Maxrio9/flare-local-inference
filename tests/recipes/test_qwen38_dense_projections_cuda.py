"""Actual NVFP4 checkpoint projections into the GDN's interleaved scratch views."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an admitted CUDA device"
)


@pytest.mark.parametrize(
    "tokens,mode", [(1, "decode"), (4, "decode"), (128, "prefill"), (4096, "prefill")]
)
@torch.no_grad()
def test_nvfp4_gdn_projections_preserve_neighbouring_scratch_columns(tokens, mode):
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from ignition.components.gated_delta import GDNScratch, GDNState, PreparedGDN
    from ignition.components.linear import QuantizedProjection
    from ignition.components.model import canonical_tree
    from ignition.components.recurrent import DeltaMetadata
    from ignition.components.tensor_fp8 import TensorFP8Weight
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.qwen38_dense.config import ModelConfig, TEXT_PREFIX

    snapshot = Path(
        os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT", "/data/models/Qwen3.8-27B-NVFP4-all")
    )
    if not snapshot.is_dir():
        pytest.skip("requires the all-NVFP4 dense checkpoint")
    config, device = ModelConfig.load(snapshot), torch.device("cuda:0")
    manifest = inspect_indexed_checkpoint(snapshot)
    tree = canonical_tree(Qwen3_5TextModel, Qwen3_5TextConfig.from_dict(config.text))
    module = tree.layers[0].linear_attn
    prefix = TEXT_PREFIX + ".layers.0.linear_attn."
    with DirectWeightLoader(manifest, device, prefixes=(prefix,)) as loader:
        projections = {
            name: loader.load_linear(prefix + name)
            for name in ("in_proj_qkv", "in_proj_z", "out_proj")
        }
        if any(
            isinstance(projections[name], TensorFP8Weight) for name in ("in_proj_qkv", "in_proj_z")
        ):
            pytest.skip("checkpoint uses tensor FP8 for the projections under test")
        for name in ("A_log", "dt_bias", "conv1d.weight", "norm.weight"):
            parent, _, field = name.rpartition(".")
            setattr(
                module.get_submodule(parent),
                field,
                torch.nn.Parameter(loader.load_tensor(prefix + name), requires_grad=False),
            )
        projections["in_proj_ba"] = torch.cat(
            tuple(loader.load_tensor(prefix + f"in_proj_{name}.weight") for name in ("b", "a"))
        )
    sequences = tokens if mode == "decode" else 1
    source = torch.randn((tokens, config.hidden), dtype=torch.bfloat16, device=device) * 0.02
    scratch = GDNScratch.allocate(tokens, device, hidden_size=config.hidden)
    state = GDNState.allocate(sequences + (2 if mode == "prefill" else 1), device)
    metadata = DeltaMetadata.allocate(sequences, device)
    with RecordedPreparationSession(
        device=device,
        checkpoint="dense-nvfp4-strided-projections",
        workload_schema="dense-gdn-projections",
        numerical_policy="default",
        compile_workers=1,
    ) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        prepared = PreparedGDN(
            module,
            projections,
            source,
            scratch,
            state,
            metadata,
            mode=mode,
            session=session,
            name="gdn",
            workspace=arena,
            projection_workspace=arena,
            projection_buffers={},
        )
        references = {}
        for name, width in (("in_proj_qkv", 10240), ("in_proj_z", 6144)):
            output = torch.empty((tokens, width), dtype=source.dtype, device=device)
            references[name] = QuantizedProjection(
                source,
                projections[name],
                output,
                session=session,
                workspace=arena,
                name=name + ".contiguous",
                activation_mode="auto",
            )
        session.tune()
        for name, destination, neighbour in (
            ("in_proj_qkv", scratch.projected_qkv, scratch.z),
            ("in_proj_z", scratch.z, scratch.projected_qkv),
        ):
            expected = references[name](source).clone()
            scratch.projected_qkvz.fill_(17)
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                prepared.projections[name](source)
            for _ in range(3):
                scratch.projected_qkvz.fill_(17)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(destination, expected, rtol=0, atol=0)
                assert neighbour.eq(17).all()
            graph.reset()
        session.freeze()

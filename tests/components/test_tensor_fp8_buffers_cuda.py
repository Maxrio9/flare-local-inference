"""Calibrated FP8 projections reuse caller storage across serialized graph work."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an admitted CUDA device"
)


class SerialProjections:
    def __init__(self, operations, sources):
        self.operations, self.sources = operations, sources

    def __call__(self):
        for operation, source in zip(self.operations, self.sources):
            operation(source)


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("tokens", (1, 4, 128, 4096))
@torch.no_grad()
def test_shared_fp8_input_preserves_each_projection_and_strided_outputs(tokens, compiled):
    from ignition.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.tuning import RecordedPreparationSession

    snapshot = Path(os.environ.get("FLARE_QWEN38_DENSE_FP8_SNAPSHOT", "/data/models/Qwen3.8-27B-NVFP4"))
    if not snapshot.is_dir():
        pytest.skip("requires the official dense checkpoint with calibrated FP8 projections")
    manifest = inspect_indexed_checkpoint(snapshot)
    prefix = "model.language_model.layers.0.linear_attn."
    with DirectWeightLoader(manifest, torch.device("cuda:0"), prefixes=(prefix,)) as loader:
        weights = [loader.load_linear(prefix + name) for name in ("in_proj_qkv", "in_proj_z")]
    assert all(isinstance(weight, TensorFP8Weight) for weight in weights)
    sources = [torch.randn((tokens, 5120), device="cuda", dtype=torch.bfloat16) * scale
               for scale in (.02, .09)]
    widths = [weight.packed.out_features for weight in weights]
    destination = torch.full((tokens, sum(widths) + 16), 31., device="cuda", dtype=torch.bfloat16)
    outputs = [destination[:, :widths[0]], destination[:, widths[0]:sum(widths)]]
    references = [torch.empty_like(output) for output in outputs]
    buffers = {}
    with RecordedPreparationSession(device=torch.device("cuda:0"),
            checkpoint="tensor-fp8-shared-input", workload_schema="serialized-projection-buffers",
            numerical_policy="default", compile_workers=1) as session:
        shared = [TensorFP8Projection(source, weight, output, session=session, name=f"shared-{i}", buffers=buffers)
                  for i, (source, weight, output) in enumerate(zip(sources, weights, outputs))]
        private = [TensorFP8Projection(source, weight, output, session=session, name=f"private-{i}")
                   for i, (source, weight, output) in enumerate(zip(sources, weights, references))]
        session.tune()
        assert shared[0].quantized.data_ptr() == shared[1].quantized.data_ptr()
        assert len(buffers) == 1
        assert sum(buffer.nbytes for buffer in buffers.values()) == tokens * 5120
        for operation, source in zip(private, sources):
            operation(source)
        for operation, source in zip(shared, sources):
            operation(source)
        forward = SerialProjections(shared, sources)
        if compiled:
            from ignition.preparation.aot import aot_forward

            artifact = aot_forward(forward)
            execute = lambda: artifact(forward)
        else:
            execute = forward
        execute()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        # Change both inputs after capture, so compiled constants cannot stand in
        # for runtime input quantization. Recompute independent private references.
        sources[0].mul_(2)
        sources[1].neg_()
        for operation, source in zip(private, sources):
            operation(source)
        # The second input differs: retaining the previous projection's quantization
        # would corrupt either its output or the next one. Both must remain correct.
        shared[0].quantized.view(torch.uint8).fill_(0x7f)
        graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(outputs, references):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert destination[:, sum(widths):].eq(31).all()
        before = torch.cuda.memory_stats()
        for _ in range(8):
            graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        assert all(before[key] == after[key] for key in (
            "allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"))

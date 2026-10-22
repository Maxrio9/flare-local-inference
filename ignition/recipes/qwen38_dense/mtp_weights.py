"""Load the checkpoint's dense BF16 predictor without requantizing it."""

from dataclasses import dataclass

import torch

from ignition.components.model import canonical_tree
from ignition.components.weights import DirectWeightLoader


@dataclass(frozen=True)
class MTPWeights:
    layer: torch.nn.Module
    norm: torch.nn.Module
    embedding_norm: torch.nn.Module
    hidden_norm: torch.nn.Module
    feedback: torch.Tensor
    projections: dict
    allocated_bytes: int

    @classmethod
    def load(cls, manifest, config, device, *, progress=None):
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        if (
            config.text.get("mtp_num_hidden_layers") != 1
            or config.text.get("mtp_use_dedicated_embeddings") is not False
        ):
            raise ValueError(
                "dense MTP requires one layer with shared target embeddings"
            )
        text = dict(config.text, num_hidden_layers=1, layer_types=["full_attention"])
        tree = canonical_tree(Qwen3_5TextModel, Qwen3_5TextConfig.from_dict(text))
        layer = tree.layers[0]
        # Dense MTP uses Qwen's zero-centered RMSNorm, including the feedback norms.
        with torch.device("meta"):
            norms = [
                Qwen3_5RMSNorm(config.hidden, eps=config.text["rms_norm_eps"])
                for _ in range(3)
            ]
        projections = {}
        before = torch.cuda.memory_allocated(device)
        with DirectWeightLoader(
            manifest, device, prefixes=("mtp.",), progress=progress
        ) as loader:
            for name, parameter in layer.named_parameters():
                key = "mtp.layers.0." + name
                tensor = loader.load_tensor(key)
                if tensor.shape != parameter.shape or tensor.dtype != torch.bfloat16:
                    raise ValueError(
                        f"{key}: native MTP requires checkpoint BF16 geometry"
                    )
                parent, _, field = name.rpartition(".")
                setattr(
                    layer.get_submodule(parent),
                    field,
                    torch.nn.Parameter(tensor, requires_grad=False),
                )
                if name.endswith("_proj.weight"):
                    projections["layers.0." + name.removesuffix(".weight")] = tensor
            for module, name in zip(
                norms,
                ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"),
                strict=True,
            ):
                tensor = loader.load_tensor(f"mtp.{name}.weight")
                if tensor.shape != (config.hidden,) or tensor.dtype != torch.bfloat16:
                    raise ValueError(
                        f"mtp.{name}: native MTP requires BF16 hidden-width normalization"
                    )
                module.weight = torch.nn.Parameter(tensor, requires_grad=False)
            feedback = loader.load_tensor("mtp.fc.weight")
            if (
                feedback.shape != (config.hidden, 2 * config.hidden)
                or feedback.dtype != torch.bfloat16
            ):
                raise ValueError(
                    "mtp.fc: native MTP requires BF16 [hidden, 2*hidden] feedback"
                )
        return cls(
            layer,
            *norms,
            feedback,
            projections,
            torch.cuda.memory_allocated(device) - before,
        )

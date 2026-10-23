"""DFlash2 BF16 and separately calibrated NVFP4 checkpoint mapping."""

from dataclasses import dataclass
import hashlib
from functools import cached_property
import json

import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from .dflash2_config import DFlash2Config


class DFlash2WeightLoader(DirectWeightLoader):
    @cached_property
    def quantization(self):
        raw = json.loads((self.manifest.snapshot / "config.json").read_text())
        return DFlash2Config(raw).quantization


@dataclass(frozen=True)
class DFlash2Weights:
    projection: torch.Tensor
    hidden_norm: torch.Tensor
    norm: torch.Tensor
    context_kv: object
    layers: tuple[dict, ...]
    selector: dict
    allocated_bytes: int
    checkpoint_sha256: str

    @classmethod
    def load(cls, snapshot, config, device, *, progress=None):
        manifest = inspect_indexed_checkpoint(snapshot)
        c = config.text
        h, i = c["hidden_size"], c["intermediate_size"]
        q, kv = c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"]
        expected = {"fc.weight": (h, len(config.target_layers) * h),
            "hidden_norm.weight": (h,), "norm.weight": (h,),
            "candidate_selector.predecessor_codebook": (c["vocab_size"], 256),
            "candidate_selector.successor_codebook": (c["vocab_size"], 256),
            "candidate_selector.hidden_projection.weight": (256, h)}
        for index in range(config.layers):
            p = f"layers.{index}."
            for name, shape in {"self_attn.q_proj.weight": (q, h), "self_attn.k_proj.weight": (kv, h),
                    "self_attn.v_proj.weight": (kv, h), "self_attn.o_proj.weight": (h, q),
                    "self_attn.q_norm.weight": (c["head_dim"],), "self_attn.k_norm.weight": (c["head_dim"],),
                    "input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
                    "mlp.gate_proj.weight": (i, h), "mlp.up_proj.weight": (i, h), "mlp.down_proj.weight": (h, i)}.items():
                expected[p + name] = shape
            for kind in ("attention_conv", "mlp_conv"):
                expected[p + kind + ".base_kernel"] = (2, 2, h)
                expected[p + kind + ".kernel_projection.weight"] = (4 * h // 16, h)
        logical = dict(expected)
        dtypes = {name: "BF16" for name in expected}
        for prefix in config.quantization:
            rows, columns = logical[prefix + ".weight"]
            expected[prefix + ".weight"] = (rows, columns // 2)
            dtypes[prefix + ".weight"] = "U8"
            expected[prefix + ".weight_scale"] = (rows, columns // 16)
            dtypes[prefix + ".weight_scale"] = "F8_E4M3"
            for field in ("input_scale", "weight_scale_2"):
                expected[prefix + "." + field] = ()
                dtypes[prefix + "." + field] = "F32"
        if set(manifest.tensors) != set(expected):
            raise ValueError("DFlash2 checkpoint tensor names differ from its declared architecture")
        for name, meta in manifest.tensors.items():
            if meta.shape != expected[name] or meta.dtype != dtypes[name]:
                raise ValueError(f"DFlash2 {name} differs from declared shape or dtype")
        digest = hashlib.sha256()
        for path in manifest.shard_paths:
            with path.open("rb") as file:
                for block in iter(lambda: file.read(8 << 20), b""):
                    digest.update(block)
        before = torch.cuda.memory_allocated(device)
        with DFlash2WeightLoader(manifest, device, progress=progress, progress_label="Load DFlash2 weights") as loader:
            def tensor(name):
                value = loader.load_tensor(name)
                if value.dtype != torch.bfloat16 or tuple(value.shape) != expected[name]:
                    raise ValueError(f"DFlash2 requires BF16 {name} with shape {expected[name]}")
                return value
            def linear(prefix):
                if prefix not in config.quantization:
                    return tensor(prefix + ".weight")
                weight = loader.load_linear(prefix)
                if (weight.out_features, weight.in_features) != logical[prefix + ".weight"]:
                    raise ValueError(f"DFlash2 {prefix} projection geometry mismatch")
                return weight
            projection = tensor("fc.weight")
            hidden_norm, norm = tensor("hidden_norm.weight"), tensor("norm.weight")
            layers = []
            for index in range(config.layers):
                p = f"layers.{index}."
                if config.quantization:
                    projections = tuple(linear(p + "self_attn." + name) for name in ("q_proj", "k_proj", "v_proj"))
                    attention = dict(projections=projections, context=projections[1:])
                else:
                    qkv = torch.cat([linear(p + "self_attn." + name) for name in ("q_proj", "k_proj", "v_proj")])
                    attention = dict(qkv=qkv, context=qkv[q:])
                layers.append(dict(**attention,
                    output=linear(p + "self_attn.o_proj"),
                    q_norm=tensor(p + "self_attn.q_norm.weight"), k_norm=tensor(p + "self_attn.k_norm.weight"),
                    input_norm=tensor(p + "input_layernorm.weight"), post_norm=tensor(p + "post_attention_layernorm.weight"),
                    mlp=tuple(linear(p + "mlp." + name) for name in ("gate_proj", "up_proj", "down_proj")),
                    **{kind: dict(base=tensor(p + kind + ".base_kernel"),
                        projection=tensor(p + kind + ".kernel_projection.weight"))
                       for kind in ("attention_conv", "mlp_conv")}))
            selector = {name: tensor("candidate_selector." + name) for name in
                ("predecessor_codebook", "successor_codebook", "hidden_projection.weight")}
        context_kv = (tuple(layer["context"] for layer in layers) if config.quantization
                      else torch.cat([layer["context"] for layer in layers]))
        return cls(projection, hidden_norm, norm, context_kv, tuple(layers), selector,
            torch.cuda.memory_allocated(device) - before, digest.hexdigest())

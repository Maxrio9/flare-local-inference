"""Pinned Qwen3.8-27B DFlash2 backbone and predecessor selector geometry."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

REPOSITORY = "z-lab/Qwen3.8-27B-DFlash2"
REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
NVFP4_REPOSITORY = "maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal"
NVFP4_REVISION = "bd7a934213c47a9e7ef69eef36bb3325f47fd1f1"


@dataclass(frozen=True)
class DFlash2Config:
    text: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target)

    @classmethod
    def from_dict(cls, c, target):
        expected = dict(architectures=["DFlash2DraftModel"], model_type="qwen3",
            hidden_size=target.hidden, vocab_size=target.vocabulary, num_target_layers=len(target.layers),
            hidden_act="silu", attention_bias=False, is_causal=False, tie_word_embeddings=False,
            num_attention_heads=32, num_key_value_heads=8, head_dim=128, num_hidden_layers=5,
            intermediate_size=17408, sliding_window=2048, use_sliding_window=True,
            layer_types=["sliding_attention"] * 5)
        for field, value in expected.items():
            if c.get(field) != value:
                raise ValueError(f"unsupported DFlash2 {field}: {c.get(field)!r}")
        if c.get("attention_sink_bias", False):
            raise ValueError("DFlash2 requires absent attention sinks")
        quant = c.get("quantization_config")
        if quant is not None:
            expected_quant = dict(quant_method="modelopt", quant_algo="NVFP4", group_size=16, kv_cache_quant_algo="FP8")
            if any(quant.get(k) != v for k, v in expected_quant.items()):
                raise ValueError("DFlash2 requires BF16 or calibrated ModelOpt NVFP4 weights")
            ignored = {"fc", "candidate_selector.hidden_projection", *(f"layers.{i}.{kind}.kernel_projection"
                for i in range(5) for kind in ("attention_conv", "mlp_conv"))}
            if (set(quant.get("exclude_modules", ())) != ignored
                    or quant.get("ignore") != quant.get("exclude_modules")):
                raise ValueError("DFlash2 NVFP4 must preserve feedback, selector and convolution projections in BF16")
        d = c["dflash_config"]
        for field, value in dict(block_size=8, conv_group_size=16, conv_kernel_size=2,
                selector_rank=256, selector_top_k=16, target_layer_ids=[5, 19, 33, 47, 61]).items():
            if d.get(field) != value:
                raise ValueError(f"unsupported DFlash2 {field}: {d.get(field)!r}")
        if type(d.get("mask_token_id")) is not int or not 0 <= d["mask_token_id"] < target.vocabulary:
            raise ValueError("DFlash2 mask token must belong to the target vocabulary")
        for field in ("input_embedding_scale", "output_multiplier", "final_logit_softcapping", "rope_scaling"):
            if c.get(field) is not None:
                raise ValueError(f"DFlash2 does not implement {field}")
        rope = c.get("rope_parameters", {})
        if rope.get("rope_type") != "default" or not all(math.isfinite(v) and v > 0 for v in (rope.get("rope_theta", 0), c.get("rms_norm_eps", 0))):
            raise ValueError("DFlash2 requires default full rotary embeddings and positive epsilon")
        return cls(c)

    @property
    def block_size(self):
        return self.text["dflash_config"]["block_size"]

    @property
    def noise_token(self):
        return self.text["dflash_config"]["mask_token_id"]

    @property
    def target_layers(self):
        return tuple(self.text["dflash_config"]["target_layer_ids"])

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def quantization(self):
        if self.text.get("quantization_config") is None:
            return {}
        return {f"layers.{i}.{module}.{name}": dict(quant_algo="NVFP4", group_size=16)
            for i in range(self.layers)
            for module, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                                  ("mlp", ("gate_proj", "up_proj", "down_proj"))) for name in names}

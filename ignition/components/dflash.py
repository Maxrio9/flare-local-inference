"""Shared context projection and sliding-window DFlash backbone operations."""

import torch
import torch.nn.functional as F

from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.state import save_hidden
from ignition.components.window_attention import stage_windows

PAGE_TOKENS = 128


def projection(source, weight, output, *, session, name, workspace):
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name)
    return QuantizedProjection(source, weight, output, session=session, name=name,
        workspace=workspace, activation_mode="auto")


def _norm(source, weight, epsilon):
    values = source.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(source.dtype)


class Rotary:
    def __init__(self, config, device):
        c = config.text
        theta = c["rope_theta"] if "rope_theta" in c else c["rope_parameters"]["rope_theta"]
        self.inverse = 1 / theta ** (torch.arange(0, c["head_dim"], 2,
            dtype=torch.float32, device=device) / c["head_dim"])

    def __call__(self, values, positions):
        angles = positions.clamp_min(0).float()[:, None] * self.inverse[None]
        cosine, sine = angles.cos().to(values.dtype).float()[:, None], angles.sin().to(values.dtype).float()[:, None]
        left, right = values.float().chunk(2, -1)
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(values.dtype)


class PreparedContext:
    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, buffers, retain_anchor=True):
        self.metadata, self.pool = metadata, pool
        c, draft = weights.config.draft, weights.draft
        tokens, hidden = auxiliary.shape[0], weights.config.hidden
        self.window, self.epsilon = c.text["sliding_window"], c.text["rms_norm_eps"]
        self.projected = buffers.zeros((tokens, hidden), torch.bfloat16)
        self.normalized = buffers.zeros((tokens, hidden), torch.bfloat16)
        self.project = projection(auxiliary, draft.projection, self.projected,
            session=session, name=name + ".project", workspace=workspace)
        self.norm = PreparedRMSNorm(self.projected, draft.hidden_norm, self.normalized,
            eps=self.epsilon, session=session, name=name + ".norm")
        shape = (tokens, c.text["num_key_value_heads"], c.text["head_dim"])
        self.key, self.value = (buffers.zeros(shape, torch.bfloat16) for _ in range(2))
        self.kv, self.project_kv, self.layer_projections = None, None, ()
        if isinstance(draft.context_kv, torch.Tensor):
            self.kv = buffers.zeros((tokens, draft.context_kv.shape[0]), torch.bfloat16)
            self.project_kv = UnquantizedProjection(self.normalized, draft.context_kv, self.kv,
                session=session, name=name + ".kv")
        else:
            # Independent calibrated NVFP4 projections retain their own scales.
            self.key_projection = buffers.zeros((tokens, self.key[0].numel()), torch.bfloat16)
            self.layer_projections = tuple((
                projection(self.normalized, key, self.key_projection, session=session,
                    name=f"{name}.{index}.key", workspace=workspace),
                projection(self.normalized, value, self.value.flatten(1), session=session,
                    name=f"{name}.{index}.value", workspace=workspace))
                for index, (key, value) in enumerate(draft.context_kv))
        self.slots = buffers.zeros((tokens,), torch.int64)
        self.rotary = Rotary(c, auxiliary.device)
        self.anchor = pool.reserved_region("draft_anchor") if retain_anchor else None
        self.stages = []
        for index, layer in enumerate(draft.layers):
            writer = PreparedPagedKVWriter(self.key, self.value, pool.reserved_region(f"draft{index}.key"),
                pool.reserved_region(f"draft{index}.value"), self.slots, session=session, name=f"{name}.{index}.write")
            self.stages.append((layer["k_norm"], writer))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        offset, count = m.positions - m.starts[m.requests], counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        self.slots.copy_(torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1))
        self.norm(self.project(auxiliary))
        if self.project_kv is not None:
            self.project_kv(self.normalized)
            chunks = self.kv.view(self.kv.shape[0], len(self.stages), 2, *self.key.shape[1:])
        for index, (norm, writer) in enumerate(self.stages):
            if self.project_kv is None:
                key, value = self.layer_projections[index]
                key(self.normalized)
                value(self.normalized)
                raw_key = self.key_projection.view_as(self.key)
            else:
                raw_key = chunks[:, index, 0]
                self.value.copy_(chunks[:, index, 1])
            self.key.copy_(self.rotary(_norm(raw_key, norm, self.epsilon), m.positions))
            writer()
        if self.anchor is not None:
            last = (m.query_start_loc[:-1] + counts - 1).long().clamp(0, self.normalized.shape[0] - 1)
            save_hidden(self.normalized[last], self.anchor, m.state_rows, counts > 0)


class DraftAttention:
    def __init__(self, weights, source, metadata, pool, config, index, *, session, name, workspace, buffers):
        self.weights, self.metadata = weights, metadata
        c, tokens = config.text, source.shape[0]
        self.window, self.epsilon = c["sliding_window"], c["rms_norm_eps"]
        qh, kh, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.query = buffers.zeros((tokens, qh, dim), torch.bfloat16)
        self.key, self.value = (buffers.zeros((tokens, kh, dim), torch.bfloat16) for _ in range(2))
        self.attended = buffers.zeros(self.query.shape, torch.bfloat16)
        self.projected, self.project, self.projections = None, None, ()
        if "qkv" in weights:
            self.projected = buffers.zeros((tokens, (qh + 2 * kh) * dim), torch.bfloat16)
            self.project = UnquantizedProjection(source, weights["qkv"], self.projected, session=session, name=name + ".qkv")
        else:
            self.raw_q = buffers.zeros((tokens, qh * dim), torch.bfloat16)
            self.raw_k = buffers.zeros((tokens, kh * dim), torch.bfloat16)
            self.raw_v = buffers.zeros((tokens, kh * dim), torch.bfloat16)
            self.projections = tuple(projection(source, weight, out, session=session, name=name + suffix,
                workspace=workspace) for weight, out, suffix in zip(weights["projections"],
                    (self.raw_q, self.raw_k, self.raw_v), (".q", ".k", ".v"), strict=True))
        self.output = projection(self.attended.flatten(1), weights["output"], buffers.zeros(source.shape, torch.bfloat16),
            session=session, name=name + ".output", workspace=workspace)
        self.rotary = Rotary(config, source.device)
        self.history = tuple(pool.reserved_region(f"draft{index}.{field}") for field in ("key", "value"))
        rows, pages = metadata.lengths.numel(), (self.window + config.block_size + PAGE_TOKENS - 1) // PAGE_TOKENS
        self.staged = tuple(buffers.zeros((rows * pages, PAGE_TOKENS, kh, dim), history.dtype)
                            for history in self.history)
        self.pages = torch.arange(rows * pages, device=source.device, dtype=torch.int32).view(rows, pages)
        self.lengths = buffers.zeros((rows,), torch.int32)
        self.attention = PreparedPagedAttention(self.query, *self.staged, self.pages, self.lengths,
            metadata.query_start_loc, self.attended, sink=weights.get("sink"), mode="verify", width=config.block_size,
            window=self.window - 1, causal=False, session=session, workspace=workspace, name=name + ".attention")

    def __call__(self, source):
        m = self.metadata
        if self.project is None:
            for operation in self.projections:
                operation(source)
            q, k, v = self.raw_q, self.raw_k, self.raw_v
        else:
            q, k, v = self.project(source).split((self.query[0].numel(), self.key[0].numel(), self.value[0].numel()), -1)
        self.query.copy_(self.rotary(_norm(q.view_as(self.query), self.weights["q_norm"], self.epsilon), m.positions))
        self.key.copy_(self.rotary(_norm(k.view_as(self.key), self.weights["k_norm"], self.epsilon), m.positions))
        self.value.copy_(v.view_as(self.value))
        stage_windows(self.history[0], self.key, self.staged[0], m)
        stage_windows(self.history[1], self.value, self.staged[1], m)
        self.lengths.copy_(m.starts.clamp(max=self.window).int() + m.lengths)
        self.attention()
        return self.output(self.attended.flatten(1))


class DraftMLP:
    def __init__(self, weights, source, *, session, name, workspace, buffers):
        gate, up, down = weights
        width = gate.shape[0] if isinstance(gate, torch.Tensor) else gate.out_features
        shape = (source.shape[0], width)
        self.intermediate = buffers.zeros(shape, torch.bfloat16)
        self.gate, self.up = (projection(source, weight, buffers.zeros(shape, torch.bfloat16),
            session=session, name=name + suffix, workspace=workspace)
            for weight, suffix in ((gate, ".gate"), (up, ".up")))
        self.down = projection(self.intermediate, down, buffers.zeros(source.shape, torch.bfloat16),
            session=session, name=name + ".down", workspace=workspace)
        self.concurrent_up = (ConcurrentProjectionChain((self.up,), parallel_plans=(self.gate.plan,),
            session=session, workspace=workspace) if workspace.overlap and not isinstance(gate, torch.Tensor) else None)

    def __call__(self, source):
        if self.concurrent_up is not None:
            self.concurrent_up.start(source)
        gate = self.gate(source)
        up = (self.up(source) if self.concurrent_up is None else self.concurrent_up.finish(source, gate))
        self.intermediate.copy_((F.silu(gate.float()) * up.float()).bfloat16())
        return self.down(self.intermediate)

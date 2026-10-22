"""Dense native MTP feedback, causal prediction and target-derived cache repair."""

from copy import copy
from types import SimpleNamespace

import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_batch import Metadata
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.state import gather_rows, save_hidden

from .decoder import FullAttention, Projection, module_copy


def absorb_feedback(previous, target, target_begins, metadata, output):
    """Pair accepted input tokens with their preceding committed target hidden state."""
    offsets = (
        torch.arange(output.shape[0], device=output.device)
        - metadata.query_start_loc[metadata.requests]
    )
    live = metadata.positions >= 0
    # Packed padding can extend far beyond a request's verified rows after a
    # rejection. It must never index the target buffer, even inside torch.where.
    source = torch.where(
        live, target_begins[metadata.requests] + offsets - 1, 0
    ).clamp_min(0)
    selected = torch.where(
        (offsets == 0)[:, None], previous[metadata.requests], target[source]
    )
    output.copy_(torch.where(live[:, None], selected, 0))


class DraftPool:
    def __init__(self, pool):
        self.pool = pool

    def attention(self, layer, config):
        return self.pool.draft_attention(config)


class PreparedMTP:
    def __init__(
        self,
        weights,
        draft,
        metadata,
        feedback,
        pool,
        *,
        session,
        workspace,
        name,
        mode,
        cache_only=False,
    ):
        self.metadata, self.feedback, self.draft = metadata, feedback, draft
        self.cache_only = cache_only
        c, tokens, device = weights.config, metadata.token_ids.numel(), feedback.device

        def zeros(width):
            return torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)

        self.embedded, self.joined, self.projected = (
            zeros(c.hidden),
            zeros(2 * c.hidden),
            zeros(c.hidden),
        )
        self.embedding = PreparedEmbedding(
            weights.tree.embed_tokens.weight,
            metadata.token_ids,
            self.embedded,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        )
        self.project = UnquantizedProjection(
            self.joined,
            draft.feedback,
            self.projected,
            session=session,
            name=name + ".feedback",
        )
        self.layer = module_copy(draft.layer)
        self.layer.self_attn = FullAttention(
            draft.layer.self_attn,
            SimpleNamespace(config=c, projections=draft.projections),
            self.projected,
            metadata,
            DraftPool(pool),
            0,
            mode=mode,
            session=session,
            workspace=workspace,
            name=name + ".attention",
            buffers={},
            cache_only=cache_only,
        )
        if not cache_only:
            self.output = zeros(c.hidden)
            self.layer.mlp = module_copy(draft.layer.mlp)
            intermediate = c.text["intermediate_size"]
            self.mlp_input = zeros(intermediate)
            self.mlp_buffers = {}
            for field in ("gate_proj", "up_proj", "down_proj"):
                source = self.projected if field != "down_proj" else self.mlp_input
                output = zeros(c.hidden if field == "down_proj" else intermediate)
                self.mlp_buffers[field] = output
                operation = UnquantizedProjection(
                    source,
                    draft.projections[f"layers.0.mlp.{field}"],
                    output,
                    session=session,
                    name=name + "." + field,
                )
                setattr(self.layer.mlp, field, Projection(operation))
        rope = c.text["rope_parameters"]
        dim = int(c.text["head_dim"] * rope["partial_rotary_factor"])
        self.inverse = 1 / rope["rope_theta"] ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
        )
        self.logits = None
        if not cache_only:
            rows = metadata.sample_indices.numel()
            self.hidden = torch.zeros(
                (rows, c.hidden), dtype=torch.bfloat16, device=device
            )
            self.logits = torch.zeros(
                (rows, c.vocabulary), dtype=torch.bfloat16, device=device
            )
            self.head = PreparedVocabularyProjection(
                self.hidden,
                weights.head,
                self.logits,
                self.logits,
                runtime=None,
                coordinator=None,
                session=session,
                name=name + ".head",
                activation_mode="auto",
            )

    def __call__(self):
        self.embedding()
        self.joined.copy_(
            torch.cat(
                (
                    self.draft.embedding_norm(self.embedded),
                    self.draft.hidden_norm(self.feedback),
                ),
                dim=-1,
            )
        )
        self.project(self.joined)
        frequencies = self.metadata.positions.float()[:, None] * self.inverse[None]
        angles = torch.cat((frequencies, frequencies), dim=-1)
        rope = angles.cos().bfloat16(), angles.sin().bfloat16()
        if self.cache_only:
            self.layer.self_attn(
                self.layer.input_layernorm(self.projected), rope, cache_only=True
            )
            return
        hidden = self.layer(self.projected, position_embeddings=rope)
        self.output.copy_(self.draft.norm(hidden))
        self.hidden.copy_(self.output[self.metadata.sample_indices])
        self.head(self.hidden)


class PrefillAbsorption:
    def __init__(self, owner, decoder, name):
        self.owner, self.decoder = owner, copy(decoder)
        self.weights, self.scratch, self.metadata, self.pool = (
            decoder.weights,
            decoder.scratch,
            decoder.metadata,
            decoder.pool,
        )
        device, tokens = (
            decoder.scratch.activations.device,
            decoder.scratch.activations.shape[0],
        )
        self.decoder.auxiliary = torch.zeros_like(decoder.scratch.activations)
        self.draft_metadata = Metadata.allocate(
            tokens, 1, owner.inputs.page_tables.shape[1], device
        )
        self.feedback = torch.zeros_like(self.decoder.auxiliary)
        self.previous = torch.zeros(
            (1, owner.weights.config.hidden), dtype=torch.bfloat16, device=device
        )
        self.indices = torch.arange(tokens, device=device)
        self.scratch_bytes = (
            tokens
            * (
                6 * owner.weights.config.hidden
                + 3
                * owner.weights.config.text["num_key_value_heads"]
                * owner.weights.config.text["head_dim"]
            )
            * 2
        )
        self.absorb = PreparedMTP(
            owner.weights,
            owner.draft_weights,
            self.draft_metadata,
            self.feedback,
            owner.pool,
            session=owner.session,
            workspace=owner.workspace,
            name=name + ".absorb",
            mode="prefill",
            cache_only=True,
        )

    def __call__(self):
        o, m, d = self.owner, self.metadata, self.draft_metadata
        # A compiled advanced index may materialize every logical row before
        # gathering. Only the admitted physical prefix is backed during preparation.
        gather_rows(o.pool.last_hidden, m.state_rows, self.previous)
        self.decoder()
        shift = (m.starts == 0).long()
        lengths = (m.lengths - shift).clamp_min(0)
        source = (self.indices + shift[0]).clamp_max(self.indices.numel() - 1)
        hidden = self.decoder.auxiliary[(source - 1).clamp_min(0)]
        self.feedback.copy_(
            torch.where(
                ((self.indices == 0) & (m.starts[0] > 0))[:, None],
                self.previous,
                hidden,
            )
        )
        slots = o.inputs.row_slot_ids[:1].long().clamp(0, o.requests.capacity - 1)
        d.set(
            (m.starts - 1).clamp_min(0),
            lengths,
            m.state_rows,
            o.draft_tables[slots],
            m.token_ids[source].view(1, -1),
            o.inputs.live_rows,
        )
        self.absorb()
        save_hidden(
            (m.starts + m.lengths)[:, None],
            o.pool.region("consumed_tokens"),
            m.state_rows,
            m.lengths > 0,
        )

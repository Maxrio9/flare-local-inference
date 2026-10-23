"""LiLiCorr context injection and bidirectional draft blocks."""

from dataclasses import dataclass

import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.grouped_convolution import PreparedGroupedConvolution
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.speculation import SpeculativeRngState
from ignition.components.vocabulary import PreparedVocabularyProjection

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder, Scratch
from .lilicorr import PreparedLatticeSelector
from ignition.components.dflash import PreparedContext, DraftAttention, DraftMLP, Rotary, _norm


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, vocabulary_runtime, coordinator,
                 session, name, workspace, buffers, layer_buffers, runtime=None):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        c, s = weights.config.draft, scratch
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, s.activations,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.layers, previous = [], None
        for index, w in enumerate(weights.draft.layers):
            prefix, storage = f"{name}.layer{index}", layer_buffers.cursor()
            norm = PreparedRMSNorm(s.activations, w["input_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=previous, session=session, name=prefix + ".norm")
            convs = []
            for kind in ("attention_conv", "mlp_conv"):
                coefficients = 2 * c.text["dflash_config"]["conv_kernel_size"] * weights.config.hidden // c.text["dflash_config"]["conv_group_size"]
                convs.append(PreparedGroupedConvolution(s.normalized, w[kind], block_size=c.block_size,
                    group_size=c.text["dflash_config"]["conv_group_size"], session=session, name=prefix + "." + kind,
                    buffers={key: storage.zeros((s.activations.shape[0], width), torch.bfloat16)
                        for key, width in (("coefficients", coefficients), ("before", weights.config.hidden), ("after", weights.config.hidden))}))
            attention = DraftAttention(w, s.normalized, metadata, pool, c, index,
                session=session, name=prefix + ".attention", workspace=workspace, buffers=storage)
            post = PreparedRMSNorm(s.activations, w["post_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=s.normalized, session=session, name=prefix + ".post_norm")
            mlp = DraftMLP(w["mlp"], s.normalized, session=session, name=prefix + ".mlp", workspace=workspace, buffers=storage)
            self.layers.append((norm, convs[0], attention, post, convs[1], mlp))
            previous = s.normalized
        self.norm = PreparedRMSNorm(s.activations, weights.draft.norm, s.normalized,
            eps=c.text["rms_norm_eps"], residual=previous, session=session, name=name + ".norm")
        self.head = PreparedVocabularyProjection(s.hidden, weights.head, s.local_logits, s.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto")
        self.anchor = buffers.zeros((metadata.lengths.numel(), weights.config.hidden), torch.bfloat16)
        self.saved_anchor = pool.reserved_region("draft_anchor")
        self.selector = PreparedLatticeSelector(weights.draft.lattice, c, weights.embedding,
            s.hidden.view(-1, c.block_size, weights.config.hidden), s.logits, self.anchor,
            session=session, name=name + ".selector", workspace=workspace, buffers=buffers, layer_buffers=layer_buffers)

    def __call__(self):
        residual, output = self.embedding(), None
        for norm, conv, attention, post, mlp_conv, mlp in self.layers:
            output = conv.finish(attention(conv.prepare(norm(residual, output))))
            output = mlp_conv.finish(mlp(mlp_conv.prepare(post(residual, output))))
        self.scratch.hidden.copy_(self.norm(residual, output)[self.metadata.sample_indices])
        self.head(self.scratch.hidden)
        c = self.weights.config.draft
        self.anchor.copy_(self.saved_anchor[self.metadata.state_rows])
        self.selector.prepare(self.scratch.hidden.view(-1, c.block_size, self.weights.config.hidden),
            self.scratch.logits.view(-1, c.block_size, self.weights.config.vocabulary), self.anchor,
            (self.metadata.starts > 0) & (self.metadata.lengths > 0))


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object = None


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, *, completion, frame, sequence,
                 session, workspace, coordinator, buffers, layer_buffers):
        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, None
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=None, vocabulary_runtime=None, coordinator=coordinator)
        self.markov_runtime = None
        self.buffers, self.layer_buffers = buffers, layer_buffers
        self.cursor = None
        self.forwards = DraftForwards(self.decoder, self.context, self.draft)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = None

    def begin(self, batch, width):
        self.cursor = self.buffers.cursor()
        self.scratch = Scratch.allocate(batch * width, batch * width, self.weights.config, self.cursor)
        self.draft_scratch = Scratch.allocate(batch * self.block_size,
            batch * self.block_size, self.weights.config, self.cursor)

    def decoder(self, weights, tables, scratch, metadata, pool, *, runtime, **kwargs):
        return PreparedDecoder(weights, scratch, metadata, pool, buffers=self.cursor, layer_buffers=self.layer_buffers, **kwargs)

    def context(self, *args, **kwargs):
        return PreparedContext(*args, buffers=self.cursor, **kwargs)

    def draft(self, *args, **kwargs):
        return PreparedDraft(*args, buffers=self.cursor, layer_buffers=self.layer_buffers, **kwargs)

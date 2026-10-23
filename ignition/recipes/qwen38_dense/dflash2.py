"""Dense Qwen DFlash2 parallel block drafting using shared prepared operations."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from ignition.components.embedding import PreparedEmbedding
from ignition.components.grouped_convolution import PreparedGroupedConvolution
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.dflash import PreparedContext, DraftAttention, DraftMLP
from .dflash2_selector import PreparedSelector


def draft_anchors(metadata, block_size):
    """Gather pending tokens in logical request order after packed computation."""
    return metadata.token_ids[metadata.sample_indices.view(-1, block_size)[:, 0]]


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
        self.selector = PreparedSelector(weights.draft.selector, c,
            s.hidden.view(-1, c.block_size, weights.config.hidden), s.logits,
            session=session, name=name + ".selector", buffers=buffers)

    def __call__(self):
        residual, output = self.embedding(), None
        for norm, conv, attention, post, mlp_conv, mlp in self.layers:
            output = conv.finish(attention(conv.prepare(norm(residual, output))))
            output = mlp_conv.finish(mlp(mlp_conv.prepare(post(residual, output))))
        self.scratch.hidden.copy_(self.norm(residual, output)[self.metadata.sample_indices])
        self.head(self.scratch.hidden)
        c = self.weights.config.draft
        self.selector.prepare(self.scratch.hidden.view(-1, c.block_size, self.weights.config.hidden),
            self.scratch.logits.view(-1, c.block_size, self.weights.config.vocabulary),
            draft_anchors(self.metadata, c.block_size))


from ignition.components.gdn_recovery import GDNLayerRecords
from ignition.components.parallel_draft import PreparedParallelDraftRound
from ignition.components.speculation import SpeculativeRngState
from ignition.components.state import save_hidden
from ignition.preparation.scratch import GraphBufferPool
from .decoder import PreparedDecoder


@dataclass(frozen=True)
class DraftScratch:
    activations: torch.Tensor
    normalized: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor
    local_logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, config, buffers):
        a, n, h = (buffers.zeros((tokens, config.hidden), torch.bfloat16) for _ in range(3))
        logits = buffers.zeros((tokens, config.vocabulary), torch.bfloat16)
        return cls(a, n, h, logits, logits)

    def rows(self, tokens, rows):
        return DraftScratch(self.activations[:tokens], self.normalized[:tokens], self.hidden[:rows],
            self.logits[:rows], self.local_logits[:rows])


class DFlashPreparation:
    def __init__(self, weights, pool, requests, inputs, scratch, completion, frame, sequence,
                 *, width, session, workspace, draft_weights):
        from .config import PAGE_TOKENS
        self.weights, self.pool, self.requests, self.inputs = weights, pool, requests, inputs
        self.tables, self.scratch, self.completion = None, scratch, completion
        self.frame, self.sequence, self.width = frame, sequence, width
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=None, vocabulary_runtime=None, coordinator=None)
        self.markov_runtime = None
        self.page_tokens, self.block_size, self.logit_offset = PAGE_TOKENS, weights.config.draft.block_size, 1
        self.forwards = SimpleNamespace(decoder=self.decoder, context=self.context, draft=self.draft, markov=None)
        device = pool.backing.device
        self.rng = SpeculativeRngState.allocate(device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.records = GDNLayerRecords.allocate((i for i, kind in enumerate(weights.config.layers)
            if kind == "linear_attention"), requests.capacity, device, key_heads=16, value_heads=48)
        self.storage = GraphBufferPool(device, chunk_bytes=2 << 20)
        self.layer_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        self.context_storage = GraphBufferPool(device, chunk_bytes=2 << 20)

    def decoder(self, weights, tables, scratch, metadata, pool, *, runtime, **kwargs):
        result = PreparedDecoder(weights, scratch, metadata, pool,
            records=self.records if metadata.max_query_tokens > 1 else None,
            verification_columns=self.width, feature_layers=weights.config.draft.target_layers, **kwargs)
        return result

    def context(self, *args, **kwargs):
        return PreparedContext(*args, buffers=self.cursor, retain_anchor=False, **kwargs)

    def draft(self, *args, **kwargs):
        kwargs.pop("runtime", None)
        return PreparedDraft(*args, buffers=self.cursor, layer_buffers=self.layer_buffers, **kwargs)

    def round(self, batch, width, name):
        self.cursor = self.storage.cursor()
        self.draft_scratch = DraftScratch.allocate(batch * self.block_size, self.weights.config, self.cursor)
        result = PreparedParallelDraftRound(self, batch, width, name, buffers=self.cursor)
        result.validator.device_pending = True
        return DenseParallelRound(result)


class DenseParallelRound:
    """Publish the shared continuation cursor after the parallel speculative transaction."""
    def __init__(self, forward):
        self.forward = forward

    def prepare(self):
        self.forward.prepare()
        if self.forward.width > 1:
            self.forward.decoder.prepare_commit(self.forward.operations[-1].emitted_count)

    def __call__(self):
        f = self.forward
        f()
        counts = f.operations[-1].emitted_count
        save_hidden((f.before + counts)[:, None], f.owner.pool.region("consumed_tokens"),
            f.metadata.state_rows, counts > 0)


class PrefillContext:
    def __init__(self, owner, decoder, name):
        from copy import copy
        self.decoder = copy(decoder)
        self.weights, self.scratch, self.metadata, self.pool = decoder.weights, decoder.scratch, decoder.metadata, decoder.pool
        self.decoder.feature_layers = self.weights.config.draft.target_layers
        # Prefill/scored variants execute serially. Each cursor keeps the tap
        # rows live through context projection, with all intermediates disjoint.
        cursor = owner.context_storage.cursor()
        self.decoder.features = cursor.zeros((decoder.scratch.activations.shape[0],
            len(self.decoder.feature_layers) * self.weights.config.hidden), torch.bfloat16)
        self.decoder.auxiliary = self.decoder.features
        self.context = PreparedContext(owner.weights, self.decoder.features, self.metadata, self.pool,
            session=owner.session, workspace=owner.workspace, name=name + ".context", buffers=cursor, retain_anchor=False)
        self.buffers = owner.context_storage

    def __call__(self):
        self.decoder()
        self.context(self.decoder.features, self.metadata.lengths)
        save_hidden((self.metadata.starts + self.metadata.lengths)[:, None],
            self.pool.region("consumed_tokens"), self.metadata.state_rows, self.metadata.lengths > 0)

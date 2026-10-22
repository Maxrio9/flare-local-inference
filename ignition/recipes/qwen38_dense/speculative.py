"""Native dense MTP proposals and accepted-prefix target verification."""

import torch

from ignition.components.gdn_recovery import GDNLayerRecords
from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.paged_batch import Metadata, PreparedBatch
from ignition.components.parallel_draft import prepare_sampling_scratch
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
    SpeculativeRngState,
)
from ignition.components.state import save_hidden, gather_rows
from ignition.preparation.scratch import GraphBufferPool

from .decoder import PreparedDecoder


class SpeculativePreparation:
    def __init__(
        self,
        weights,
        pool,
        requests,
        inputs,
        scratch,
        completion,
        frame,
        sequence,
        *,
        width,
        session,
        workspace,
        draft_weights,
    ):
        self.weights, self.pool, self.requests, self.inputs = (
            weights,
            pool,
            requests,
            inputs,
        )
        self.scratch, self.completion, self.frame, self.sequence = (
            scratch,
            completion,
            frame,
            sequence,
        )
        self.width, self.session, self.workspace = width, session, workspace
        self.draft_weights = draft_weights
        self.draft_tables = torch.full_like(inputs.page_tables, -1)
        self.draft_generation = torch.zeros_like(requests.generation)
        self.draft_offset = torch.zeros_like(requests.generation)
        device, capacity = pool.backing.device, requests.capacity
        self.rng = SpeculativeRngState.allocate(device, capacity=capacity)
        self.records = GDNLayerRecords.allocate(
            (
                i
                for i, kind in enumerate(weights.config.layers)
                if kind == "linear_attention"
            ),
            capacity,
            device,
            key_heads=16,
            value_heads=48,
        )
        self.storage = GraphBufferPool(device, chunk_bytes=2 << 20)

    def round(self, batch, width, name):
        return PreparedRound(self, batch, width, name)


class PreparedRound:
    def __init__(self, owner, batch, width, name):
        self.owner, self.batch, self.width = owner, batch, width
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows = torch.arange(b, device=device)
        self.metadata = Metadata.allocate(
            b * w, b, blocks, device, samples=b * w, max_query_tokens=w
        )
        self.validator = PreparedBatch(
            o.inputs,
            o.requests,
            Metadata.allocate(b, b, blocks, device),
            mode="decode",
            shard=o.weights.shard,
            pool=o.pool,
            verification_width=w,
        )
        # Proposals are device-owned; validate the pending token directly.
        self.validator.device_pending = True
        self.decoder = PreparedDecoder(
            o.weights,
            o.scratch.rows(b * w, b * w),
            self.metadata,
            o.pool,
            mode="decode" if w == 1 else "verify",
            vocabulary_runtime=None,
            coordinator=None,
            session=o.session,
            name=name + ".target",
            workspace=o.workspace,
            records=o.records if w > 1 else None,
            verification_columns=o.width,
            retain_feedback=True,
        )
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        buffers = o.storage.cursor()
        self.probabilities = buffers.zeros(
            (b, w - 1, o.requests.vocab_size), torch.float32
        )
        self.sampling = prepare_sampling_scratch(
            b,
            w,
            o.requests.vocab_size,
            device,
            buffers=buffers,
            proposal=True,
        )
        self.copy_mask = torch.zeros(b, dtype=torch.int64, device=device)
        self.draft = None
        from .mtp import PreparedMTP

        self.previous = torch.zeros(
            (b, o.weights.config.hidden), dtype=torch.bfloat16, device=device
        )
        self.feedback = torch.zeros_like(self.previous)
        self.draft_metadata = Metadata.allocate(b, b, blocks, device)
        if w > 1:
            self.draft = PreparedMTP(
                o.weights,
                o.draft_weights,
                self.draft_metadata,
                self.feedback,
                o.pool,
                session=o.session,
                workspace=o.workspace,
                name=name + ".draft",
                mode="decode",
            )
        self.absorb_metadata = Metadata.allocate(
            b * w, b, blocks, device, max_query_tokens=w
        )
        self.absorb_feedback = torch.zeros(
            (b * w, o.weights.config.hidden), dtype=torch.bfloat16, device=device
        )
        self.absorb = PreparedMTP(
            o.weights,
            o.draft_weights,
            self.absorb_metadata,
            self.absorb_feedback,
            o.pool,
            session=o.session,
            workspace=o.workspace,
            name=name + ".absorb",
            mode="verify",
            cache_only=True,
        )
        self.seed, self.offset, self.base_offset, self.sampled = (
            torch.zeros(b, dtype=torch.int64, device=device) for _ in range(4)
        )
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        args = (
            o.requests,
            o.inputs.row_slot_ids[:b],
            o.inputs.row_generations[:b],
            o.inputs.live_rows,
        )
        target = PreparedSpeculativeDistributions(
            self.decoder.scratch.logits.view(b, w, -1),
            *args,
            self.drafts,
            workspace=self.sampling["target"],
        )
        proposals = tuple(
            PreparedDraftDistribution(
                self.draft.logits.view(b, 1, -1),
                *args,
                self.drafts,
                prefix_offset=i,
                seed=self.seed,
                offset=self.offset,
                sampled=self.sampled,
                workspace=self.sampling["proposal"],
            )
            for i in range(w - 1)
        )
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(
            self.probabilities,
            self.drafts,
            target.probabilities,
            *args,
            rng=o.rng,
            draft_valid=self.valid,
            target_valid=target.valid,
            token_limits=o.inputs.query_lengths[:b],
            workspace=self.sampling["verify"],
        )
        verify.prepare()
        commit = PreparedSpeculativeCommit(
            o.requests,
            verify.output,
            self.candidates,
            o.inputs.row_slot_ids[:b],
            o.inputs.row_generations[:b],
            o.inputs.live_rows,
            o.completion,
            o.frame,
            o.sequence,
            logits=self.decoder.scratch.logits.view(b, w, -1),
            rng=o.rng,
            copy_offered_mask=self.copy_mask,
            token_limits=o.inputs.query_lengths[:b],
            draft_context_lag=1,
        )
        commit.prepare()
        if w > 1:
            self.decoder.prepare_commit(commit.emitted_count)
        self.operations = target, proposals, verify, commit

    def __call__(self):
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        validation = self.validator.validate()
        self.validator.begin(validation)
        slots = o.inputs.row_slot_ids[:b].long().clamp(0, o.requests.capacity - 1)
        self.active.copy_(self.validator.token_state.active.bool())
        self.before.copy_(o.requests.consumed_tokens[slots])
        gather_rows(
            o.pool.last_hidden, o.inputs.request_state_rows[slots, 0], self.previous
        )
        self.drafts.zero_()
        self.valid.zero_()
        self.copy_mask.zero_()
        self.base_offset.copy_(
            torch.where(
                o.draft_generation[slots] == o.requests.generation[slots],
                o.draft_offset[slots],
                0,
            )
        )
        self.seed.copy_(o.requests.seed.view(torch.int64)[slots] ^ DRAFT_RNG_DOMAIN)
        for index, proposal in enumerate(proposals):
            proposing = self.active & (index < o.inputs.query_lengths[:b] - 1)
            tokens = (
                o.requests.pending_token[slots]
                if index == 0
                else self.drafts[:, index - 1]
            )
            self.draft_metadata.set(
                self.before - 1 + index,
                proposing.int(),
                o.inputs.request_state_rows[slots, 0],
                o.draft_tables[slots],
                tokens[:, None],
                o.inputs.live_rows,
            )
            self.feedback.copy_(
                (self.previous if index == 0 else self.draft.hidden)[
                    self.draft_metadata.requests
                ]
            )
            self.draft()
            self.offset.copy_(self.base_offset + 4 * index)
            proposal()
            self.drafts[:, index].copy_(self.sampled)
            self.probabilities[:, index].copy_(proposal.probabilities[:, 0])
            self.valid[:, index].copy_(proposal.valid[:, 0] & proposing)
        if w > 1:
            matches = self.active[None] & (
                o.inputs.row_slot_ids[:b][None]
                == torch.arange(o.requests.capacity, device=slots.device)[:, None]
            )
            base = torch.where(matches, self.base_offset[None], 0).sum(1)
            o.draft_offset.copy_(
                torch.where(matches.any(1), base + 4 * (w - 1), o.draft_offset)
            )
            o.draft_generation.copy_(
                torch.where(matches.any(1), o.requests.generation, o.draft_generation)
            )
        self.active.logical_and_(o.requests.status[slots] == 1)
        self.candidates[:, 0].copy_(o.requests.pending_token[slots])
        self.candidates[:, 1:].copy_(self.drafts)
        lengths = torch.where(self.active, o.inputs.query_lengths[:b], 0)
        self.metadata.set(
            self.before,
            lengths,
            o.inputs.request_state_rows[slots, 0],
            o.inputs.page_tables[slots],
            self.candidates,
            o.inputs.live_rows,
        )
        self.decoder()
        target()
        verify()
        commit()
        counts = commit.emitted_count
        if w > 1:
            self.decoder.commit(counts)
        hidden = self.decoder.scratch.hidden.view(b, w, -1)[
            self.rows, (counts - 1).long().clamp_min(0)
        ]
        save_hidden(hidden, o.pool.last_hidden, self.metadata.state_rows, counts > 0)

        m = self.absorb_metadata
        m.set(
            (self.before - 1).clamp_min(0),
            counts,
            self.metadata.state_rows,
            o.draft_tables[slots],
            self.candidates,
            o.inputs.live_rows,
        )
        from .mtp import absorb_feedback

        absorb_feedback(
            self.previous,
            self.decoder.auxiliary,
            self.metadata.query_start_loc,
            m,
            self.absorb_feedback,
        )
        self.absorb()
        # Dense request counters are separate from continuation storage; publish the
        # declared cursor after commit, before any snapshot or following round.
        save_hidden(
            o.requests.consumed_tokens[slots, None],
            o.pool.region("consumed_tokens"),
            self.metadata.state_rows,
            counts > 0,
        )

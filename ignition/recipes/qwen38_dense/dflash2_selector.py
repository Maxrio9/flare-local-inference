"""Predecessor-conditioned DFlash2 scores over the shared head's top candidates."""

import torch

from ignition.components.linear import UnquantizedProjection


def edge_scores(predecessors, successors, hidden, candidate_ids, unary, anchor):
    predecessor_ids = torch.cat((anchor[:, None, None].expand(-1, 1, candidate_ids.shape[-1]),
        candidate_ids[:, :-1]), dim=1)
    return unary[:, :, None] + torch.einsum("bspr,bscr->bspc",
        predecessors[predecessor_ids] * hidden[:, :, None], successors[candidate_ids])


class PreparedSelector:
    def __init__(self, weights, config, hidden, logits, *, session, name, buffers):
        self.weights = weights
        self.top_k = config.text["dflash_config"]["selector_top_k"]
        batch, block, width = hidden.shape
        self.batch, self.slots = batch, block - 1
        self.projected = buffers.zeros((batch * self.slots, 256), torch.bfloat16)
        self.source = buffers.zeros((batch * self.slots, width), torch.bfloat16)
        self.project = UnquantizedProjection(self.source, weights["hidden_projection.weight"], self.projected,
            session=session, name=name + ".hidden")
        self.ids = buffers.zeros((batch, self.slots, self.top_k), torch.int64)
        self.unary = buffers.zeros((batch, self.slots, self.top_k), torch.float32)
        self.scores = buffers.zeros((batch, self.slots, self.top_k, self.top_k), torch.float32)
        self.rows = torch.arange(batch, device=hidden.device)

    def prepare(self, hidden, logits, anchor, anchor_valid=None):
        self.source.copy_(hidden[:, 1:].reshape_as(self.source))
        self.project(self.source)
        values, ids = logits[:, 1:].float().topk(self.top_k, dim=-1, sorted=True)
        self.ids.copy_(ids)
        self.unary.copy_(values)
        # The anchor is the pending token, not a target hidden vector.
        self.scores.copy_(edge_scores(self.weights["predecessor_codebook"],
            self.weights["successor_codebook"], self.projected.view(self.batch, self.slots, -1),
            self.ids, self.unary, anchor))

    def __call__(self, logits, predecessor, position):
        step = position - 1
        previous = ((self.ids[:, step - 1] == predecessor[:, None]).int().argmax(-1)
                    if step else torch.zeros_like(predecessor))
        selected = self.scores[self.rows, step, previous]
        if step:
            found = (self.ids[:, step - 1] == predecessor[:, None]).any(-1)
            selected = torch.where(found[:, None], selected, -torch.inf)
        logits.copy_(torch.full_like(logits, -torch.inf).scatter(1, self.ids[:, step], selected))

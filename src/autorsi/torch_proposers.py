"""In-process gradient-informed proposers. They only *propose*; the gates still decide."""
from __future__ import annotations

from .contracts import DeltaEdit, EditRecipe, ProposalContext


class GradientProposer:
    """Propose a descent step on a training loss; AutoRSI accepts or rejects it.

    mode: "full" (dense step), "lowrank" (rank-r SVD of the gradient, applied in
    place), or "sign" (signSGD step on every element). The step is normalised so
    the per-element size is ~`scale` weight units, then clipped to the backend's
    max_abs_delta. `scale` grows after an acceptance and shrinks after a rejection
    (a trust region driven by the gate decisions).
    """
    def __init__(self, backend, loss_fn, *, names=None, scale=2e-4, mode="full", rank=8,
                 grow=1.3, shrink=0.5, min_scale=1e-6):
        self.backend, self.loss_fn = backend, loss_fn
        self.names = tuple(names or backend.recipe_space["parameters"])
        self.scale, self.mode, self.rank = scale, mode, rank
        self.grow, self.shrink, self.min_scale = grow, shrink, min_scale
        self.log = []

    def propose(self, context: ProposalContext):
        import torch
        if context.history:
            self.scale *= self.grow if context.history[-1]["status"] == "accepted" else self.shrink
        if self.scale < self.min_scale:
            return None
        grads, loss = self.backend.gradients(self.loss_fn, self.names)
        cap = float(context.recipe_space["max_abs_delta"])
        edits = []
        for k, g in grads.items():
            if self.mode == "lowrank" and g.dim() == 2:
                U, S, V = torch.svd_lowrank(g, q=min(self.rank, min(g.shape)))
                g = (U * S) @ V.T
            if self.mode == "sign":
                d = -self.scale * torch.sign(g)
            else:
                d = -self.scale * g / g.pow(2).mean().sqrt().clamp_min(1e-20)
            edits.append(DeltaEdit(k, d.clamp_(-cap, cap)))
        self.log.append({"train_loss": loss, "scale": self.scale})
        return EditRecipe(tuple(edits), f"{self.mode} descent step, scale={self.scale:.2e}")

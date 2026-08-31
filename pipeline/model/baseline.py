"""
The agent's EDITABLE architecture — not the scored baseline.

The organizer's actual reference baseline (FM, k=16, lr=0.001) is vendored
verbatim in `starter_kit/baseline.py` and is what the agent must beat; it
is reproduced directly (never reimplemented) by
`pipeline/official_baseline.py`, called once at the start of every
orchestrator run (Task Requirement #1). This module is the agent's own
starting model — the thing the LLM-driven iteration loop actually rewrites
and improves — not a stand-in for the organizer baseline.

Architecture: a small embedding + MLP CTR model (user_id, video_id, tab,
numeric signals -> long_view probability). This is a standard, unglamorous
DeepFM-lite reference point, chosen because it's cheap to train and gives
the agent's ablation loop (agent/ablation.py) an obvious set of blocks to
improve on: embeddings, feature crosses, the MLP tower, the loss. Per the
Starter Kit's own priority list (agent/skill_store/tier1_core.md), the
highest-leverage first moves are a pairwise/listwise loss and sequence
modeling, not architecture swaps.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCTRModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_videos: int,
        n_tabs: int = 15,
        embed_dim: int = 16,
        numeric_dim: int = 4,
        hidden_dims: tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.video_embed = nn.Embedding(n_videos, embed_dim)
        self.tab_embed = nn.Embedding(n_tabs, embed_dim)

        tower_in = embed_dim * 3 + numeric_dim
        layers = []
        prev = tower_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.tower = nn.Sequential(*layers)

    def forward(
        self,
        user_ids: torch.Tensor,
        video_ids: torch.Tensor,
        tab_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        u = self.user_embed(user_ids)
        v = self.video_embed(video_ids)
        t = self.tab_embed(tab_ids)
        x = torch.cat([u, v, t, numeric_features], dim=-1)
        logits = self.tower(x).squeeze(-1)
        return logits  # raw logits — apply sigmoid / BCEWithLogitsLoss outside

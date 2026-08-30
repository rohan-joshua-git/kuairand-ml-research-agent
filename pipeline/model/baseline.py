"""
This pipeline's own starting model — NOT the scored baseline.

The challenge rules are explicit: the agent is scored against the
organizer-provided reference baseline, not one it (or we) wrote ourselves.
That reference is the vendored `starter_kit/baseline.py` (FM) and its
numbers in `config.starter_kit.official_baseline` — see README "Confirmed
by the Starter Kit". This module is what the agent's own pipeline starts
from and iterates on; it needs to reach and then beat the FM numbers, not
replace them.

Architecture: a small embedding + MLP CTR model (user_id, video_id, tab,
numeric signals -> long_view probability). This is a standard, unglamorous
DeepFM-lite reference point, chosen because it's cheap to train and gives
the agent's ablation loop (agent/ablation.py) an obvious set of blocks to
improve on: embeddings, feature crosses, the MLP tower, the loss.
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

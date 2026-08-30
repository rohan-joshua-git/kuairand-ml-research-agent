"""
Training entrypoint. This is the other primary surface the agent rewrites
each iteration (Figure 1 "train + tune" stage) — loss function, optimizer,
hyperparameters, and (via pipeline/model/architectures/) the model class
itself are all fair game for the agent to change.

Running this file directly trains this pipeline's own starting model
end-to-end on whatever's in `config.dataset.raw_dir` and reports val
metrics — not the scored baseline, see `pipeline/model/baseline.py`. The
orchestrator calls into `run_training` programmatically rather than
shelling out, so it can capture metrics/losses per epoch for the run log.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pipeline.data.features import NUMERIC_SIGNAL_COLUMNS, build_features
from pipeline.data.label import resolve_primary_label
from pipeline.data.loader import KuaiRandSplit, load_config, load_split
from pipeline.evaluate import RankingMetrics, compute_ranking_metrics
from pipeline.model.baseline import BaselineCTRModel


@dataclass
class TrainResult:
    model: nn.Module
    id_maps: dict
    val_metrics: RankingMetrics
    epoch_losses: list[float] = field(default_factory=list)


class InteractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_maps: dict, label: pd.Series):
        self.user_ids = torch.tensor(df["user_id"].map(id_maps["user"]).to_numpy(), dtype=torch.long)
        self.video_ids = torch.tensor(df["video_id"].map(id_maps["video"]).to_numpy(), dtype=torch.long)
        self.tab_ids = torch.tensor(df["tab"].map(id_maps["tab"]).fillna(0).to_numpy(), dtype=torch.long)
        numeric = df[[c for c in NUMERIC_SIGNAL_COLUMNS if c in df.columns]].to_numpy(dtype=np.float32)
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.labels = torch.tensor(label.to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.user_ids[idx], self.video_ids[idx], self.tab_ids[idx], self.numeric[idx], self.labels[idx]


def _build_id_maps(train_df: pd.DataFrame) -> dict:
    users = train_df["user_id"].unique()
    videos = train_df["video_id"].unique()
    tabs = train_df["tab"].unique()
    return {
        "user": {u: i for i, u in enumerate(users)},
        "video": {v: i for i, v in enumerate(videos)},
        "tab": {t: i for i, t in enumerate(tabs)},
    }


def run_training(
    split: KuaiRandSplit | None = None,
    epochs: int = 3,
    batch_size: int = 2048,
    lr: float = 1e-3,
    pos_weight: float | None = None,
    device: str = "cpu",
) -> TrainResult:
    cfg = load_config()
    split = split or load_split(cfg)

    train_feat = build_features(split.train)
    val_feat = build_features(split.val)

    id_maps = _build_id_maps(train_feat)

    train_label = resolve_primary_label(train_feat)
    val_label = resolve_primary_label(val_feat)

    # unseen ids in val fall back to index 0 rather than crashing — a
    # simplification worth revisiting (e.g. a dedicated UNK embedding slot)
    # once this stops being the pipeline's iteration-0 starting model
    train_feat = train_feat[train_feat["user_id"].isin(id_maps["user"]) & train_feat["video_id"].isin(id_maps["video"])]
    val_feat = val_feat.copy()
    val_feat["user_id"] = val_feat["user_id"].where(val_feat["user_id"].isin(id_maps["user"]), next(iter(id_maps["user"])))
    val_feat["video_id"] = val_feat["video_id"].where(val_feat["video_id"].isin(id_maps["video"]), next(iter(id_maps["video"])))

    train_ds = InteractionDataset(train_feat, id_maps, train_label)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = BaselineCTRModel(
        n_users=len(id_maps["user"]),
        n_videos=len(id_maps["video"]),
        n_tabs=max(len(id_maps["tab"]), 1),
        numeric_dim=len([c for c in NUMERIC_SIGNAL_COLUMNS if c in train_feat.columns]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    epoch_losses = []
    model.train()
    for _epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for user_ids, video_ids, tab_ids, numeric, labels in train_loader:
            user_ids, video_ids, tab_ids = user_ids.to(device), video_ids.to(device), tab_ids.to(device)
            numeric, labels = numeric.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(user_ids, video_ids, tab_ids, numeric)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        epoch_losses.append(total_loss / max(n_batches, 1))

    val_metrics = _evaluate_on_split(model, val_feat, id_maps, val_label, device)

    return TrainResult(model=model, id_maps=id_maps, val_metrics=val_metrics, epoch_losses=epoch_losses)


def score_split(model: nn.Module, feat_df: pd.DataFrame, id_maps: dict, device: str = "cpu") -> np.ndarray:
    """Runs inference only (no labels needed) — shared by the val-scoring
    path below and `agent/referee.py`'s unbiased-probe scoring, so there's
    one implementation of "how does this model score a DataFrame of
    (user, video, tab, numeric) rows" rather than two that could drift."""
    dummy_label = pd.Series(0, index=feat_df.index, dtype="float32")
    ds = InteractionDataset(feat_df, id_maps, dummy_label)
    loader = DataLoader(ds, batch_size=4096, shuffle=False)

    model.eval()
    scores = []
    with torch.no_grad():
        for user_ids, video_ids, tab_ids, numeric, _labels in loader:
            logits = model(
                user_ids.to(device), video_ids.to(device), tab_ids.to(device), numeric.to(device)
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


def _evaluate_on_split(model: nn.Module, feat_df: pd.DataFrame, id_maps: dict, label: pd.Series, device: str) -> RankingMetrics:
    eval_df = feat_df[["user_id"]].copy()
    eval_df["score"] = score_split(model, feat_df, id_maps, device)
    eval_df["label"] = label.to_numpy()

    return compute_ranking_metrics(eval_df)


if __name__ == "__main__":
    result = run_training()
    print(f"epoch losses: {result.epoch_losses}")
    print(f"val GAUC:     {result.val_metrics.gauc:.4f}")
    print(f"val nDCG@5:   {result.val_metrics.ndcg_at_5:.4f}")

"""
Training entrypoint updated with multi-task / auxiliary signal learning heads.
Computes primary loss on `long_view` alongside auxiliary BCE losses on signals 
('is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward') weighted by 0.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pipeline.data.features import NUMERIC_SIGNAL_COLUMNS, build_features
from pipeline.data.label import resolve_label
from pipeline.data.loader import KuaiRandSplit, load_config, load_split
from pipeline.evaluate import RankingMetrics, compute_ranking_metrics

AUX_COLUMNS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward"]


class MultiTaskCTRModel(nn.Module):
    def __init__(self, n_users: int, n_videos: int, n_tabs: int, numeric_dim: int, emb_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.video_emb = nn.Embedding(n_videos, emb_dim)
        self.tab_emb = nn.Embedding(n_tabs, emb_dim)

        in_dim = emb_dim * 3 + numeric_dim
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.main_head = nn.Linear(64, 1)
        self.aux_heads = nn.Linear(64, len(AUX_COLUMNS))

    def forward(self, user_ids, video_ids, tab_ids, numeric):
        u = self.user_emb(user_ids)
        v = self.video_emb(video_ids)
        t = self.tab_emb(tab_ids)

        x = torch.cat([u, v, t, numeric], dim=-1)
        shared = self.shared_mlp(x)
        main_logits = self.main_head(shared).squeeze(-1)
        aux_logits = self.aux_heads(shared)
        return main_logits, aux_logits


@dataclass
class TrainResult:
    model: nn.Module
    id_maps: dict
    val_metrics: RankingMetrics
    epoch_losses: list[float] = field(default_factory=list)


class MultiTaskInteractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_maps: dict, label: pd.Series):
        self.user_ids = torch.tensor(df["user_id"].map(id_maps["user"]).to_numpy(), dtype=torch.long)
        self.video_ids = torch.tensor(df["video_id"].map(id_maps["video"]).to_numpy(), dtype=torch.long)
        self.tab_ids = torch.tensor(df["tab"].map(id_maps["tab"]).fillna(0).to_numpy(), dtype=torch.long)
        numeric = df[[c for c in NUMERIC_SIGNAL_COLUMNS if c in df.columns]].to_numpy(dtype=np.float32)
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.labels = torch.tensor(label.to_numpy(dtype=np.float32), dtype=torch.float32)

        aux_data = []
        for col in AUX_COLUMNS:
            if col in df.columns:
                aux_data.append(df[col].fillna(0).to_numpy(dtype=np.float32))
            else:
                aux_data.append(np.zeros(len(df), dtype=np.float32))
        self.aux_labels = torch.tensor(np.stack(aux_data, axis=-1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.user_ids[idx],
            self.video_ids[idx],
            self.tab_ids[idx],
            self.numeric[idx],
            self.labels[idx],
            self.aux_labels[idx],
        )


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
    aux_loss_weight: float = 0.2,
    device: str = "cpu",
    seed: int = 0,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = load_config()
    split = split or load_split(cfg)

    train_feat = build_features(split.train)
    val_feat = build_features(split.val)

    id_maps = _build_id_maps(train_feat)

    train_feat = train_feat[train_feat["user_id"].isin(id_maps["user"]) & train_feat["video_id"].isin(id_maps["video"])]
    val_feat = val_feat.copy()
    val_feat["user_id"] = val_feat["user_id"].where(val_feat["user_id"].isin(id_maps["user"]), next(iter(id_maps["user"])))
    val_feat["video_id"] = val_feat["video_id"].where(val_feat["video_id"].isin(id_maps["video"]), next(iter(id_maps["video"])))

    train_label = resolve_label(train_feat)
    val_label = resolve_label(val_feat)

    train_ds = MultiTaskInteractionDataset(train_feat, id_maps, train_label)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = MultiTaskCTRModel(
        n_users=len(id_maps["user"]),
        n_videos=len(id_maps["video"]),
        n_tabs=max(len(id_maps["tab"]), 1),
        numeric_dim=len([c for c in NUMERIC_SIGNAL_COLUMNS if c in train_feat.columns]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_losses = []
    model.train()
    for _epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for user_ids, video_ids, tab_ids, numeric, labels, aux_labels in train_loader:
            user_ids, video_ids, tab_ids = user_ids.to(device), video_ids.to(device), tab_ids.to(device)
            numeric, labels, aux_labels = numeric.to(device), labels.to(device), aux_labels.to(device)

            optimizer.zero_grad()
            main_logits, aux_logits = model(user_ids, video_ids, tab_ids, numeric)
            
            main_loss = loss_fn(main_logits, labels)
            aux_loss = loss_fn(aux_logits, aux_labels)
            
            loss = main_loss + aux_loss_weight * aux_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        epoch_losses.append(total_loss / max(n_batches, 1))

    val_metrics = _evaluate_on_split(model, val_feat, id_maps, val_label, device)

    return TrainResult(model=model, id_maps=id_maps, val_metrics=val_metrics, epoch_losses=epoch_losses)


def _evaluate_on_split(model: nn.Module, feat_df: pd.DataFrame, id_maps: dict, label: pd.Series, device: str) -> RankingMetrics:
    ds = MultiTaskInteractionDataset(feat_df, id_maps, label)
    loader = DataLoader(ds, batch_size=4096, shuffle=False)

    model.eval()
    scores = []
    with torch.no_grad():
        for user_ids, video_ids, tab_ids, numeric, _labels, _aux_labels in loader:
            main_logits, _ = model(
                user_ids.to(device), video_ids.to(device), tab_ids.to(device), numeric.to(device)
            )
            scores.append(torch.sigmoid(main_logits).cpu().numpy())

    eval_df = feat_df[["user_id"]].copy()
    eval_df["score"] = np.concatenate(scores) if scores else np.array([])
    eval_df["label"] = label.to_numpy()

    return compute_ranking_metrics(eval_df)


def score_dataframe(model: nn.Module, id_maps: dict, feat_df: pd.DataFrame, device: str = "cpu") -> np.ndarray:
    user_ids = feat_df["user_id"].map(id_maps["user"]).fillna(0).astype(int).to_numpy()
    video_ids = feat_df["video_id"].map(id_maps["video"]).fillna(0).astype(int).to_numpy()
    tab_ids = feat_df["tab"].map(id_maps["tab"]).fillna(0).astype(int).to_numpy()
    numeric = feat_df[[c for c in NUMERIC_SIGNAL_COLUMNS if c in feat_df.columns]].to_numpy(dtype=np.float32)

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(user_ids), 8192):
            end = start + 8192
            out = model(
                torch.tensor(user_ids[start:end], dtype=torch.long, device=device),
                torch.tensor(video_ids[start:end], dtype=torch.long, device=device),
                torch.tensor(tab_ids[start:end], dtype=torch.long, device=device),
                torch.tensor(numeric[start:end], dtype=torch.float32, device=device),
            )
            main_logits = out[0] if isinstance(out, tuple) else out
            scores.append(torch.sigmoid(main_logits).cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


if __name__ == "__main__":
    result = run_training()
    print(f"epoch losses: {result.epoch_losses}")
    print(f"val GAUC:    {result.val_metrics.gauc:.4f}")
    print(f"val nDCG@5:  {result.val_metrics.ndcg_at_5:.4f}")
    print(f"val primary: {result.val_metrics.primary:.4f}  (official baseline: 0.6016)")

"""
Training entrypoint — the agent's editable "train + tune" surface (Figure 1).
Loss function, optimizer, schedule and the model itself are all fair game for
the agent to rewrite; `agent/code_editor.py` targets this file directly.

STARTING MODEL (human-authored, iteration 0): a Factorization Machine over
the same five fields the official baseline uses — user_id, video_id,
author_id, tab, dur_bucket (`starter_kit/data.py` FIELDS). This is
deliberate. Task Requirement #1 asks the end-to-end pipeline to reach the
official baseline's validation score before any agent iteration begins, and
an editable pipeline that starts far below baseline makes the agent spend
its iterations climbing back to parity instead of past it.

Two properties of this data make an FM the right starting point rather than
a deeper tower, both measured rather than assumed:
  - Ranking is WITHIN user, so anything constant within a user cannot move
    that user's order. The signal lives in the user x item crossing, which
    is exactly what an FM's second-order term models.
  - The organizer's own ablation (`starter_kit/ablation_features.py`) found
    more features and larger embeddings both flat. Capacity is not the
    bottleneck; objective alignment is — which is the agent's job to fix
    (pairwise/listwise loss, auxiliary heads), not the starting model's.

`duration_ms` is bucketed by train-set quantiles rather than fed raw: on real
data it has mean ~9.7e4 and std ~9.5e4, so as a raw numeric input alongside
small-initialised embeddings it dominates the tower. The official baseline
buckets it for the same reason.

CONTRACT other modules depend on (asserted in pipeline/smoke_test.py, so a
rewrite that breaks either is rejected and rolled back):
    run_training(split=..., epochs=..., ...) -> TrainResult
    score_dataframe(model, id_maps, feat_df) -> np.ndarray, one finite score
        per input row, in input order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pipeline.data.features import EXTRA_CATEGORICAL_FIELDS, build_features
from pipeline.data.label import resolve_label
from pipeline.data.loader import KuaiRandSplit, load_config, load_split
from pipeline.evaluate import RankingMetrics, compute_ranking_metrics

# Mirrors starter_kit/data.py FIELDS — the official baseline's field list.
# The official baseline's five fields (starter_kit/data.py FIELDS).
OFFICIAL_FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]


def resolve_fields(frame: pd.DataFrame) -> list[str]:
    """Official fields plus any extra categorical column that features.py
    registered in EXTRA_CATEGORICAL_FIELDS and actually produced.

    Resolved once per training run and stored in id_maps, so scoring always
    encodes the same fields the model was fitted on even if the registry
    changes underneath a saved model."""
    extras = [f for f in EXTRA_CATEGORICAL_FIELDS
              if f not in OFFICIAL_FIELDS and (f in frame.columns or f == "dur_bucket")]
    return OFFICIAL_FIELDS + extras


# Back-compat alias; the authoritative list per-run lives in id_maps["fields"].
CATEGORICAL_FIELDS = OFFICIAL_FIELDS + [f for f in EXTRA_CATEGORICAL_FIELDS if f not in OFFICIAL_FIELDS]
N_DUR_BUCKETS = 10


@dataclass
class TrainResult:
    model: nn.Module
    id_maps: dict
    val_metrics: RankingMetrics
    epoch_losses: list[float] = field(default_factory=list)


class FactorizationMachine(nn.Module):
    """Second-order FM over one-hot categorical fields sharing a single
    embedding table (offsets make field values globally unique), matching
    `starter_kit/baseline.py`'s formulation:

        y = b + sum_i w[x_i] + 0.5 * ( (sum_i v[x_i])^2 - sum_i v[x_i]^2 )
    """

    def __init__(self, total_dim: int, k: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, k)
        self.linear = nn.Embedding(total_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x)                      # (B, F, k)
        summed = e.sum(dim=1)                      # (B, k)
        interaction = 0.5 * ((summed ** 2).sum(dim=1) - (e ** 2).sum(dim=(1, 2)))
        return self.bias + self.linear(x).sum(dim=(1, 2)) + interaction


def _dur_bucket(duration_ms: pd.Series, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, duration_ms.fillna(0).to_numpy())


def _build_id_maps(train_df: pd.DataFrame) -> dict:
    """Per-field vocabularies with a trailing UNK slot, plus global offsets so
    every field's values occupy a disjoint range of one shared table — same
    scheme as starter_kit/data.py::encode."""
    edges = np.quantile(train_df["duration_ms"].fillna(0).to_numpy(),
                        np.linspace(0, 1, N_DUR_BUCKETS + 1)[1:-1])
    frame = train_df.copy()
    frame["dur_bucket"] = _dur_bucket(frame["duration_ms"], edges)

    fields = resolve_fields(frame)
    vocabs, dims = {}, []
    for fieldname in fields:
        values = frame[fieldname].astype(str).unique() if fieldname in frame.columns else np.array([], dtype=str)
        vocabs[fieldname] = {v: i for i, v in enumerate(values)}
        dims.append(len(values) + 1)  # +1 UNK

    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)
    return {
        "fields": fields,
        "vocabs": vocabs,
        "unk": {f: len(vocabs[f]) for f in fields},
        "offsets": {f: int(o) for f, o in zip(fields, offsets)},
        "total_dim": int(sum(dims)),
        "dur_edges": edges,
    }


def _encode(df: pd.DataFrame, id_maps: dict) -> np.ndarray:
    """(N, len(CATEGORICAL_FIELDS)) int64 of globally-offset indices. Unseen
    values fall into their field's UNK slot rather than crashing."""
    frame = df.copy()
    frame["dur_bucket"] = _dur_bucket(frame["duration_ms"], id_maps["dur_edges"])

    columns = []
    for fieldname in id_maps.get("fields", CATEGORICAL_FIELDS):
        vocab = id_maps["vocabs"][fieldname]
        unk = id_maps["unk"][fieldname]
        offset = id_maps["offsets"][fieldname]
        if fieldname in frame.columns:
            codes = frame[fieldname].astype(str).map(vocab).fillna(unk).to_numpy(dtype=np.int64)
        else:
            codes = np.full(len(frame), unk, dtype=np.int64)
        columns.append(codes + offset)
    return np.stack(columns, axis=1)


def run_training(
    split: KuaiRandSplit | None = None,
    epochs: int = 12,
    batch_size: int = 8192,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    patience: int = 3,
    embed_dim: int = 16,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = load_config()
    split = split or load_split(cfg)

    # build_features auto-joins the static video-side table (author_id) for
    # every caller, so training, submission and the referee probe all encode
    # identical features — see pipeline/data/features.py.
    train_feat = build_features(split.train)
    val_feat = build_features(split.val)

    id_maps = _build_id_maps(train_feat)

    x_train = torch.tensor(_encode(train_feat, id_maps), dtype=torch.long)
    y_train = torch.tensor(resolve_label(train_feat).to_numpy(dtype=np.float32), dtype=torch.float32)
    val_label = resolve_label(val_feat)

    model = FactorizationMachine(total_dim=id_maps["total_dim"], k=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    generator = torch.Generator().manual_seed(seed)
    n = len(y_train)
    epoch_losses: list[float] = []
    best_primary, best_state, bad_epochs = -1.0, None, 0

    for _epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx].to(device)), y_train[idx].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        epoch_losses.append(total_loss / max(n_batches, 1))

        # Early stopping on the competition's own primary metric, not on loss —
        # the same rule (patience on validation primary) the official baseline uses.
        metrics = _evaluate(model, val_feat, id_maps, val_label, device)
        if metrics.primary > best_primary + 1e-5:
            best_primary = metrics.primary
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        id_maps=id_maps,
        val_metrics=_evaluate(model, val_feat, id_maps, val_label, device),
        epoch_losses=epoch_losses,
    )


def _evaluate(model: nn.Module, feat_df: pd.DataFrame, id_maps: dict, label: pd.Series, device: str) -> RankingMetrics:
    eval_df = feat_df[["user_id"]].copy()
    eval_df["score"] = score_dataframe(model, id_maps, feat_df, device=device)
    eval_df["label"] = label.to_numpy()
    return compute_ranking_metrics(eval_df)


def score_dataframe(model: nn.Module, id_maps: dict, feat_df: pd.DataFrame, device: str = "cpu") -> np.ndarray:
    """STABLE SCORING CONTRACT — see module docstring. One finite score per
    input row, in input order. Used by pipeline/submit.py for the graded
    submission and by pipeline/train_runner.py for the unbiased referee probe."""
    x = torch.tensor(_encode(feat_df, id_maps), dtype=torch.long)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 16384):
            out.append(torch.sigmoid(model(x[start:start + 16384].to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


if __name__ == "__main__":
    result = run_training()
    print(f"epoch losses: {[round(x, 4) for x in result.epoch_losses]}")
    print(f"val GAUC:    {result.val_metrics.gauc:.4f}")
    print(f"val nDCG@5:  {result.val_metrics.ndcg_at_5:.4f}")
    print(f"val primary: {result.val_metrics.primary:.4f}  (official baseline: 0.6016)")

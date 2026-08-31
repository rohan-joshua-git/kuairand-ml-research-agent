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
    crosses = [f for f in CROSS_TE_FIELDS if f in frame.columns]
    return OFFICIAL_FIELDS + extras + crosses


# Back-compat alias; the authoritative list per-run lives in id_maps["fields"].
CATEGORICAL_FIELDS = OFFICIAL_FIELDS + [f for f in EXTRA_CATEGORICAL_FIELDS if f not in OFFICIAL_FIELDS]
N_DUR_BUCKETS = 10

# --- explicit feature crosses ------------------------------------------------
# `tab` (the feed surface) is the master context variable in this dataset: its
# long_view base rate runs from 0.004 (tab 3) to 0.489 (tab 4), and EVERY item
# feature gains sharply when crossed with it. Measured on validation, smoothed
# target encoding fit on train, standalone within-user GAUC:
#     video_id     0.6387  ->  video_id x tab     0.6479
#     tag1         0.5604  ->  tag1 x tab         0.6153
#     upload_type  0.5214  ->  upload_type x tab  0.5938
# The FM's second-order term already approximates <e_video, e_tab>, but only at
# rank k=16; an explicit cross gives the interaction its own parameters.
#
# Fed as a smoothed, quantile-BUCKETED target encoding rather than a raw cross
# ID: 43% of (video, tab) cells have fewer than 5 training rows (median 6), so a
# free embedding per cell would memorise noise. The smoothing is what produces
# the 0.6479 above.
# MEASURED FLAT — disabled by default. Enabling {"te_video_x_tab":
# ["video_id", "tab"]} scored 0.6046 vs 0.6046 without it (3 seeds, std 0.0001).
# The registry and its out-of-fold machinery are kept because they are correct
# and reusable (the leakage guard is verified: on cells with <=3 training rows,
# corr(encoding, label) is 0.7531 in-fold vs 0.1748 out-of-fold), so a future
# iteration can add a spec here in one line. It is the HYPOTHESIS that failed,
# not the code. Add a spec only after checking that the model does not already
# contain both parent fields — see the note above.
CROSS_TE_SPECS: dict[str, list[str]] = {}
CROSS_TE_FIELDS = [f"{name}_bucket" for name in CROSS_TE_SPECS]
N_TE_BUCKETS = 32
TE_SMOOTHING = 20.0
TE_FOLDS = 5


def _te_key(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = frame[cols[0]].astype(str)
    for col in cols[1:]:
        out = out + "|" + frame[col].astype(str)
    return out


def _te_table(keys: pd.Series, y: np.ndarray, prior: float, m: float = TE_SMOOTHING) -> pd.Series:
    """Empirical-Bayes smoothed mean: a cell with few rows is pulled toward the
    global rate, which is what makes a 6-row (video, tab) cell usable at all."""
    grouped = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (grouped["sum"] + prior * m) / (grouped["count"] + m)


def _fit_cross_te(train_frame: pd.DataFrame, y: np.ndarray, seed: int = 0) -> tuple[dict, dict]:
    """Returns (state, oof). `state[name]` holds the FULL-train table used to
    encode val/test/submission rows; `oof[name]` holds K-fold OUT-OF-FOLD values
    for the training rows themselves.

    The out-of-fold pass is not optional. Encoding a training row with a table
    its own label helped build makes the feature look far more predictive during
    fitting than it is at serving time, so the model over-weights it and
    generalises worse — the classic target-encoding failure. Folds, not
    leave-one-out: LOO is invertible on small cells (the model can back out the
    row's own label from the dip), which is the pathology CatBoost's ordered
    statistics exist to avoid.
    """
    prior = float(np.mean(y))
    folds = np.random.RandomState(seed).randint(0, TE_FOLDS, size=len(train_frame))
    state, oof = {}, {}
    for name, cols in CROSS_TE_SPECS.items():
        keys = _te_key(train_frame, cols)
        values = np.full(len(train_frame), prior, dtype=np.float64)
        for fold in range(TE_FOLDS):
            held = folds == fold
            table = _te_table(keys[~held], y[~held], prior)
            values[held] = keys[held].map(table).fillna(prior).to_numpy()
        # Bucket edges come from the OOF values, so train and eval rows are
        # bucketed on the same scale the model actually saw during fitting.
        state[name] = {
            "cols": cols,
            "table": _te_table(keys, y, prior),
            "prior": prior,
            "edges": np.quantile(values, np.linspace(0, 1, N_TE_BUCKETS + 1)[1:-1]),
        }
        oof[name] = values
    return state, oof


def _add_cross_te_buckets(frame: pd.DataFrame, cross_te: dict) -> None:
    """Adds the `*_bucket` categorical columns in place. Uses the raw encoding
    already on the frame when present (the training frame carries OOF values),
    otherwise encodes from the full-train table — one code path for train,
    validation, the referee probe and the submission."""
    for name, spec in cross_te.items():
        if name in frame.columns:
            raw = frame[name].to_numpy(dtype=np.float64)
        else:
            raw = (_te_key(frame, spec["cols"]).map(spec["table"])
                   .fillna(spec["prior"]).to_numpy(dtype=np.float64))
        frame[f"{name}_bucket"] = np.searchsorted(spec["edges"], raw)


@dataclass
class TrainResult:
    model: nn.Module
    id_maps: dict
    val_metrics: RankingMetrics
    epoch_losses: list[float] = field(default_factory=list)


class FactorizationMachineWithMLP(nn.Module):
    """Combination of Factorization Machine and a two-layer MLP branch
    ([32, 16] with ReLU activations) taking concatenated sparse embeddings as input."""

    def __init__(self, total_dim: int, num_fields: int, k: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, k)
        self.linear = nn.Embedding(total_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        # MLP branch: input dim is num_fields * k
        mlp_input_dim = num_fields * k
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x)                      # (B, F, k)
        
        # FM part
        summed = e.sum(dim=1)                      # (B, k)
        interaction = 0.5 * ((summed ** 2).sum(dim=1) - (e ** 2).sum(dim=(1, 2)))
        fm_out = self.bias + self.linear(x).sum(dim=(1, 2)) + interaction

        # MLP part
        mlp_in = e.view(e.size(0), -1)             # (B, F * k)
        mlp_out = self.mlp(mlp_in).squeeze(-1)     # (B,)

        return fm_out + mlp_out


class CrossNetV2(nn.Module):
    """DCN-V2 cross layers (Wang et al., arXiv:2008.13535).

    x_{l+1} = x0 * (W_l x_l + b_l) + x_l

    Each layer multiplies element-wise by the ORIGINAL embedding vector, so L
    layers represent interactions up to degree L+1 explicitly, rather than
    hoping an MLP discovers them. This is the one architecture family with a
    principled claim on this problem: the FM's second-order term is a RANK-k
    bilinear form and cannot represent arbitrary (video, tab, context) 3-way
    structure. `rank` enables the paper's low-rank variant (W = U V^T).
    """

    def __init__(self, dim: int, n_layers: int = 3, rank: int | None = None):
        super().__init__()
        self.n_layers = n_layers
        if rank is None:
            self.w = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
            self.u = self.v = None
        else:
            self.u = nn.ParameterList([nn.Parameter(torch.randn(dim, rank) * 0.01) for _ in range(n_layers)])
            self.v = nn.ParameterList([nn.Parameter(torch.randn(rank, dim) * 0.01) for _ in range(n_layers)])
            self.b = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(n_layers)])
            self.w = None

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for i in range(self.n_layers):
            proj = self.w[i](x) if self.w is not None else (x @ self.v[i].T) @ self.u[i].T + self.b[i]
            x = x0 * proj + x
        return x


class DCNv2(nn.Module):
    """Parallel DCN-V2: explicit cross network alongside a deep MLP, both fed
    the concatenated field embeddings, plus the first-order linear term."""

    def __init__(self, total_dim: int, num_fields: int, k: int = 16,
                 cross_layers: int = 3, rank: int | None = None, hidden=(64, 32)):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, k)
        self.linear = nn.Embedding(total_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        dim = num_fields * k
        self.cross = CrossNetV2(dim, cross_layers, rank)
        layers, prev = [], dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(dim + prev, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.embedding(x).view(x.size(0), -1)
        out = self.head(torch.cat([self.cross(x0), self.deep(x0)], dim=1)).squeeze(-1)
        return out + self.bias + self.linear(x).sum(dim=(1, 2))


def _dur_bucket(duration_ms: pd.Series, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, duration_ms.fillna(0).to_numpy())


def _build_id_maps(train_df: pd.DataFrame, cross_te: dict | None = None) -> dict:
    """Per-field vocabularies with a trailing UNK slot, plus global offsets so
    every field's values occupy a disjoint range of one shared table — same
    scheme as starter_kit/data.py::encode."""
    edges = np.quantile(train_df["duration_ms"].fillna(0).to_numpy(),
                        np.linspace(0, 1, N_DUR_BUCKETS + 1)[1:-1])
    frame = train_df.copy()
    frame["dur_bucket"] = _dur_bucket(frame["duration_ms"], edges)
    cross_te = cross_te or {}
    _add_cross_te_buckets(frame, cross_te)

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
        "cross_te": cross_te,
    }


def _encode(df: pd.DataFrame, id_maps: dict) -> np.ndarray:
    """(N, len(CATEGORICAL_FIELDS)) int64 of globally-offset indices. Unseen
    values fall into their field's UNK slot rather than crashing."""
    frame = df.copy()
    frame["dur_bucket"] = _dur_bucket(frame["duration_ms"], id_maps["dur_edges"])
    _add_cross_te_buckets(frame, id_maps.get("cross_te", {}))

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
    model_type: str = "deepfm",
    cross_layers: int = 3,
    cross_rank: int | None = None,
    patience: int = 3,
    embed_dim: int = 16,
    weight_decay: float = 1e-6,
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

    y_np = resolve_label(train_feat).to_numpy(dtype=np.float32)

    # Fit the cross target encodings BEFORE building vocabularies, and write the
    # out-of-fold values onto the training frame so every downstream step —
    # vocab building, encoding, scoring — reads them from one place.
    cross_state, cross_oof = _fit_cross_te(train_feat, y_np, seed=seed)
    for name, values in cross_oof.items():
        train_feat[name] = values

    id_maps = _build_id_maps(train_feat, cross_state)

    x_train = torch.tensor(_encode(train_feat, id_maps), dtype=torch.long)
    y_train = torch.tensor(y_np, dtype=torch.float32)
    val_label = resolve_label(val_feat)

    num_fields = len(id_maps["fields"])
    if model_type == "dcnv2":
        model = DCNv2(total_dim=id_maps["total_dim"], num_fields=num_fields, k=embed_dim,
                      cross_layers=cross_layers, rank=cross_rank).to(device)
    else:
        model = FactorizationMachineWithMLP(total_dim=id_maps["total_dim"], num_fields=num_fields, k=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

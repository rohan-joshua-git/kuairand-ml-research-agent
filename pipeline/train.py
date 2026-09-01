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
    # Raw validation predictions, in val-frame row order, so pipeline/eval_protocol.py
    # can re-aggregate over any user subset (selection / confirmation half, bootstrap
    # resample) without a second scoring pass.
    val_scores: np.ndarray | None = None
    val_user_ids: np.ndarray | None = None
    val_labels: np.ndarray | None = None


class FactorizationMachineWithMLP(nn.Module):
    """Combination of Factorization Machine and a two-layer MLP branch
    ([32, 16] with ReLU activations) taking concatenated sparse embeddings as input.

    `use_fm` / `use_mlp` exist to DECOMPOSE the model's advantage: running
    FM-only and MLP-only against the full model attributes the gain to the
    pairwise interaction term, the nonlinear branch, or their combination.
    Both default True, which is the shipped model.
    """

    def __init__(self, total_dim: int, num_fields: int, k: int = 16,
                 use_fm: bool = True, use_mlp: bool = True, n_aux: int = 0):
        super().__init__()
        self.use_fm, self.use_mlp = use_fm, use_mlp
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
        ) if use_mlp else None

        # Auxiliary multi-task head. Predicts related engagement outcomes
        # (is_click, is_like, ...) from the SHARED embedding, as extra training
        # TARGETS only. It is never consulted at inference — `forward` returns
        # the main logit unless `with_aux` is set — so no outcome field is ever
        # an input. The hypothesis is that a denser related signal regularises
        # the shared representation; long_view fires on 31.3% of rows while
        # is_click fires on 46.3%.
        self.aux = nn.Linear(mlp_input_dim, n_aux) if n_aux > 0 else None

        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor, with_aux: bool = False):
        e = self.embedding(x)                      # (B, F, k)

        # First-order term is always present; it is what an FM-free model keeps.
        out = self.bias + self.linear(x).sum(dim=(1, 2))

        if self.use_fm:
            summed = e.sum(dim=1)                  # (B, k)
            out = out + 0.5 * ((summed ** 2).sum(dim=1) - (e ** 2).sum(dim=(1, 2)))

        if self.use_mlp:
            out = out + self.mlp(e.view(e.size(0), -1)).squeeze(-1)

        if with_aux:
            if self.aux is None:
                raise ValueError("with_aux=True but the model has no auxiliary head")
            return out, self.aux(e.view(e.size(0), -1))
        return out


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


def _build_id_maps(train_df: pd.DataFrame, cross_te: dict | None = None,
                   drop_fields: set[str] | None = None,
                   shuffle_fields: dict[str, int] | None = None) -> dict:
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
    if drop_fields:
        fields = [f for f in fields if f not in drop_fields]
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
        # {field: seed} — _encode permutes that field's codes ACROSS ROWS with a
        # seeded RNG. This is the memorisation control: it preserves cardinality
        # and the exact code-frequency distribution while destroying the link
        # between a row and its true entity. A *bijective* code remap would not
        # work: relabelling embedding rows is a symmetry of the model and trains
        # to an identical result.
        "shuffle_fields": dict(shuffle_fields or {}),
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
        shuffle_seed = id_maps.get("shuffle_fields", {}).get(fieldname)
        if shuffle_seed is not None:
            codes = np.random.default_rng(shuffle_seed).permutation(codes)
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
    user_weight_decay: float = 0.0,
    early_stop_mask: np.ndarray | None = None,
    user_id_mode: str = "real",
    loss_mode: str = "bce",
    bpr_alpha: float = 1.0,
    aux_labels: list[str] | None = None,
    aux_weight: float = 0.3,
) -> TrainResult:
    """`user_weight_decay` adds decay on the `user_id` rows of the shared
    embedding table, ON TOP OF the global `weight_decay` (Adam applies decay
    per-tensor, and the user rows live inside one shared table, so they cannot
    be excluded from the global term). It is expressed in Adam's own units: the
    penalty is 0.5 * wd * ||W_user||^2, whose gradient is wd * W_user, matching
    what `weight_decay` adds to the gradient.

    Only the k-dim `embedding` rows are penalised, not the first-order `linear`
    rows: a per-user constant added to the logit provably cannot change a
    within-user ranking, so decaying it could not move GAUC or nDCG.

    `early_stop_mask` is a boolean mask over validation ROWS used for the
    early-stopping metric. Pass the selection half during a sweep so held-out
    confirmation users never participate in model selection. None = all rows
    (the shipped default).

    `user_id_mode` decomposes what the user embedding contributes:
      "real"     — shipped model.
      "shuffled" — user codes permuted across rows in every split. Same
                   parameter count, same code-frequency distribution, zero
                   user identity. If this matches "real", the embedding is
                   pure capacity; if it matches "removed", identity is the
                   whole story. The `user_id` COLUMN is untouched, so metric
                   grouping still uses each row's true user.
      "removed"  — user_id dropped from the encoded field list entirely.
    """
    if user_id_mode not in ("real", "shuffled", "removed"):
        raise ValueError(f"user_id_mode must be real/shuffled/removed, got {user_id_mode!r}")
    if loss_mode not in ("bce", "bpr", "hybrid"):
        raise ValueError(f"loss_mode must be bce/bpr/hybrid, got {loss_mode!r}")
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

    id_maps = _build_id_maps(
        train_feat, cross_state,
        drop_fields={"user_id"} if user_id_mode == "removed" else None,
        # Distinct seeds per split so the permutations are independent; both are
        # derived from `seed` so the whole arm stays reproducible.
        shuffle_fields={"user_id": 90_000 + seed} if user_id_mode == "shuffled" else None,
    )

    x_train = torch.tensor(_encode(train_feat, id_maps), dtype=torch.long)
    y_train = torch.tensor(y_np, dtype=torch.float32)
    val_label = resolve_label(val_feat)

    # --- within-user pairwise sampling -------------------------------------
    # GAUC is a within-user AUC, and AUC is exactly the probability that a
    # positive outranks a negative FROM THE SAME USER. Pointwise BCE optimises
    # calibration and reaches that objective only indirectly; a pair whose two
    # rows share a user is a direct sample of the quantity being scored.
    #
    # Pairs are built once (as offsets into a user-sorted index) and RESAMPLED
    # every epoch, so the model sees many different (pos, neg) combinations
    # without ever materialising the full O(sum n_pos * n_neg) pair set.
    #
    # Users with no positive or no negative training row are dropped: they
    # generate no pair, which mirrors the metric itself — GAUC is undefined for
    # a single-class user and the official evaluator excludes those users.
    pos_rows = pos_owner = neg_rows = neg_off = neg_cnt = None
    if loss_mode in ("bpr", "hybrid"):
        _tru = train_feat["user_id"].to_numpy()
        _ordr = np.argsort(_tru, kind="stable")
        _su, _sy = _tru[_ordr], y_np[_ordr]
        _bnd = np.flatnonzero(np.r_[True, _su[1:] != _su[:-1]])
        _p_chunks, _n_chunks, _owner, _offs, _cnts = [], [], [], [], []
        _running = 0
        for _s, _e in zip(_bnd, np.r_[_bnd[1:], len(_su)]):
            _blk, _yb = _ordr[_s:_e], _sy[_s:_e]
            _p, _n = _blk[_yb > 0.5], _blk[_yb <= 0.5]
            if len(_p) == 0 or len(_n) == 0:
                continue
            _p_chunks.append(_p)
            _n_chunks.append(_n)
            _owner.append(np.full(len(_p), len(_offs), dtype=np.int64))
            _offs.append(_running)
            _cnts.append(len(_n))
            _running += len(_n)
        if not _p_chunks:
            raise ValueError("loss_mode requires pairs but no user has both classes")
        pos_rows = np.concatenate(_p_chunks)
        pos_owner = np.concatenate(_owner)
        neg_rows = np.concatenate(_n_chunks)
        neg_off = np.asarray(_offs, dtype=np.int64)
        neg_cnt = np.asarray(_cnts, dtype=np.int64)
        print(f"[train] loss_mode={loss_mode}: {len(pos_rows):,} pairable positives "
              f"across {len(neg_off):,} two-class users "
              f"({len(pos_rows) / len(y_np):.1%} of train rows)")

    # Auxiliary TARGETS (never inputs). These columns are outcomes of the
    # impression being predicted, so using them as features would be label
    # leakage; used as extra heads they only shape the shared embedding during
    # training and are absent from every inference path.
    y_aux = None
    if aux_labels:
        missing = [c for c in aux_labels if c not in train_feat.columns]
        if missing:
            raise ValueError(f"aux_labels not in training frame: {missing}")
        if model_type == "dcnv2":
            raise ValueError("aux_labels is implemented for the FM/MLP model only")
        y_aux = torch.tensor(
            train_feat[list(aux_labels)].to_numpy(dtype=np.float32),
            dtype=torch.float32)
        print(f"[train] aux heads {list(aux_labels)} weight={aux_weight}")

    num_fields = len(id_maps["fields"])
    if model_type == "dcnv2":
        model = DCNv2(total_dim=id_maps["total_dim"], num_fields=num_fields, k=embed_dim,
                      cross_layers=cross_layers, rank=cross_rank).to(device)
    else:
        model = FactorizationMachineWithMLP(
            total_dim=id_maps["total_dim"], num_fields=num_fields, k=embed_dim,
            use_fm=model_type in ("deepfm", "fm"),
            use_mlp=model_type in ("deepfm", "mlp"),
            n_aux=0 if y_aux is None else y_aux.shape[1],
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    user_slice = None
    if user_weight_decay > 0.0:
        if "user_id" not in id_maps["fields"]:
            raise ValueError("user_weight_decay set but user_id is not an encoded field")
        _lo = id_maps["offsets"]["user_id"]
        _hi = _lo + len(id_maps["vocabs"]["user_id"]) + 1  # +1 UNK slot
        user_slice = (_lo, _hi)

    es_mask = None if early_stop_mask is None else np.asarray(early_stop_mask, dtype=bool)
    if es_mask is not None and len(es_mask) != len(val_feat):
        raise ValueError(f"early_stop_mask has {len(es_mask)} rows, val has {len(val_feat)}")

    generator = torch.Generator().manual_seed(seed)
    n = len(y_train)
    epoch_losses: list[float] = []
    best_primary, best_state, bad_epochs = -1.0, None, 0

    pair_rng = np.random.default_rng(seed + 777)
    n_batches_total = (n + batch_size - 1) // batch_size

    for _epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)

        # Resample one negative per pairable positive, drawn from that SAME
        # user's negatives. Fresh every epoch: the pair set is combinatorially
        # large and re-drawing is what stops the model fitting one arbitrary
        # sample of it.
        pair_a = pair_b = None
        pair_bs = 0
        if pos_rows is not None:
            _j = neg_off[pos_owner] + (pair_rng.random(len(pos_rows))
                                       * neg_cnt[pos_owner]).astype(np.int64)
            _shuf = pair_rng.permutation(len(pos_rows))
            pair_a = torch.as_tensor(pos_rows[_shuf], dtype=torch.long)
            pair_b = torch.as_tensor(neg_rows[_j][_shuf], dtype=torch.long)
            pair_bs = (len(pos_rows) + n_batches_total - 1) // n_batches_total

        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()

            point = None
            if loss_mode in ("bce", "hybrid"):
                if y_aux is None:
                    point = loss_fn(model(x_train[idx].to(device)),
                                    y_train[idx].to(device))
                else:
                    _main, _ax = model(x_train[idx].to(device), with_aux=True)
                    point = (loss_fn(_main, y_train[idx].to(device))
                             + aux_weight * loss_fn(_ax, y_aux[idx].to(device)))

            pair = None
            if pair_a is not None:
                _o = n_batches * pair_bs
                _sa, _sb = pair_a[_o:_o + pair_bs], pair_b[_o:_o + pair_bs]
                if len(_sa) > 0:
                    # BPR: -log sigma(s_pos - s_neg). Its gradient pushes the
                    # two scores apart, which is precisely what a within-user
                    # AUC counts, and it is invariant to any per-user constant
                    # (which cannot reorder that user's candidates anyway).
                    _delta = (model(x_train[_sa].to(device))
                              - model(x_train[_sb].to(device)))
                    pair = -nn.functional.logsigmoid(_delta).mean()

            if point is not None and pair is not None:
                loss = point + bpr_alpha * pair
            elif pair is not None:
                loss = pair
            elif point is not None:
                loss = point
            else:  # pure-bpr epoch that ran out of pairs; fall back pointwise
                loss = loss_fn(model(x_train[idx].to(device)),
                               y_train[idx].to(device))

            if user_slice is not None:
                w_user = model.embedding.weight[user_slice[0]:user_slice[1]]
                loss = loss + 0.5 * user_weight_decay * (w_user ** 2).sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        epoch_losses.append(total_loss / max(n_batches, 1))

        # Early stopping on the competition's own primary metric, not on loss —
        # the same rule (patience on validation primary) the official baseline uses.
        metrics = _evaluate(model, val_feat, id_maps, val_label, device, row_mask=es_mask)
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

    final_scores = score_dataframe(model, id_maps, val_feat, device=device)
    final_eval = val_feat[["user_id"]].copy()
    final_eval["score"] = final_scores
    final_eval["label"] = val_label.to_numpy()

    return TrainResult(
        model=model,
        id_maps=id_maps,
        val_metrics=compute_ranking_metrics(final_eval),
        epoch_losses=epoch_losses,
        val_scores=final_scores,
        val_user_ids=val_feat["user_id"].to_numpy(),
        val_labels=val_label.to_numpy(),
    )


def _evaluate(model: nn.Module, feat_df: pd.DataFrame, id_maps: dict, label: pd.Series, device: str,
              row_mask: np.ndarray | None = None) -> RankingMetrics:
    eval_df = feat_df[["user_id"]].copy()
    eval_df["score"] = score_dataframe(model, id_maps, feat_df, device=device)
    eval_df["label"] = label.to_numpy()
    if row_mask is not None:
        eval_df = eval_df[row_mask]
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

"""
Evaluation protocol — the instrument that decides whether a candidate is real.

Motivation (docs/research_process.md section 9): the `mlp`-only candidate scored
0.6051 vs 0.6047, passed an 8-seed paired test at t = 4.49, and was still noise.
Seed repetition closes exactly one of four threats:

    (a) seed noise           -> paired multi-seed test        [closed by seeds]
    (b) user sampling error  -> bootstrap over users          [this module]
    (c) temporal transfer    -> split validation by date      [this module]
    (d) selection bias       -> hold out users from selection [this module]

Threat (d) is the one that is currently unbounded: ~30 configurations have been
compared against the whole validation split, and the maximum of 30 noisy
comparisons is biased upward even when every true difference is zero. So this
module partitions validation BY USER HASH into a selection half and a
confirmation half. All exploration happens on selection; confirmation is looked
at only after a candidate has already won, and every look spends some of its
independence.

The split is by user hash rather than by date on purpose — date must stay intact
so that threat (c) can be tested separately, and the metric aggregates over
users, so a user-level split keeps both halves unbiased estimates of the same
quantity.

Organizer-rule note: FAQ 2.9.1(a) permits a team to declare its own protocol
provided the values are fixed before the run and recorded in the run log. The
salt and fractions below are those fixed values — do not tune them.

Everything here is a DECOMPOSITION of `starter_kit/evaluate.py`, never a
reimplementation of the metric. `verify_decomposition()` asserts the identity
against the official script and is run by `python -m pipeline.eval_protocol`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from pipeline.evaluate import _official_evaluate

# Fixed before the run and recorded here. Changing the salt reshuffles the
# halves, which would silently re-open threat (d) — treat as immutable.
SPLIT_SALT = b"kuairand-pure-eval-protocol-v1"
SELECTION_FRACTION = 0.5

K = 5


@dataclass
class PerUserStats:
    """Per-user sufficient statistics for GAUC and nDCG@5.

    GAUC is a positive-count-weighted mean of per-user AUCs over *eligible*
    users (0 < positives < impressions); nDCG@5 is a plain mean over ALL users.
    Holding these per user makes both metrics exactly re-aggregable over any
    subset or bootstrap resample of users.
    """

    user_ids: np.ndarray      # (n_users,)
    npos: np.ndarray          # (n_users,) positive count
    auc: np.ndarray           # (n_users,) per-user AUC, 0.0 where not eligible
    eligible: np.ndarray      # (n_users,) bool, 0 < npos < n_impressions
    ndcg: np.ndarray          # (n_users,) per-user nDCG@5


def _auc_one(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U with tie correction — mirrors starter_kit.evaluate.auc."""
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    n = len(s)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = float(y.sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return 0.5
    srank = float(ranks[y == 1].sum())
    return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)


def _ndcg_one(labels_desc: np.ndarray, k: int = K) -> float:
    """`labels_desc` already sorted by descending score."""
    disc = np.log2(np.arange(k) + 2.0)
    top = labels_desc[:k]
    dcg = float((((2.0 ** top) - 1) / disc[: len(top)]).sum())
    ideal = np.sort(labels_desc)[::-1][:k]
    idcg = float((((2.0 ** ideal) - 1) / disc[: len(ideal)]).sum())
    return 0.0 if idcg == 0 else dcg / idcg


def per_user_stats(user_ids, labels, scores, k: int = K) -> PerUserStats:
    """Decompose a scored split into per-user sufficient statistics."""
    user_ids = np.asarray(user_ids)
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(user_ids, kind="mergesort")
    u_sorted, l_sorted, s_sorted = user_ids[order], labels[order], scores[order]
    uniq, starts = np.unique(u_sorted, return_index=True)
    bounds = np.append(starts, len(u_sorted))

    n_users = len(uniq)
    npos = np.zeros(n_users)
    aucs = np.zeros(n_users)
    elig = np.zeros(n_users, dtype=bool)
    ndcgs = np.zeros(n_users)

    for i in range(n_users):
        lo, hi = bounds[i], bounds[i + 1]
        y, sc = l_sorted[lo:hi], s_sorted[lo:hi]
        p = float(y.sum())
        npos[i] = p
        # Descending score; mergesort on the negated score keeps ties stable in
        # the same order the official script's `sort(key=-score)` produces.
        desc = np.argsort(-sc, kind="mergesort")
        ndcgs[i] = _ndcg_one(y[desc], k)
        if 0 < p < (hi - lo):
            elig[i] = True
            aucs[i] = _auc_one(y, sc)

    return PerUserStats(uniq, npos, aucs, elig, ndcgs)


def aggregate(stats: PerUserStats, idx: np.ndarray | None = None) -> dict:
    """Re-aggregate GAUC / nDCG@5 / primary over a subset or resample of users.

    `idx` is an index array into the per-user arrays; it may contain repeats,
    which is what makes the bootstrap exact.
    """
    npos = stats.npos if idx is None else stats.npos[idx]
    auc = stats.auc if idx is None else stats.auc[idx]
    elig = stats.eligible if idx is None else stats.eligible[idx]
    ndcg = stats.ndcg if idx is None else stats.ndcg[idx]

    den = float(npos[elig].sum())
    gauc = float((npos[elig] * auc[elig]).sum() / den) if den else 0.5
    nd = float(ndcg.mean()) if len(ndcg) else 0.0
    return {"GAUC": gauc, f"nDCG@{K}": nd, "primary": (gauc + nd) / 2.0}


# --------------------------------------------------------------------------
# Threat (d): selection / confirmation split by user hash
# --------------------------------------------------------------------------

def user_half(user_ids, salt: bytes = SPLIT_SALT) -> np.ndarray:
    """Stable per-user assignment to the selection half (True) or confirmation.

    Uses blake2b, NOT Python's builtin hash(), which is randomised per process
    and would silently reshuffle the halves between runs.
    """
    uniq = np.unique(np.asarray(user_ids))
    cut = int(SELECTION_FRACTION * (2 ** 32))
    sel = {}
    for u in uniq:
        h = hashlib.blake2b(str(u).encode("utf-8"), digest_size=4, key=salt).digest()
        sel[u] = int.from_bytes(h, "big") < cut
    return np.array([sel[u] for u in np.asarray(user_ids)], dtype=bool)


def split_stats(stats: PerUserStats, salt: bytes = SPLIT_SALT) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays for (selection, confirmation) users of a PerUserStats."""
    is_sel = user_half(stats.user_ids, salt)
    return np.where(is_sel)[0], np.where(~is_sel)[0]


# --------------------------------------------------------------------------
# Threat (b): bootstrap over users
# --------------------------------------------------------------------------

def bootstrap_delta(
    stats_a: PerUserStats,
    stats_b: PerUserStats,
    idx: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """User-level bootstrap CI on primary(b) - primary(a).

    Both arms must be scored on the SAME users in the same order — resampling
    is paired at the user level, so the same users are drawn for both arms.
    """
    if not np.array_equal(stats_a.user_ids, stats_b.user_ids):
        raise ValueError("bootstrap requires both arms scored on the same users")

    pool = np.arange(len(stats_a.user_ids)) if idx is None else np.asarray(idx)
    rng = np.random.default_rng(seed)
    point = aggregate(stats_b, pool)["primary"] - aggregate(stats_a, pool)["primary"]

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        draw = pool[rng.integers(0, len(pool), len(pool))]
        deltas[i] = aggregate(stats_b, draw)["primary"] - aggregate(stats_a, draw)["primary"]

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_users": int(len(pool)),
        "n_boot": n_boot,
    }


def verify_decomposition(user_ids, labels, scores, tol: float = 1e-12) -> float:
    """Assert this module reproduces starter_kit/evaluate.py exactly.

    Returns the max absolute discrepancy. Run this before trusting any
    bootstrap built on the decomposition.
    """
    official = _official_evaluate(list(user_ids), list(labels), list(scores), k=K)
    ours = aggregate(per_user_stats(user_ids, labels, scores))
    diffs = [abs(official[m] - ours[m]) for m in ("GAUC", f"nDCG@{K}", "primary")]
    worst = max(diffs)
    if worst > tol:
        raise AssertionError(
            f"decomposition does NOT match starter_kit/evaluate.py (max diff {worst:.3e}).\n"
            f"  official: {official}\n  ours:     {ours}"
        )
    return worst


def main() -> None:
    from pipeline.data.label import resolve_label
    from pipeline.data.loader import load_split

    split = load_split()
    val = split.val
    y = resolve_label(val).to_numpy()
    users = val["user_id"].to_numpy()

    rng = np.random.default_rng(0)
    worst = 0.0
    for trial, name in enumerate(["random", "label-correlated", "many-ties"]):
        if name == "random":
            sc = rng.normal(size=len(y))
        elif name == "label-correlated":
            sc = y + rng.normal(scale=0.5, size=len(y))
        else:
            sc = rng.integers(0, 3, size=len(y)).astype(float)
        d = verify_decomposition(users, y, sc)
        worst = max(worst, d)
        print(f"  {name:<18} max |diff| vs official = {d:.3e}")

    stats = per_user_stats(users, y, rng.normal(size=len(y)))
    sel, conf = split_stats(stats)
    print(f"\ndecomposition verified against starter_kit/evaluate.py: max diff {worst:.3e}")
    print(f"validation users: {len(stats.user_ids):,}")
    print(f"  selection half:    {len(sel):,} users ({len(sel)/len(stats.user_ids):.1%})")
    print(f"  confirmation half: {len(conf):,} users ({len(conf)/len(stats.user_ids):.1%})")
    print(f"  salt: {SPLIT_SALT.decode()}  (fixed before the run — do not tune)")


if __name__ == "__main__":
    main()

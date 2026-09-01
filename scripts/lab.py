"""Overnight research harness.

Two ideas make this fast enough to actually search:

  1. CACHE THE SCORE VECTOR, not the model. Every experiment writes its
     validation score vector to .npy. Blending, reweighting and rank-averaging
     experiments then cost milliseconds instead of a retrain, so the expensive
     axis (architectures x seeds) is paid once and the cheap axis (how to
     combine them) can be searched exhaustively.

  2. SCORE ON THE SELECTION HALF ONLY. Everything here reports selection-half
     primary. The confirmation half is not read by this file at all — see
     `confirm.py` for the one place it is, deliberately, spent.

NOISE FLOOR. The single-seed std on this task is 0.00049 (20-seed pool), so a
difference of 0.0005 between two single runs is meaningless and a difference
under ~0.0015 is not evidence of anything. Every comparison here is
seed-matched and reports mean +/- std over >= 3 seeds. `delta()` refuses to
call anything a win that does not clear 2 standard errors.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE = Path(os.environ.get(
    "LAB_CACHE",
    Path(__file__).resolve().parents[1] / ".lab_cache",
))
CACHE.mkdir(parents=True, exist_ok=True)

_GEOM = {}


def geometry():
    """user_ids / labels / selection mask for the validation frame, in the
    row order run_training returns (build_features sorts by user_id)."""
    if not _GEOM:
        from pipeline.data.features import build_features
        from pipeline.data.loader import load_split
        from pipeline.eval_protocol import user_half
        val = build_features(load_split().val)
        from pipeline.data.label import resolve_label
        _GEOM["users"] = val["user_id"].to_numpy()
        _GEOM["labels"] = resolve_label(val).to_numpy().astype(np.float64)
        _GEOM["sel"] = user_half(_GEOM["users"])
    return _GEOM["users"], _GEOM["labels"], _GEOM["sel"]


def score_on(scores, half="sel"):
    """Primary metric on one half of validation, via the official evaluator."""
    from pipeline.eval_protocol import aggregate, per_user_stats
    users, labels, sel = geometry()
    mask = sel if half == "sel" else (~sel if half == "conf" else np.ones_like(sel))
    st = per_user_stats(users[mask], labels[mask], np.asarray(scores)[mask])
    return aggregate(st)


def key(**kw):
    canon = json.dumps(kw, sort_keys=True, default=str)
    return hashlib.blake2b(canon.encode(), digest_size=8).hexdigest()


def run(tag, seed, **kw):
    """Train (or load from cache) and return the validation score vector."""
    path = CACHE / f"{tag}_{key(**kw)}_s{seed}.npy"
    if path.exists():
        return np.load(path)
    from pipeline.data.features import build_features
    from pipeline.data.loader import load_split
    from pipeline.eval_protocol import user_half
    from pipeline.train import run_training
    val = build_features(load_split().val)
    es = user_half(val["user_id"].to_numpy())
    res = run_training(seed=seed, early_stop_mask=es, **kw)
    v = np.asarray(res.val_scores, dtype=np.float64)
    np.save(path, v)
    return v


def ranks(v):
    """Rank-transform. Only within-user ORDER is scored and ranks are
    scale-free, so no model's score distribution can dominate a blend."""
    v = np.asarray(v)
    order = v.argsort()
    out = np.empty(len(v), dtype=np.float64)
    out[order] = np.arange(len(v))
    return out


def arm(tag, seeds=(0, 1, 2), half="sel", **kw):
    """Seed-matched arm. Returns (per-seed primaries, rank-averaged primary)."""
    vs = [run(tag, s, **kw) for s in seeds]
    singles = [score_on(v, half)["primary"] for v in vs]
    ens = score_on(np.mean([ranks(v) for v in vs], axis=0), half)["primary"]
    return np.array(singles), ens, vs


def delta(a, b, label_a="A", label_b="B"):
    """Seed-matched paired comparison. Reports the paired difference and its
    standard error; calls a win only at 2 SE. Paired because the seeds are
    matched, which removes most of the between-seed variance."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    verdict = "WIN" if len(d) > 1 and d.mean() > 2 * se else (
        "LOSS" if len(d) > 1 and d.mean() < -2 * se else "null")
    print(f"  {label_a} {a.mean():.5f}+/-{a.std(ddof=1):.5f} | "
          f"{label_b} {b.mean():.5f}+/-{b.std(ddof=1):.5f} | "
          f"paired {d.mean():+.5f} (SE {se:.5f}) -> {verdict}")
    return d.mean(), se, verdict


def report(name, singles, ens):
    print(f"  {name:38s} single {singles.mean():.5f}+/-{singles.std(ddof=1):.5f} "
          f"n={len(singles)} | ens {ens:.5f}")

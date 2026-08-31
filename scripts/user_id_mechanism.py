"""
Step 2 — what is `user_id` actually doing?

The record contains two results that look contradictory:

  * deleting user_id costs -0.0091
  * user x video affinity is chance (0.4970 / 0.4973 / 0.4981, measured 3x)

So the user embedding earns its -0.0091 through something that is not
affinity. Three candidate explanations, and this script separates them:

  H1 IDENTITY   the embedding stores genuine per-user behavioural information
  H2 CAPACITY   the parameters help regardless of what they encode; any
                extra input of the same cardinality would do as well
  H3 MEMORISED  it memorises frequent users' label rates and does not
                generalise

Arms (one conceptual change each):

  real       shipped model
  shuffled   user codes permuted ACROSS ROWS in every split. Identical
             parameter count, identical code-frequency distribution, zero
             user identity. NOTE this is a row-level permutation, not a
             relabelling of embedding rows — a bijective remap is a symmetry
             of the model and would train to an identical result.
  removed    user_id dropped from the encoded field list

Reading the result:

  shuffled ~= real      -> H2. The embedding is capacity, not information.
  shuffled ~= removed   -> H1. Identity carries essentially all of the gain.
  in between            -> both, in the proportion shown.

H3 is separated by the frequency stratification at the end: if the gain is
memorisation, the gap must concentrate in users with many training impressions.
Stratify against `removed`, not `shuffled` — shuffling can hand a rare user a
heavily-trained embedding, which inflates the low-frequency bins. A flat
profile across frequency bins argues against memorisation.

Run:  python scripts/user_id_mechanism.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.data.features import build_features        # noqa: E402
from pipeline.data.loader import load_split              # noqa: E402
from pipeline.eval_protocol import (aggregate, per_user_stats,  # noqa: E402
                                    split_stats, user_half)
from pipeline.train import run_training                  # noqa: E402

ARMS = ["real", "shuffled", "removed"]
SEEDS = [0, 1, 2]
FREQ_BINS = [0, 1, 5, 15, 40, 10 ** 9]
OUT = REPO_ROOT / "logs" / "user_id_mechanism.json"


def main() -> None:
    split = load_split()
    val_feat = build_features(split.val)
    es_mask = user_half(val_feat["user_id"].to_numpy())

    probe = per_user_stats(val_feat["user_id"].to_numpy(),
                           np.zeros(len(val_feat)), np.zeros(len(val_feat)))
    sel_idx, _ = split_stats(probe)

    # Training impressions per user — the axis H3 predicts the gain lives on.
    train_counts = split.train["user_id"].value_counts()
    freq = np.array([train_counts.get(u, 0) for u in probe.user_ids])

    records, stats_by_arm = [], {}
    t0 = time.time()
    for arm in ARMS:
        per_seed = []
        for seed in SEEDS:
            r = run_training(split=split, seed=seed, user_id_mode=arm,
                             early_stop_mask=es_mask)
            st = per_user_stats(r.val_user_ids, r.val_labels, r.val_scores)
            full = aggregate(st)["primary"]
            sel = aggregate(st, sel_idx)["primary"]
            per_seed.append(st)
            records.append({"arm": arm, "seed": seed, "full": full, "selection": sel})
            print(f"  {arm:<9} seed={seed}  full={full:.4f}  selection={sel:.4f}", flush=True)
        stats_by_arm[arm] = per_seed
        fulls = [r["full"] for r in records if r["arm"] == arm]
        print(f"{arm:<9}  FULL mean={np.mean(fulls):.4f} std={np.std(fulls):.4f}"
              f"   [{time.time()-t0:.0f}s]\n", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")

    mean = {a: float(np.mean([r["full"] for r in records if r["arm"] == a])) for a in ARMS}
    print("=" * 70)
    for a in ARMS:
        print(f"{a:>10}  full primary {mean[a]:.4f}   vs real {mean[a]-mean['real']:+.4f}")
    print("=" * 70)

    span = mean["real"] - mean["removed"]
    if abs(span) < 1e-6:
        print("\nreal and removed are identical — nothing to attribute.")
        return
    identity_share = (mean["real"] - mean["shuffled"]) / span
    print(f"\ntotal user_id contribution (real - removed): {span:+.4f}")
    print(f"attributable to IDENTITY  (real - shuffled): {mean['real']-mean['shuffled']:+.4f}"
          f"  = {identity_share:.0%} of it")
    print(f"attributable to CAPACITY  (shuffled - removed): {mean['shuffled']-mean['removed']:+.4f}"
          f"  = {1-identity_share:.0%} of it")

    # H3: does the identity gain concentrate in frequently-seen users?
    #
    # Stratify against BOTH references. `shuffled` is CONFOUNDED for this
    # question: it preserves the code-frequency distribution while reassigning
    # rows, so a rare user can receive a heavily-trained embedding belonging to
    # some frequent user. That is actively misleading input rather than absent
    # input, and it inflates the apparent gain exactly in the low-frequency bins
    # the conclusion rests on. `removed` gives every user no user-input at all,
    # so it is the clean reference here. Read the vs-rem column.
    print("")
    print("Identity gain by training frequency (full val, 3-seed means):")
    hdr = ("train impressions", "users", "real", "shuf", "rem", "vs shuf", "vs rem")
    print(f"  {hdr[0]:>18} {hdr[1]:>7} {hdr[2]:>8} {hdr[3]:>8} {hdr[4]:>8} {hdr[5]:>9} {hdr[6]:>9}")
    for lo, hi in zip(FREQ_BINS[:-1], FREQ_BINS[1:]):
        idx = np.where((freq > lo) & (freq <= hi))[0] if lo else np.where(freq <= hi)[0]
        if len(idx) < 200:
            continue
        r_p = float(np.mean([aggregate(st, idx)["primary"] for st in stats_by_arm["real"]]))
        s_p = float(np.mean([aggregate(st, idx)["primary"] for st in stats_by_arm["shuffled"]]))
        d_p = float(np.mean([aggregate(st, idx)["primary"] for st in stats_by_arm["removed"]]))
        band = 0.0008 * (22377 / len(idx)) ** 0.5
        label = f"{lo+1}-{hi}" if hi < 10 ** 8 else f"{lo+1}+"
        print(f"  {label:>18} {len(idx):>7,} {r_p:>8.4f} {s_p:>8.4f} {d_p:>8.4f}"
              f" {r_p-s_p:>+9.4f} {r_p-d_p:>+9.4f}   (+/-{band:.4f})")
    print("")
    print("Read the vs-rem column. A gain RISING with frequency argues memorisation;")
    print("a flat profile argues a transferable per-user prior learnable from few rows.")
    print("The +/- column is that bin's noise band, 0.0008*sqrt(22377/n) — small bins are wide.")


if __name__ == "__main__":
    main()

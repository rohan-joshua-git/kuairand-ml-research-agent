"""Batch 2 — does a ranking objective beat a pointwise one?

GAUC is a within-user AUC. AUC is exactly P(s_pos > s_neg) for a pos/neg pair
from the same user. We train pointwise BCE, which optimises calibration and
only indirectly produces the ordering the metric scores. This batch asks
whether closing that mismatch is worth anything.

First read (seed 0, selection half): base 0.60805, pure BPR 0.60488,
hybrid@alpha=1 0.60516. Both losses. But alpha=1 is arbitrary and pure BPR
consumes only 33.5% as many samples per step as the pointwise path, so it may
simply be under-trained. This sweeps alpha and gives the pairwise arms a longer
epoch budget before drawing any conclusion.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab

SEEDS = (0, 1, 2)
t0 = time.time()
res = {}


def go(name, seeds=SEEDS, **kw):
    try:
        singles, ens, _ = lab.arm(name, seeds=seeds, **kw)
    except Exception as exc:
        print(f"  {name:34s} FAILED: {type(exc).__name__}: {exc}", flush=True)
        return None
    lab.report(name, singles, ens)
    res[name] = singles
    return singles


print("=" * 72, flush=True)
print("BATCH 2  ranking objective (selection half, 3 seeds)", flush=True)
print("=" * 72, flush=True)

base = go("base")

print("\n-- hybrid BCE + alpha*BPR --", flush=True)
for a in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
    go(f"hyb{a:g}", loss_mode="hybrid", bpr_alpha=a)

print("\n-- pure BPR, longer budget (33.5% samples/step vs pointwise) --", flush=True)
go("bpr_e12", loss_mode="bpr")
go("bpr_e25", loss_mode="bpr", epochs=25, patience=5)
go("bpr_e25_lr2", loss_mode="bpr", epochs=25, patience=5, lr=2e-3)

print("\n-- best hybrid, longer budget --", flush=True)
go("hyb0.5_e20", loss_mode="hybrid", bpr_alpha=0.5, epochs=20, patience=5)
go("hyb0.25_e20", loss_mode="hybrid", bpr_alpha=0.25, epochs=20, patience=5)

print("\n" + "=" * 72, flush=True)
if base is not None:
    print("PAIRED vs base:", flush=True)
    for name, s in sorted(res.items(), key=lambda x: -x[1].mean()):
        if name != "base":
            lab.delta(s, base, name, "base")
print(f"\nbatch 2 wall clock: {(time.time() - t0) / 60:.1f} min", flush=True)

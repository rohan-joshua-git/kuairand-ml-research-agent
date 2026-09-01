"""Batch 4 — auxiliary multi-task heads.

long_view fires on 31.3% of training rows. is_click fires on 46.3% of the same
rows and is a strictly related outcome. The hypothesis: predicting the denser
signal from the SHARED embedding regularises the representation that the
long_view head reads from.

These columns are outcomes of the impression being predicted, so as INPUTS they
would be label leakage. They are used only as training-time targets; the
auxiliary head is absent from every inference path, and `forward` returns the
main logit unless explicitly asked for aux. That distinction is the whole
reason this is legitimate.

The rare labels (is_follow 0.1%, is_forward 0.1%, is_hate 0.04%) are tested
both in and out, because a head on a 0.04%-positive target mostly contributes
gradient noise.
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
        print(f"  {name:30s} FAILED: {type(exc).__name__}: {exc}", flush=True)
        return None
    lab.report(name, singles, ens)
    res[name] = singles
    return singles


print("=" * 72, flush=True)
print("BATCH 4  auxiliary multi-task heads (selection half, 3 seeds)", flush=True)
print("=" * 72, flush=True)

base = go("base")

print("\n-- is_click only, weight sweep --", flush=True)
for w in (0.1, 0.3, 1.0, 2.0):
    go(f"aux_click_w{w:g}", aux_labels=["is_click"], aux_weight=w)

print("\n-- click + like (the two non-trivial rates) --", flush=True)
for w in (0.3, 1.0):
    go(f"aux_cl_w{w:g}", aux_labels=["is_click", "is_like"], aux_weight=w)

print("\n-- all six engagement outcomes --", flush=True)
for w in (0.3, 1.0):
    go(f"aux_all_w{w:g}",
       aux_labels=["is_click", "is_like", "is_follow",
                   "is_comment", "is_forward", "is_hate"], aux_weight=w)

print("\n" + "=" * 72, flush=True)
if base is not None:
    print("PAIRED vs base:", flush=True)
    for name, s in sorted(res.items(), key=lambda x: -x[1].mean()):
        if name != "base":
            lab.delta(s, base, name, "base")
print(f"\nbatch 4 wall clock: {(time.time() - t0) / 60:.1f} min", flush=True)

"""Batch 1 — capacity, regularisation and architecture, on the SELECTION half.

No code changes needed for any of this: every knob here is already a
run_training parameter. The point of running it first is that it fills the
score-vector cache, which makes every later blending experiment free.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab

SEEDS = (0, 1, 2)
t0 = time.time()
results = {}


def go(name, seeds=SEEDS, **kw):
    try:
        singles, ens, _ = lab.arm(name, seeds=seeds, **kw)
    except Exception as exc:
        print(f"  {name:38s} FAILED: {type(exc).__name__}: {exc}")
        return None
    lab.report(name, singles, ens)
    results[name] = (singles, ens, kw)
    return singles


print("=" * 72)
print("BATCH 1  (selection half; 3 seeds unless noted; noise floor ~0.0015)")
print("=" * 72)

print("\n-- reference --")
base = go("base", seeds=(0, 1, 2, 3, 4))

print("\n-- embedding capacity --")
for k in (8, 24, 32, 48, 64):
    go(f"k{k}", embed_dim=k)

print("\n-- learning rate --")
for lr in (5e-4, 2e-3, 3e-3):
    go(f"lr{lr:g}", lr=lr)

print("\n-- global weight decay --")
for wd in (1e-5, 1e-4, 1e-3):
    go(f"wd{wd:g}", weight_decay=wd)

print("\n-- longer training / more patience --")
go("ep20p5", epochs=20, patience=5)
go("ep30p6", epochs=30, patience=6)

print("\n-- batch size --")
for bs in (2048, 4096, 16384):
    go(f"bs{bs}", batch_size=bs)

print("\n-- architecture --")
go("dcn3", model_type="dcnv2", cross_layers=3)
go("dcn2", model_type="dcnv2", cross_layers=2)
go("fm_only", model_type="fm")
go("mlp_only", model_type="mlp")

print("\n" + "=" * 72)
if base is not None:
    print("PAIRED vs base (first 3 seeds):")
    for name, (singles, ens, kw) in sorted(results.items(),
                                           key=lambda x: -x[1][0].mean()):
        if name == "base" or len(singles) != 3:
            continue
        lab.delta(singles, base[:3], name, "base")
print(f"\nbatch 1 wall clock: {(time.time() - t0) / 60:.1f} min")

"""
Fetches / verifies KuaiRand-Pure locally.

The core dataset IS directly fetchable — the Starter Kit README
(`starter_kit/README.md`) confirms and this was verified working (single
~47MB tarball, no registration required):
    https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
`--fetch` downloads and unpacks it straight into `raw_dir`.

The Zenodo record for the *supplementary* files (video captions + video
category taxonomy) is a separate, stable deposit:
https://zenodo.org/records/18159199

Usage:
    python -m pipeline.data.download --check        # verify raw_dir has what we need
    python -m pipeline.data.download --fetch         # download + unpack the core dataset
    python -m pipeline.data.download --supplements   # fetch the Zenodo caption/category files
"""
import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agent_config.yaml"

KUAIRAND_PURE_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
ZENODO_SUPPLEMENT_BASE = "https://zenodo.org/records/18159199/files"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def required_files(cfg: dict) -> list[str]:
    logs = cfg["dataset"]["logs"]
    feats = cfg["dataset"]["features"]
    return [logs["standard_train"], logs["standard_evalset"], logs["random_unbiased"],
            feats["video_basic"], feats["video_statistic"], feats["user_features"]]


def check(cfg: dict) -> bool:
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    missing = [f for f in required_files(cfg) if not (raw_dir / f).exists()]
    if missing:
        print(f"[download] Missing from {raw_dir}:")
        for m in missing:
            print(f"  - {m}")
        print(f"\nRun `python -m pipeline.data.download --fetch` to download and unpack it into {raw_dir.resolve()}.")
        return False
    print(f"[download] All required KuaiRand-Pure files present in {raw_dir}.")
    return True


def fetch_core_dataset(cfg: dict) -> None:
    """Downloads and unpacks the core KuaiRand-Pure dataset (~47MB tarball,
    no registration required) straight into raw_dir."""
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "KuaiRand-Pure.tar.gz"
        print(f"[download] Fetching {KUAIRAND_PURE_URL} ...")
        urllib.request.urlretrieve(KUAIRAND_PURE_URL, tar_path)
        print(f"[download] Unpacking into {raw_dir.resolve()} ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp)
        data_dir = Path(tmp) / "KuaiRand-Pure" / "data"
        for csv_path in data_dir.glob("*.csv"):
            shutil.copy(csv_path, raw_dir / csv_path.name)
    print(f"[download] Done. {sum(1 for _ in raw_dir.glob('*.csv'))} CSV files in {raw_dir.resolve()}.")


def fetch_supplements(cfg: dict) -> None:
    """Fetch the official Zenodo supplements (captions, category taxonomy).

    Gated behind config: only fetch if the organizer has confirmed these are
    in-scope for the challenge (see README "Open Questions").
    """
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    supplements = cfg["dataset"]["supplements"]
    for _key, filename in supplements.items():
        dest = raw_dir / filename
        if dest.exists():
            print(f"[download] {filename} already present, skipping.")
            continue
        url = f"{ZENODO_SUPPLEMENT_BASE}/{filename}"
        print(f"[download] Fetching {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:  # noqa: BLE001 — surfaced to user, not swallowed
            print(f"[download] Failed to fetch {filename}: {e}", file=sys.stderr)
            print(
                "This file may have moved, or the Zenodo record version changed. "
                "Verify at https://zenodo.org/records/18159199 and update ZENODO_SUPPLEMENT_BASE."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--supplements", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    if args.fetch:
        fetch_core_dataset(cfg)
    elif args.supplements:
        fetch_supplements(cfg)
    else:
        ok = check(cfg)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

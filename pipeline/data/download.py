"""
Fetches / verifies KuaiRand-Pure locally. This script does NOT fabricate a
direct file URL for the core dataset — KuaiRand is distributed from its
official project site (https://kuairand.com/) behind a request flow that
changes over time, so the safest thing this script can do is point you at
the right place and then verify what lands in `raw_dir`.

The Zenodo record for the *supplementary* files (video captions + video
category taxonomy) is stable and directly fetchable, since these were
published as a standalone Zenodo deposit: https://zenodo.org/records/18159199

Usage:
    python -m pipeline.data.download --check        # verify raw_dir has what we need
    python -m pipeline.data.download --supplements   # fetch the Zenodo caption/category files
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agent_config.yaml"

ZENODO_SUPPLEMENT_BASE = "https://zenodo.org/records/18159199/files"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def required_files(cfg: dict) -> list[str]:
    ds = cfg["dataset"]
    logs = ds["logs"]
    return [
        logs["standard_train"],
        logs["standard_evalset"],
        logs["random_unbiased"],
        ds["video_features_basic"],  # author_id — used by starter_kit/baseline.py's FM fields
    ]


def check(cfg: dict) -> bool:
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    missing = [f for f in required_files(cfg) if not (raw_dir / f).exists()]
    if missing:
        print(f"[download] Missing from {raw_dir}:")
        for m in missing:
            print(f"  - {m}")
        print(
            "\nKuaiRand-Pure is not auto-downloadable from a stable direct URL. "
            "Get it from the official project site: https://kuairand.com/\n"
            f"Place the CSVs listed above into: {raw_dir.resolve()}"
        )
        return False
    print(f"[download] All required KuaiRand-Pure logs present in {raw_dir}.")
    return True


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
    parser.add_argument("--supplements", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    if args.supplements:
        fetch_supplements(cfg)
    else:
        ok = check(cfg)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download Kaggle dataset to data/raw without committing the DB to git.

Dataset: datalearn/airlines-db (used by the referenced Kaggle notebook)
Expected output: data/raw/airlines_db.sqlite
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET = "datalearn/airlines-db"
DEFAULT_RAW_DIR = Path("data/raw")
MIN_DB_BYTES = 1_000_000


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"STDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n"
        )


def _ensure_kaggle_cli() -> None:
    if shutil.which("kaggle"):
        return
    raise FileNotFoundError(
        "Kaggle CLI not found. Install it first:\n"
        "  pip install kaggle\n\n"
        "Then place your API token at ~/.kaggle/kaggle.json (chmod 600)."
    )


def _ensure_kaggle_token() -> None:
    token_path = Path("~/.kaggle/kaggle.json").expanduser()
    if not token_path.is_file():
        raise FileNotFoundError(
            "Kaggle API token not found at ~/.kaggle/kaggle.json.\n"
            "Create a token from your Kaggle account and place it there (chmod 600)."
        )


def _find_sqlite_file(root: Path) -> Path:
    # Prefer *.sqlite, then *.db
    cands = list(root.rglob("*.sqlite")) + list(root.rglob("*.db"))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        raise FileNotFoundError(
            f"No .sqlite/.db file found after download under: {root.resolve()}"
        )
    large = [p for p in cands if p.stat().st_size >= MIN_DB_BYTES]
    if not large:
        sizes = ", ".join(
            f"{p.name} ({p.stat().st_size} bytes)" for p in sorted(cands)[:5]
        )
        raise RuntimeError(
            f"Found SQLite files but all are smaller than {MIN_DB_BYTES} bytes: {sizes}.\n"
            "Download may have failed; try re-running the script."
        )
    # Heuristic: prefer names containing 'air' or 'airline'
    def score(p: Path) -> tuple[int, int]:
        name = p.name.lower()
        s1 = int("airline" in name) * 2 + int("air" in name)
        s2 = p.stat().st_size  # larger is usually the real DB
        return (s1, s2)

    large.sort(key=score, reverse=True)
    return large[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle dataset into data/raw")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Kaggle dataset slug (default: env KAGGLE_DATASET or datalearn/airlines-db)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RAW_DIR),
        help="Output directory for raw data (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing files",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print manual download instructions and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset = args.dataset or os.getenv("KAGGLE_DATASET", DEFAULT_DATASET)
    raw_dir = Path(args.output)
    target_db = raw_dir / "airlines_db.sqlite"

    if args.manual:
        print("Manual download steps:")
        print("1) Visit: https://www.kaggle.com/datasets/datalearn/airlines-db")
        print("2) Download and extract the dataset")
        print(f"3) Place the SQLite file at: {target_db}")
        return 0

    raw_dir.mkdir(parents=True, exist_ok=True)

    if target_db.exists() and target_db.stat().st_size >= MIN_DB_BYTES and not args.force:
        print(f"[fetch_data] OK: {target_db} already exists. Use --force to re-download.")
        return 0

    _ensure_kaggle_cli()
    _ensure_kaggle_token()

    print(
        "[fetch_data] Source: Kaggle dataset datalearn/airlines-db "
        "(use subject to Kaggle terms; data is not redistributed)."
    )

    if args.force:
        slug = dataset.split("/")[-1]
        zip_path = raw_dir / f"{slug}.zip"
        if zip_path.exists():
            zip_path.unlink()
        if target_db.exists():
            target_db.unlink()

    # Download & unzip into raw_dir
    # --unzip: Kaggle CLI will unzip the downloaded archive in-place
    print(f"[fetch_data] Downloading Kaggle dataset: {dataset} -> {raw_dir}")
    _run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir), "--unzip"])

    # Find the sqlite file and normalize name
    found = _find_sqlite_file(raw_dir)

    if target_db.exists():
        # If it already exists and matches, do nothing; otherwise overwrite.
        if target_db.samefile(found):
            print(f"[fetch_data] OK: {target_db} already points to the downloaded DB.")
            return 0
        print(f"[fetch_data] Overwriting existing {target_db}")

    # Copy (not move) to preserve original filename for debugging
    shutil.copy2(found, target_db)
    print(f"[fetch_data] OK: wrote {target_db} (source: {found.name})")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[fetch_data] ERROR: {e}", file=sys.stderr)
        return_code = 1
        raise SystemExit(return_code)

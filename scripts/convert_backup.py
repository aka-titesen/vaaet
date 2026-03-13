#!/usr/bin/env python3
"""Manual utility to convert traffic_data.backup into traffic_data_raw.csv.

This script is not used by the active notebooks at runtime. It exists as a
one-time local helper for preparing a Colab-friendly CSV from a PostgreSQL
binary backup.

Run this script locally (where the correct PostgreSQL version is installed)
to produce a CSV that can be committed to the repo.  Once committed, the
data-preparation notebook will use the CSV via Tier 2, bypassing pg_restore
entirely on Google Colab.

Usage:
    python scripts/convert_backup.py

    # Custom paths:
    python scripts/convert_backup.py --backup path/to/file.backup --output path/to/out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.db import load_from_backup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a pg_dump backup to CSV for Colab compatibility.",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "traffic_data.backup",
        help="Path to the .backup file (default: data/raw/traffic_data.backup)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "traffic_data_raw.csv",
        help="Path to write the CSV (default: data/raw/traffic_data_raw.csv)",
    )
    args = parser.parse_args()

    if not args.backup.is_file():
        print(f"🔴 Backup file not found: {args.backup}")
        sys.exit(1)

    print(f"🔄 Converting {args.backup} → {args.output}")
    df = load_from_backup(args.backup, cache_csv=args.output)
    print(f"✅ Done — {len(df)} records written to {args.output}")
    print(f"   You can now commit {args.output.relative_to(_REPO_ROOT)} to the repo.")


if __name__ == "__main__":
    main()

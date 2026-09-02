# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
#!/usr/bin/env python3
"""Utilidad manual para convertir ``traffic_data.backup`` a CSV crudo.

Los notebooks activos no lo usan durante el runtime. Su propósito es preparar
una única vez, en un entorno local con cliente PostgreSQL 17, un ``RawCsvSource``
apto para Colab. El contenido crudo es sensible y permanece ignorado por Git.

Uso:
    python scripts/convert-postgres-backup.py

    # Paths personalizados:
    python scripts/convert-postgres-backup.py --backup path/to/file.backup --output path/to/out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vaaet_ml.data.database import load_from_backup

_REPO_ROOT = Path(__file__).resolve().parent.parent


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
    print(f"   Declare RawCsvSource({args.output.relative_to(_REPO_ROOT)!s}) in training.")


if __name__ == "__main__":
    main()

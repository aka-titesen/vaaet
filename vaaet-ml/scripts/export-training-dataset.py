#!/usr/bin/env python3
"""Build a checksum-protected vaaet-training-dataset-v1 package from CSV exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vaaet_ml.data.ingestion import create_dataset_package, load_dataset_package


def _read(path: Path | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and verify a portable VAAET training dataset package."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--validations", type=Path)
    parser.add_argument("--origin", default="manual-administrative-export")
    args = parser.parse_args()

    output = create_dataset_package(
        args.output,
        raw=_read(args.raw),
        features=_read(args.features),
        predictions=_read(args.predictions),
        validations=_read(args.validations),
        provenance={"origin": args.origin},
    )
    frames = load_dataset_package(output)
    summary = ", ".join(f"{name}={len(frame)}" for name, frame in frames.items())
    print(f"Created and verified {output}: {summary}")


if __name__ == "__main__":
    main()

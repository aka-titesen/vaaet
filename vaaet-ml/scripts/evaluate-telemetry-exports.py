# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluador offline manual de baseline y candidato sobre clips reales.

Los notebooks activos no importan esta utilidad. Se usa para revisar cambios de
robustez sobre CSV exportados y compara picos de velocidad, falsos negativos de
accidente y confusión entre ``Reduced`` y ``Congested``.

Cada archivo debe contener las columnas de telemetría requeridas por
``vaaet.features.engineering.engineer_features``. Las predicciones son
opcionales: si faltan, se derivan mediante las reglas vigentes.

El ground truth es opcional, pero se requiere para FN y confusión. La alineación
recomendada usa ``clip_id,minute_index``; sin esas claves se recurre al índice,
una alternativa menos robusta que el informe debe interpretar con cautela.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from vaaet.features.engineering import engineer_features
from vaaet.features.labeling import assign_traffic_state

from vaaet_ml.settings import STATE_LABELS

LABEL_TO_STATE = {name.lower(): code for code, name in STATE_LABELS.items()}
DEFAULT_JOIN_KEYS = ["clip_id", "minute_index"]


@dataclass(frozen=True)
class SpikeMetrics:
    count: int
    ratio: float
    mean_jump: float
    max_jump: float


@dataclass(frozen=True)
class RunMetrics:
    n_rows: int
    spikes: SpikeMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate before/after robustness over real clips: spikes, "
            "accident FN, reduced/congested confusion"
        ),
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline CSV")
    parser.add_argument("--candidate", required=True, help="Path to candidate CSV")
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Optional path to ground-truth CSV with true_state/true_label",
    )
    parser.add_argument(
        "--join-keys",
        nargs="+",
        default=DEFAULT_JOIN_KEYS,
        help="Join keys to align baseline/candidate/ground-truth",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=12.0,
        help="Absolute km/h jump threshold to count a spike (default: 12.0)",
    )
    parser.add_argument(
        "--clip-col",
        default="clip_id",
        help="Clip grouping column for per-clip spike detection (default: clip_id)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path to save full metrics JSON",
    )
    return parser.parse_args()


def _map_state_label(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = str(value).strip().lower()
    return LABEL_TO_STATE.get(key)


def ensure_predicted_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "traffic_state" in out.columns:
        out["traffic_state"] = pd.to_numeric(out["traffic_state"], errors="coerce")
        return out

    if "state_label" in out.columns:
        out["traffic_state"] = out["state_label"].map(_map_state_label)
        return out

    # El fallback reproduce las reglas sólo cuando el CSV no trae predicciones.
    feat = engineer_features(out)
    if feat.empty:
        out["traffic_state"] = np.nan
        return out

    states = assign_traffic_state(feat)
    out = out.iloc[-len(states) :].copy().reset_index(drop=True)
    out["traffic_state"] = states.to_numpy()
    return out


def ensure_true_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "true_state" in out.columns:
        out["true_state"] = pd.to_numeric(out["true_state"], errors="coerce")
        return out
    if "true_label" in out.columns:
        out["true_state"] = out["true_label"].map(_map_state_label)
        return out
    raise ValueError("Ground-truth file must contain true_state or true_label")


def _valid_join_keys(columns: list[str], requested: list[str]) -> list[str]:
    return [k for k in requested if k in columns]


def align_pair(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    join_keys: list[str],
) -> pd.DataFrame:
    """Alinea por claves comunes y recurre al índice sólo si no están disponibles."""

    keys = _valid_join_keys(
        list(set(baseline.columns).intersection(candidate.columns)),
        join_keys,
    )

    left = baseline.copy()
    right = candidate.copy()

    if keys:
        merged = left.merge(
            right,
            on=keys,
            how="inner",
            suffixes=("_baseline", "_candidate"),
        )
    else:
        n = min(len(left), len(right))
        left = left.iloc[:n].reset_index(drop=True)
        right = right.iloc[:n].reset_index(drop=True)
        merged = pd.concat(
            [left.add_suffix("_baseline"), right.add_suffix("_candidate")], axis=1
        )

    return merged


def align_ground_truth(
    merged: pd.DataFrame,
    ground_truth: pd.DataFrame,
    join_keys: list[str],
) -> pd.DataFrame:
    gt = ensure_true_states(ground_truth)

    keys = _valid_join_keys(
        list(set(merged.columns).intersection(gt.columns)), join_keys
    )
    if keys:
        return merged.merge(gt[keys + ["true_state"]], on=keys, how="inner")

    n = min(len(merged), len(gt))
    out = merged.iloc[:n].copy().reset_index(drop=True)
    out["true_state"] = gt.iloc[:n]["true_state"].to_numpy()
    return out


def compute_spike_metrics(
    df: pd.DataFrame,
    speed_col: str,
    clip_col: str,
    spike_threshold: float,
) -> SpikeMetrics:
    values = pd.to_numeric(df[speed_col], errors="coerce")
    work = df.copy()
    work[speed_col] = values

    if clip_col in work.columns:
        jumps = work.groupby(clip_col, dropna=False)[speed_col].diff().abs()
    else:
        jumps = work[speed_col].diff().abs()

    jumps = jumps.dropna()
    if jumps.empty:
        return SpikeMetrics(count=0, ratio=0.0, mean_jump=0.0, max_jump=0.0)

    spike_mask = jumps > spike_threshold
    spike_count = int(spike_mask.sum())
    return SpikeMetrics(
        count=spike_count,
        ratio=round(float(spike_count / len(jumps)), 4),
        mean_jump=round(float(jumps.mean()), 3),
        max_jump=round(float(jumps.max()), 3),
    )


def compute_run_metrics(
    merged: pd.DataFrame,
    speed_col: str,
    clip_col: str,
    spike_threshold: float,
) -> RunMetrics:
    spikes = compute_spike_metrics(
        merged,
        speed_col=speed_col,
        clip_col=clip_col,
        spike_threshold=spike_threshold,
    )
    return RunMetrics(n_rows=len(merged), spikes=spikes)


def accident_fn(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    true_accident = y_true == 3
    denom = int(true_accident.sum())
    if denom == 0:
        return {"fn": 0, "total_true_accident": 0, "fn_rate": 0.0}

    fn = int(((y_pred != 3) & true_accident).sum())
    return {
        "fn": fn,
        "total_true_accident": denom,
        "fn_rate": round(float(fn / denom), 4),
    }


def reduced_congested_confusion(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    mask = y_true.isin([1, 2])
    yt = y_true[mask]
    yp = y_pred[mask]

    return {
        "n_true_reduced_or_congested": int(mask.sum()),
        "reduced_to_congested": int(((yt == 1) & (yp == 2)).sum()),
        "congested_to_reduced": int(((yt == 2) & (yp == 1)).sum()),
        "correct_reduced": int(((yt == 1) & (yp == 1)).sum()),
        "correct_congested": int(((yt == 2) & (yp == 2)).sum()),
    }


def to_dict_metrics(run: RunMetrics) -> dict[str, object]:
    return {
        "n_rows": run.n_rows,
        "spikes": {
            "count": run.spikes.count,
            "ratio": run.spikes.ratio,
            "mean_jump": run.spikes.mean_jump,
            "max_jump": run.spikes.max_jump,
        },
    }


def main() -> None:
    args = parse_args()

    baseline = ensure_predicted_states(pd.read_csv(args.baseline))
    candidate = ensure_predicted_states(pd.read_csv(args.candidate))

    merged = align_pair(baseline, candidate, args.join_keys)
    merged = merged.dropna(subset=["traffic_state_baseline", "traffic_state_candidate"])

    baseline_metrics = compute_run_metrics(
        merged,
        speed_col="avg_speed_baseline",
        clip_col=f"{args.clip_col}_baseline",
        spike_threshold=args.spike_threshold,
    )
    candidate_metrics = compute_run_metrics(
        merged,
        speed_col="avg_speed_candidate",
        clip_col=f"{args.clip_col}_candidate",
        spike_threshold=args.spike_threshold,
    )

    report: dict[str, object] = {
        "n_aligned_rows": int(len(merged)),
        "spike_threshold": args.spike_threshold,
        "baseline": to_dict_metrics(baseline_metrics),
        "candidate": to_dict_metrics(candidate_metrics),
        "delta": {
            "spike_count": candidate_metrics.spikes.count
            - baseline_metrics.spikes.count,
            "spike_ratio": round(
                candidate_metrics.spikes.ratio - baseline_metrics.spikes.ratio,
                4,
            ),
            "mean_jump": round(
                candidate_metrics.spikes.mean_jump - baseline_metrics.spikes.mean_jump,
                3,
            ),
            "max_jump": round(
                candidate_metrics.spikes.max_jump - baseline_metrics.spikes.max_jump,
                3,
            ),
        },
    }

    if args.ground_truth:
        merged_gt = align_ground_truth(
            merged, pd.read_csv(args.ground_truth), args.join_keys
        )
        merged_gt = merged_gt.dropna(
            subset=["true_state", "traffic_state_baseline", "traffic_state_candidate"],
        )

        y_true = merged_gt["true_state"].astype(int)
        y_base = merged_gt["traffic_state_baseline"].astype(int)
        y_cand = merged_gt["traffic_state_candidate"].astype(int)

        report["ground_truth_rows"] = int(len(merged_gt))
        report["accident_fn"] = {
            "baseline": accident_fn(y_true, y_base),
            "candidate": accident_fn(y_true, y_cand),
        }
        report["reduced_congested_confusion"] = {
            "baseline": reduced_congested_confusion(y_true, y_base),
            "candidate": reduced_congested_confusion(y_true, y_cand),
        }

    print("\n=== Real Clip Robustness Evaluation ===")
    print(f"Aligned rows: {report['n_aligned_rows']}")
    print(f"Spike threshold (km/h jump): {args.spike_threshold}")

    b = report["baseline"]
    c = report["candidate"]
    d = report["delta"]
    print("\n[Spikes]")
    print(
        "Baseline  -> "
        f"count={b['spikes']['count']}, ratio={b['spikes']['ratio']}, "
        f"mean_jump={b['spikes']['mean_jump']}, max_jump={b['spikes']['max_jump']}"
    )
    print(
        "Candidate -> "
        f"count={c['spikes']['count']}, ratio={c['spikes']['ratio']}, "
        f"mean_jump={c['spikes']['mean_jump']}, max_jump={c['spikes']['max_jump']}"
    )
    print(
        "Delta     -> "
        f"count={d['spike_count']}, ratio={d['spike_ratio']}, "
        f"mean_jump={d['mean_jump']}, max_jump={d['max_jump']}"
    )

    if "accident_fn" in report:
        print("\n[Accident FN]")
        print(f"Baseline  -> {report['accident_fn']['baseline']}")
        print(f"Candidate -> {report['accident_fn']['candidate']}")

    if "reduced_congested_confusion" in report:
        print("\n[Reduced/Congested Confusion]")
        print(f"Baseline  -> {report['reduced_congested_confusion']['baseline']}")
        print(f"Candidate -> {report['reduced_congested_confusion']['candidate']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {out_path}")


if __name__ == "__main__":
    main()

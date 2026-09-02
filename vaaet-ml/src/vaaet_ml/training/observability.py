# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Informes inmutables y comparables para corridas supervisadas del laboratorio."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.settings import MODEL_STATE_LABELS

from vaaet_ml.data.artifact_serialization import (
    atomic_json_write,
    is_sha256,
    json_safe,
    sha256_bytes,
    stable_uuid,
    utc_now,
    valid_uuid,
)
from vaaet_ml.data.training_input_lock import TRAINING_INPUT_LOCK_CONTRACT, TrainingInputLock
from vaaet_ml.exceptions import TrainingStabilityError
from vaaet_ml.training.eligibility import CandidateEligibility
from vaaet_ml.training.execution import TrainingFitConfig, validate_training_history
from vaaet_ml.training.holdout_contract import HUMAN_HOLDOUT_CONTRACT
from vaaet_ml.training.lifecycle import TrainingMode

__all__ = [
    "CalibrationBin",
    "TrainingEvaluationEvidence",
    "TrainingReportComparison",
    "TrainingRunReport",
    "build_training_evaluation_evidence",
    "build_training_run_report",
    "compare_training_run_reports",
    "list_training_run_reports",
    "load_training_run_report",
    "write_training_run_report",
]

TRAINING_OBSERVABILITY_REPORT_CONTRACT = "vaaet-training-observability-report-v1"
TRAINING_OBSERVABILITY_REPORT_FILE = "training-observability-report.json"
TRAINING_OBSERVABILITY_SUMMARY_FILE = "training-summary.md"
_REPORT_SCHEMA_VERSION = 1
_RUNTIME_FIELDS = frozenset(
    {
        "declared_extras",
        "framework_gpu_available",
        "git_commit",
        "keras_version",
        "nvidia_smi",
        "python_version",
        "tensorflow_version",
        "total_ram_gib",
        "available_ram_gib",
        "content_free_gib",
    }
)
_COMPARABLE_METRICS = (
    "f1_macro",
    "expected_confusion_cost",
    "ece",
    "brier_score",
    "direct_normal_congested_error",
    "false_candidates_per_hour",
)
_SENSITIVE_FIELD_TOKENS = frozenset(
    {
        "credential",
        "dsn",
        "password",
        "private",
        "secret",
        "token",
        "url",
    }
)


@dataclass(frozen=True)
class CalibrationBin:
    """Agregado de confiabilidad sin conservar probabilidades individuales."""

    lower: float
    upper: float
    records: int
    confidence: float
    accuracy: float


@dataclass(frozen=True)
class TrainingEvaluationEvidence:
    """Evidencia agregada de test usada por reportes y diagnósticos."""

    direct_confusion: tuple[tuple[int, ...], ...]
    policy_confusion: tuple[tuple[int, ...], ...]
    reliability_bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True)
class TrainingRunReport:
    """Documento canónico de una corrida, independiente del bundle binario."""

    document: Mapping[str, object]
    path: Path | None = None

    @property
    def run_id(self) -> str:
        """Devuelve el UUID seguro de la corrida asociada al input lock."""

        return str(self.document["training_pipeline_run_id"])


@dataclass(frozen=True)
class TrainingReportComparison:
    """Deltas sólo descriptivos entre dos corridas con benchmark compatible."""

    comparable: bool
    reasons: tuple[str, ...]
    metric_deltas: Mapping[str, float]
    current_run_id: str
    reference_run_id: str


def build_training_evaluation_evidence(
    actual: Sequence[int],
    direct_predictions: Sequence[int],
    policy_predictions: Sequence[int],
    probabilities: np.ndarray,
) -> TrainingEvaluationEvidence:
    """Reduce el test a matrices y bins, sin guardar targets ni probabilidades crudas."""

    truth = np.asarray(actual, dtype=int)
    direct = np.asarray(direct_predictions, dtype=int)
    policy = np.asarray(policy_predictions, dtype=int)
    calibrated = np.asarray(probabilities, dtype=float)
    if truth.ndim != 1 or direct.shape != truth.shape or policy.shape != truth.shape:
        raise ValueError("Evaluation evidence requires aligned one-dimensional predictions.")
    if not np.isin(truth, (0, 1, 2)).all() or not np.isin(direct, (0, 1, 2)).all():
        raise ValueError("Evaluation evidence supports only the three stable model states.")
    if not np.isin(policy, (0, 1, 2)).all() or calibrated.shape != (len(truth), 3):
        raise ValueError("Evaluation evidence has an invalid policy prediction or probability shape.")
    if (
        not np.isfinite(calibrated).all()
        or (calibrated < 0).any()
        or not np.allclose(calibrated.sum(axis=1), 1.0, rtol=1e-5, atol=1e-8)
    ):
        raise ValueError("Evaluation evidence probabilities must be finite normalized values.")
    return TrainingEvaluationEvidence(
        direct_confusion=_matrix_tuple(confusion_matrix(truth, direct, labels=[0, 1, 2])),
        policy_confusion=_matrix_tuple(confusion_matrix(truth, policy, labels=[0, 1, 2])),
        reliability_bins=_reliability_bins(truth, calibrated),
    )


def build_training_run_report(
    *,
    training_input_lock: TrainingInputLock,
    training_lifecycle: Mapping[str, object],
    fit_config: TrainingFitConfig,
    supervision_report: Mapping[str, object],
    partition_rows: Mapping[str, int],
    selected_balance_strategy: str,
    balance_candidates: Sequence[Mapping[str, object]],
    training_history: object,
    direct_metrics: Mapping[str, float | None],
    policy_metrics: Mapping[str, float | None],
    incident_metrics: Mapping[str, float | int | None],
    support_table: pd.DataFrame,
    evaluation_evidence: TrainingEvaluationEvidence,
    eligibility: CandidateEligibility,
    decision_policy: Mapping[str, object],
    runtime: Mapping[str, object],
    model_version: str,
    cross_validation: Mapping[str, object] | None = None,
) -> TrainingRunReport:
    """Construye evidencia agregada que puede persistirse sin datos operativos."""

    lock = _validate_input_lock(training_input_lock)
    lifecycle = _validate_lifecycle(training_lifecycle)
    safe_model_version = _safe_identifier(model_version, "model_version")
    safe_balance_strategy = _safe_identifier(
        selected_balance_strategy,
        "selected_balance_strategy",
    )
    history = _history_document(validate_training_history(training_history))
    partitions = _partition_rows(partition_rows)
    support = _support_records(support_table)
    metrics = {
        "direct": _numeric_metrics(direct_metrics),
        "policy": _numeric_metrics(policy_metrics),
        "incident": _numeric_metrics(incident_metrics),
    }
    evidence = _evaluation_evidence_document(evaluation_evidence)
    runtime_evidence = _runtime_evidence(runtime)
    holdout = _holdout_descriptor(lock.document.get("human_holdout"))
    objectives = _observational_objectives(
        training_mode=lifecycle["training_mode"],
        supervision_report=supervision_report,
        human_holdout=holdout,
        eligibility=eligibility,
    )
    fingerprint_payload = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "training_pipeline_run_id": lock.document["training_pipeline_run_id"],
        "training_input_lock": lock.descriptor,
        "training_lifecycle": lifecycle,
        "model_version": safe_model_version,
        "model_output_mapping": {str(code): label for code, label in MODEL_STATE_LABELS.items()},
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fit_config": _fit_config_document(fit_config),
        "supervision": _safe_supervision_report(supervision_report),
        "partition_rows": partitions,
        "selected_balance_strategy": safe_balance_strategy,
        "balance_candidates": _candidate_records(balance_candidates),
        "history": history,
        "metrics": metrics,
        "support": support,
        "evaluation_evidence": evidence,
        "eligibility": _eligibility_document(eligibility),
        "decision_policy": _safe_aggregate_mapping(decision_policy, label="decision_policy"),
        "human_holdout": holdout,
        "runtime": runtime_evidence,
        "cross_validation": _optional_safe_aggregate_mapping(
            cross_validation,
            label="cross_validation",
        ),
        "objectives": objectives,
    }
    canonical = json.dumps(json_safe(fingerprint_payload), sort_keys=True, separators=(",", ":"))
    fingerprint = sha256_bytes(canonical.encode("utf-8"))
    document = {
        "contract": TRAINING_OBSERVABILITY_REPORT_CONTRACT,
        "schema_version": _REPORT_SCHEMA_VERSION,
        "report_id": stable_uuid("training-observability-report", lock.descriptor["lock_id"], fingerprint),
        "fingerprint": fingerprint,
        "created_at": utc_now().isoformat(),
        **fingerprint_payload,
    }
    return TrainingRunReport(document=MappingProxyType(document))


def write_training_run_report(
    output_root: str | Path,
    report: TrainingRunReport,
) -> TrainingRunReport:
    """Persiste el JSON y resumen sólo si la identidad de la corrida no cambia."""

    document = _validate_report_document(report.document)
    directory = Path(output_root) / str(document["training_pipeline_run_id"])
    report_path = directory / TRAINING_OBSERVABILITY_REPORT_FILE
    if report_path.is_file():
        existing = load_training_run_report(report_path)
        if existing.document["fingerprint"] != document["fingerprint"]:
            raise TrainingStabilityError("Training run already has a different observability report.")
        _write_summary_if_missing(directory, existing.document)
        return existing
    atomic_json_write(report_path, document)
    _write_summary_if_missing(directory, document)
    return TrainingRunReport(document=MappingProxyType(document), path=report_path.resolve())


def load_training_run_report(path: str | Path) -> TrainingRunReport:
    """Carga y valida un informe persistido sin leer modelos, datasets ni bundles."""

    report_path = Path(path)
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingStabilityError("Training observability report could not be read.") from exc
    if not isinstance(raw, dict):
        raise TrainingStabilityError("Training observability report must be a JSON object.")
    return TrainingRunReport(document=MappingProxyType(_validate_report_document(raw)), path=report_path.resolve())


def list_training_run_reports(
    output_root: str | Path,
    *,
    limit: int = 20,
) -> tuple[TrainingRunReport, ...]:
    """Lista de forma acotada informes válidos, sin crear un índice mutable."""

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")
    root = Path(output_root)
    paths = sorted(
        root.glob(f"*/{TRAINING_OBSERVABILITY_REPORT_FILE}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]
    return tuple(load_training_run_report(path) for path in paths)


def compare_training_run_reports(
    current: TrainingRunReport,
    reference: TrainingRunReport,
) -> TrainingReportComparison:
    """Compara sólo benchmarks idénticos; nunca recomienda una promoción automática."""

    current_document = _validate_report_document(current.document)
    reference_document = _validate_report_document(reference.document)
    reasons = _comparison_reasons(current_document, reference_document)
    if reasons:
        return TrainingReportComparison(
            comparable=False,
            reasons=tuple(reasons),
            metric_deltas=MappingProxyType({}),
            current_run_id=str(current_document["training_pipeline_run_id"]),
            reference_run_id=str(reference_document["training_pipeline_run_id"]),
        )
    current_metrics = _flat_metrics(current_document)
    reference_metrics = _flat_metrics(reference_document)
    deltas = {
        key: current_metrics[key] - reference_metrics[key]
        for key in _COMPARABLE_METRICS
        if key in current_metrics and key in reference_metrics
    }
    return TrainingReportComparison(
        comparable=True,
        reasons=(),
        metric_deltas=MappingProxyType(deltas),
        current_run_id=str(current_document["training_pipeline_run_id"]),
        reference_run_id=str(reference_document["training_pipeline_run_id"]),
    )


def _validate_input_lock(lock: TrainingInputLock) -> TrainingInputLock:
    descriptor = lock.descriptor
    if descriptor["contract"] != TRAINING_INPUT_LOCK_CONTRACT or not is_sha256(descriptor["fingerprint"]):
        raise ValueError("Training observability requires a valid training input lock.")
    if not valid_uuid(lock.document.get("training_pipeline_run_id")):
        raise ValueError("Training input lock has an invalid pipeline run identifier.")
    return lock


def _validate_lifecycle(value: Mapping[str, object]) -> dict[str, object]:
    required = {"training_mode", "supervision", "deployment_stage", "input_policy", "production_eligible"}
    if required - set(value):
        raise ValueError("training_lifecycle is incomplete.")
    try:
        mode = TrainingMode(str(value["training_mode"]))
    except ValueError as exc:
        raise ValueError("training_lifecycle has an unsupported training_mode.") from exc
    production_eligible = value["production_eligible"]
    if not isinstance(production_eligible, bool):
        raise ValueError("training_lifecycle production_eligible must be boolean.")
    return {
        "training_mode": mode.value,
        "supervision": _safe_identifier(value["supervision"], "supervision"),
        "deployment_stage": _safe_identifier(value["deployment_stage"], "deployment_stage"),
        "input_policy": _safe_identifier(value["input_policy"], "input_policy"),
        "production_eligible": production_eligible,
    }


def _partition_rows(value: Mapping[str, int]) -> dict[str, int]:
    if set(value) != {"train", "validation", "test"}:
        raise ValueError("partition_rows must contain train, validation, and test exactly once.")
    result = {name: int(rows) for name, rows in value.items()}
    if any(rows < 0 for rows in result.values()):
        raise ValueError("partition_rows cannot contain negative counts.")
    return result


def _support_records(table: pd.DataFrame) -> list[dict[str, object]]:
    required = {
        "traffic_state",
        "state_label",
        "support",
        "predicted",
        "precision",
        "precision_ci_95",
        "recall",
        "recall_ci_95",
    }
    if required - set(table):
        raise ValueError("support_table is missing required classification aggregates.")
    records = table.loc[:, sorted(required)].to_dict(orient="records")
    if len(records) != 3:
        raise ValueError("support_table must contain exactly the three stable states.")
    return [dict(record) for record in json_safe(records)]


def _numeric_metrics(value: Mapping[str, float | int | None]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for name, metric in value.items():
        safe_name = _safe_identifier(name, "metric")
        if metric is None:
            result[safe_name] = None
            continue
        if isinstance(metric, bool):
            raise ValueError(f"Metric {safe_name!r} must be numeric, not boolean.")
        numeric = float(metric)
        if not np.isfinite(numeric) or numeric < 0:
            raise ValueError(f"Metric {safe_name!r} must be finite and non-negative.")
        result[safe_name] = int(metric) if isinstance(metric, int) else numeric
    return result


def _evaluation_evidence_document(value: TrainingEvaluationEvidence) -> dict[str, object]:
    return {
        "direct_confusion": [list(row) for row in value.direct_confusion],
        "policy_confusion": [list(row) for row in value.policy_confusion],
        "reliability_bins": [
            {
                "lower": item.lower,
                "upper": item.upper,
                "records": item.records,
                "confidence": item.confidence,
                "accuracy": item.accuracy,
            }
            for item in value.reliability_bins
        ],
    }


def _runtime_evidence(value: Mapping[str, object]) -> dict[str, object]:
    """Reduce el runtime a evidencia agregada, portable y libre de paths."""

    unknown = set(value) - _RUNTIME_FIELDS
    if unknown:
        raise ValueError(f"Runtime evidence contains unsupported fields: {sorted(unknown)}")
    result: dict[str, object] = {}
    for name in sorted(value):
        item = value[name]
        if name == "declared_extras":
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
                raise ValueError("declared_extras must be a sequence of identifiers.")
            result[name] = [_safe_identifier(extra, "declared extra") for extra in item]
        elif name in {"framework_gpu_available"}:
            if not isinstance(item, bool):
                raise ValueError(f"Runtime evidence {name!r} must be boolean.")
            result[name] = item
        elif name in {"total_ram_gib", "available_ram_gib", "content_free_gib"}:
            if item is not None and (not isinstance(item, (float, int)) or float(item) < 0):
                raise ValueError(f"Runtime evidence {name!r} must be a non-negative number or null.")
            result[name] = None if item is None else float(item)
        else:
            result[name] = _safe_identifier(item, name)
    return result


def _safe_supervision_report(value: Mapping[str, object]) -> dict[str, object]:
    required = {
        "training_mode",
        "human_support",
        "human_support_targets",
        "human_support_progress",
        "human_support_deficit",
        "proxy_memory_weight",
        "effective_weight_by_class",
        "synthetic_multiplier",
        "synthetic_congested_effective_weight",
        "synthetic_congested_limit",
    }
    if required - set(value):
        raise ValueError("supervision_report is incomplete.")
    return _safe_aggregate_mapping(
        {name: value[name] for name in required},
        label="supervision_report",
    )


def _candidate_records(value: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    records = [_safe_aggregate_mapping(record, label="balance candidate") for record in value]
    if not records:
        raise ValueError("balance_candidates must not be empty.")
    return records


def _fit_config_document(config: TrainingFitConfig) -> dict[str, float | int]:
    return {
        "random_seed": config.random_seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "early_stopping_patience": config.early_stopping_patience,
        "reduce_learning_rate_patience": config.reduce_learning_rate_patience,
        "reduce_learning_rate_factor": config.reduce_learning_rate_factor,
        "minimum_learning_rate": config.minimum_learning_rate,
    }


def _history_document(history: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Convierte las series validadas a la forma JSON persistida del informe."""

    return {name: [float(value) for value in values] for name, values in history.items()}


def _eligibility_document(value: CandidateEligibility) -> dict[str, object]:
    return {
        "metric_gates": dict(value.metric_gates),
        "promotion_blockers": list(value.promotion_blockers),
        "human_holdout": value.human_holdout,
        "congested_minutes": value.congested_minutes,
        "congested_clips": value.congested_clips,
        "production_eligible": value.production_eligible,
    }


def _holdout_descriptor(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("human_holdout descriptor must be a mapping or null.")
    fingerprint = value.get("fingerprint")
    if not is_sha256(fingerprint):
        raise ValueError("human_holdout descriptor must contain a SHA-256 fingerprint.")
    contract = _safe_identifier(value.get("contract"), "human_holdout contract")
    snapshot_id = value.get("snapshot_id")
    generation = value.get("generation")
    if not valid_uuid(snapshot_id) or isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("human_holdout descriptor has an invalid snapshot or generation.")
    if contract != HUMAN_HOLDOUT_CONTRACT:
        raise ValueError("human_holdout descriptor has an unsupported contract.")
    return {
        "contract": contract,
        "snapshot_id": str(snapshot_id),
        "generation": generation,
        "fingerprint": str(fingerprint),
    }


def _observational_objectives(
    *,
    training_mode: object,
    supervision_report: Mapping[str, object],
    human_holdout: Mapping[str, object] | None,
    eligibility: CandidateEligibility,
) -> dict[str, object]:
    mode = TrainingMode(str(training_mode))
    support = supervision_report["human_support"]
    targets = supervision_report["human_support_targets"]
    proxy_weights = supervision_report["proxy_memory_weight"]
    if not all(isinstance(value, Mapping) for value in (support, targets, proxy_weights)):
        raise ValueError("supervision_report support objectives are malformed.")
    proxy_progress = {
        str(state): {
            "human_support": int(support.get(state, support.get(str(state), 0))),
            "target": int(targets.get(state, targets.get(str(state), 0))),
            "proxy_memory_weight": float(proxy_weights.get(state, proxy_weights.get(str(state), 0.0))),
            "status": _proxy_objective_status(mode, support, targets, state),
        }
        for state in (0, 1, 2)
    }
    gate_status = "met" if all(eligibility.metric_gates.values()) else "blocked"
    if mode is TrainingMode.SEED_BOOTSTRAP:
        gate_status = "not_applicable"
    holdout_status = "met" if human_holdout is not None and eligibility.human_holdout else "insufficient_evidence"
    incident_status = _incident_status(eligibility.promotion_blockers)
    return {
        "proxy_replacement": proxy_progress,
        "frozen_human_holdout": {"status": holdout_status},
        "candidate_quality": {"status": gate_status, "metric_gates": dict(eligibility.metric_gates)},
        "incident_safety": {"status": incident_status},
        "manual_decision_required": True,
    }


def _proxy_objective_status(
    mode: TrainingMode,
    support: Mapping[object, object],
    targets: Mapping[object, object],
    state: int,
) -> str:
    if mode is TrainingMode.SEED_BOOTSTRAP:
        return "not_applicable"
    current = int(support.get(state, support.get(str(state), 0)))
    target = int(targets.get(state, targets.get(str(state), 0)))
    if target <= 0:
        return "insufficient_evidence"
    return "met" if current >= target else "in_progress"


def _incident_status(blockers: Sequence[str]) -> str:
    if any("incident candidate rate" in blocker for blocker in blockers):
        return "blocked"
    if any("incident negative exposure" in blocker for blocker in blockers):
        return "insufficient_evidence"
    return "met"


def _comparison_reasons(current: Mapping[str, object], reference: Mapping[str, object]) -> list[str]:
    reasons: list[str] = []
    if current["feature_schema_version"] != reference["feature_schema_version"]:
        reasons.append("feature schema versions differ")
    if current["model_output_mapping"] != reference["model_output_mapping"]:
        reasons.append("model output mappings differ")
    current_holdout = current.get("human_holdout")
    reference_holdout = reference.get("human_holdout")
    if not isinstance(current_holdout, Mapping) or not isinstance(reference_holdout, Mapping):
        reasons.append("both reports require a frozen human holdout")
    elif current_holdout.get("fingerprint") != reference_holdout.get("fingerprint"):
        reasons.append("frozen human holdout fingerprints differ")
    return reasons


def _flat_metrics(document: Mapping[str, object]) -> dict[str, float]:
    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    result: dict[str, float] = {}
    for group in ("direct", "policy", "incident"):
        values = metrics.get(group)
        if isinstance(values, Mapping):
            for name, value in values.items():
                if isinstance(value, (float, int)) and np.isfinite(float(value)):
                    result[str(name)] = float(value)
    return result


def _validate_report_document(value: Mapping[str, object]) -> dict[str, object]:
    """Valida contrato, identidad y fingerprint antes de aceptar un informe."""

    document = dict(value)
    if document.get("contract") != TRAINING_OBSERVABILITY_REPORT_CONTRACT:
        raise TrainingStabilityError("Unsupported training observability report contract.")
    if document.get("schema_version") != _REPORT_SCHEMA_VERSION:
        raise TrainingStabilityError("Unsupported training observability report schema version.")
    if not valid_uuid(document.get("report_id")) or not valid_uuid(document.get("training_pipeline_run_id")):
        raise TrainingStabilityError("Training observability report has an invalid identifier.")
    if not is_sha256(document.get("fingerprint")):
        raise TrainingStabilityError("Training observability report has an invalid fingerprint.")
    lock = document.get("training_input_lock")
    if (
        not isinstance(lock, Mapping)
        or lock.get("contract") != TRAINING_INPUT_LOCK_CONTRACT
        or not valid_uuid(lock.get("lock_id"))
        or not is_sha256(lock.get("fingerprint"))
    ):
        raise TrainingStabilityError("Training observability report has an invalid input lock descriptor.")
    expected_report_id = stable_uuid("training-observability-report", lock["lock_id"], document["fingerprint"])
    if document["report_id"] != expected_report_id:
        raise TrainingStabilityError("Training observability report identifier does not match its contents.")
    expected = _report_fingerprint(document)
    if document["fingerprint"] != expected:
        raise TrainingStabilityError("Training observability report fingerprint does not match its contents.")
    return document


def _report_fingerprint(document: Mapping[str, object]) -> str:
    """Calcula la identidad sin campos creados durante la persistencia."""

    excluded = {"contract", "report_id", "fingerprint", "created_at"}
    payload = {key: value for key, value in document.items() if key not in excluded}
    canonical = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))
    return sha256_bytes(canonical.encode("utf-8"))


def _write_summary_if_missing(directory: Path, document: Mapping[str, object]) -> None:
    path = directory / TRAINING_OBSERVABILITY_SUMMARY_FILE
    summary = _render_summary(document)
    if path.is_file():
        if path.read_text(encoding="utf-8") != summary:
            raise TrainingStabilityError("Training run already has a different observability summary.")
        return
    _atomic_text_write(path, summary)


def _render_summary(document: Mapping[str, object]) -> str:
    lifecycle = document["training_lifecycle"]
    metrics = document["metrics"]
    eligibility = document["eligibility"]
    objectives = document["objectives"]
    lock = document["training_input_lock"]
    if not all(isinstance(value, Mapping) for value in (lifecycle, metrics, eligibility, objectives, lock)):
        raise TrainingStabilityError("Training observability report is malformed for summary rendering.")
    policy_metrics = metrics.get("policy", {})
    if not isinstance(policy_metrics, Mapping):
        policy_metrics = {}
    blockers = eligibility.get("promotion_blockers", [])
    if not isinstance(blockers, Sequence) or isinstance(blockers, (str, bytes)):
        blockers = []
    lines = [
        "# Informe de entrenamiento VAAET",
        "",
        f"- Corrida: `{document['training_pipeline_run_id']}`",
        f"- Input lock: `{lock['fingerprint']}`",
        f"- Modo: `{lifecycle['training_mode']}`",
        f"- Etapa: `{lifecycle['deployment_stage']}`",
        f"- Elegibilidad calculada: `{eligibility['production_eligible']}`; la decisión sigue siendo humana.",
        "",
        "## KPIs de calidad",
        "",
        f"- F1 macro con política: {_format_metric(policy_metrics.get('f1_macro'))}",
        f"- Coste de confusión: {_format_metric(policy_metrics.get('expected_confusion_cost'))}",
        f"- ECE: {_format_metric(policy_metrics.get('ece'))}",
        f"- Brier: {_format_metric(policy_metrics.get('brier_score'))}",
        "",
        "## Objetivos observacionales",
        "",
        f"- Holdout humano congelado: `{_objective_status(objectives, 'frozen_human_holdout')}`",
        f"- Calidad del candidato: `{_objective_status(objectives, 'candidate_quality')}`",
        f"- Seguridad de incidentes: `{_objective_status(objectives, 'incident_safety')}`",
        "",
        "## Bloqueos vigentes",
        "",
        *([f"- {item}" for item in blockers] or ["- Ninguno calculado; la promoción sigue siendo manual."]),
        "",
        "## Diagnósticos",
        "",
        "- [Curvas de optimización](diagnostics/optimization-curves.png)",
        "- [Calidad y confiabilidad de test](diagnostics/test-quality.png)",
        "- [Supervisión y gobernanza](diagnostics/supervision-governance.png)",
        "",
    ]
    return "\n".join(lines)


def _format_metric(value: object) -> str:
    return f"{float(value):.4f}" if isinstance(value, (float, int)) else "sin evidencia"


def _objective_status(objectives: Mapping[object, object], name: str) -> str:
    """Obtiene el estado visible sin asumir una forma no validada del JSON."""

    objective = objectives.get(name)
    return str(objective.get("status", "sin evidencia")) if isinstance(objective, Mapping) else "sin evidencia"


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _reliability_bins(truth: np.ndarray, probabilities: np.ndarray) -> tuple[CalibrationBin, ...]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    edges = np.linspace(0.0, 1.0, 11)
    bins: list[CalibrationBin] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            bins.append(
                CalibrationBin(
                    lower=float(lower),
                    upper=float(upper),
                    records=int(mask.sum()),
                    confidence=float(confidence[mask].mean()),
                    accuracy=float(correct[mask].mean()),
                )
            )
    return tuple(bins)


def _json_mapping(value: Mapping[str, object], *, label: str) -> dict[str, object]:
    safe = json_safe(value)
    if not isinstance(safe, dict):
        raise ValueError(f"{label} must be a JSON object.")
    try:
        json.dumps(safe, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not JSON serializable.") from exc
    return safe


def _safe_aggregate_mapping(value: Mapping[str, object], *, label: str) -> dict[str, object]:
    """Acepta sólo agregados portables y rechaza campos que podrían filtrar datos."""

    safe = _json_mapping(value, label=label)
    _reject_sensitive_values(safe, label=label)
    return safe


def _reject_sensitive_values(value: object, *, label: str) -> None:
    """Rechaza recursivamente campos o valores que podrían filtrar datos sensibles."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_name = str(key).lower()
            if any(token in key_name for token in _SENSITIVE_FIELD_TOKENS):
                raise ValueError(f"{label} contains a sensitive field name.")
            _reject_sensitive_values(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_values(item, label=label)
    elif isinstance(value, str) and any(token in value for token in ("/", "\\", "://", "\n", "\r")):
        raise ValueError(f"{label} contains a sensitive path or URL value.")


def _optional_safe_aggregate_mapping(
    value: Mapping[str, object] | None,
    *,
    label: str,
) -> dict[str, object] | None:
    return None if value is None else _safe_aggregate_mapping(value, label=label)


def _safe_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(token in value for token in ("/", "\\", "://", "\n", "\r")):
        raise ValueError(f"{label} must be a non-sensitive identifier.")
    if label == "git_commit" and not re.fullmatch(r"[0-9a-fA-F]{7,40}", value):
        raise ValueError("git_commit must be a hexadecimal revision.")
    return value

# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Diagnósticos persistibles para informes agregados de entrenamiento."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from vaaet.settings import STATE_LABELS

from vaaet_ml.training.observability import TrainingRunReport

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["save_training_run_diagnostics"]

_DIAGNOSTIC_FILENAMES = (
    "optimization-curves.png",
    "test-quality.png",
    "supervision-governance.png",
)
_STATE_COLORS = ("#0072B2", "#E69F00", "#D55E00")
_STATUS_COLORS = {
    "met": "#009E73",
    "in_progress": "#0072B2",
    "insufficient_evidence": "#E69F00",
    "not_applicable": "#999999",
    "blocked": "#D55E00",
}


def save_training_run_diagnostics(
    report: TrainingRunReport,
    output_root: str | Path,
) -> Mapping[str, Path]:
    """Guarda tres figuras idempotentes sin acceder a datos ni artefactos binarios."""

    document = report.document
    directory = Path(output_root) / report.run_id / "diagnostics"
    import matplotlib.pyplot as plt

    figures = {
        "optimization-curves": _optimization_figure(document),
        "test-quality": _test_quality_figure(document),
        "supervision-governance": _supervision_figure(document),
    }
    paths: dict[str, Path] = {}
    for name, filename in zip(figures, _DIAGNOSTIC_FILENAMES, strict=True):
        path = directory / filename
        _save_figure_once(figures[name], path)
        plt.close(figures[name])
        paths[name] = path.resolve()
    return paths


def _optimization_figure(document: Mapping[str, object]) -> Figure:
    """Dibuja curvas agregadas sin reabrir el dataset ni el modelo."""

    import matplotlib.pyplot as plt

    history = _mapping(document, "history")
    val_loss = _series(history, "val_loss")
    best_epoch = int(np.argmin(val_loss)) + 1
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(13, 4.8))
    epochs = np.arange(1, len(val_loss) + 1)
    loss_axis.plot(epochs, _series(history, "loss"), label="Entrenamiento", color="#0072B2")
    loss_axis.plot(epochs, val_loss, label="Validación", color="#D55E00")
    loss_axis.axvline(best_epoch, color="#333333", linestyle="--", linewidth=1, label="Mejor época")
    loss_axis.set(title="Pérdida de optimización", xlabel="Época", ylabel="Pérdida")
    loss_axis.legend()
    loss_axis.grid(True, alpha=0.25)
    accuracy_axis.plot(epochs, _series(history, "accuracy"), label="Entrenamiento", color="#0072B2")
    accuracy_axis.plot(epochs, _series(history, "val_accuracy"), label="Validación", color="#D55E00")
    accuracy_axis.set(title="Exactitud de optimización", xlabel="Época", ylabel="Exactitud")
    accuracy_axis.legend()
    accuracy_axis.grid(True, alpha=0.25)
    figure.suptitle(
        f"Diagnóstico de optimización — mejor pérdida de validación en época {best_epoch}.\n"
        "Hallazgo: la selección posterior sigue usando sólo validation; test permanece congelado.",
        fontsize=11,
    )
    figure.tight_layout()
    return figure


def _test_quality_figure(document: Mapping[str, object]) -> Figure:
    """Dibuja matrices y confiabilidad a partir del informe canónico."""

    import matplotlib.pyplot as plt

    evidence = _mapping(document, "evaluation_evidence")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    _plot_confusion(axes[0], evidence["direct_confusion"], "Salida directa")
    _plot_confusion(axes[1], evidence["policy_confusion"], "Después de política")
    reliability = evidence["reliability_bins"]
    if not isinstance(reliability, Sequence):
        raise ValueError("Training report reliability bins are malformed.")
    confidence = [float(item["confidence"]) for item in reliability if isinstance(item, Mapping)]
    accuracy = [float(item["accuracy"]) for item in reliability if isinstance(item, Mapping)]
    records = [int(item["records"]) for item in reliability if isinstance(item, Mapping)]
    axes[2].plot([0, 1], [0, 1], "--", color="#555555", label="Ideal")
    axes[2].plot(confidence, accuracy, marker="o", color="#0072B2", label="Calibrado")
    for x_value, y_value, support in zip(confidence, accuracy, records, strict=True):
        axes[2].annotate(str(support), (x_value, y_value), xytext=(4, 3), textcoords="offset points")
    axes[2].set(title="Confiabilidad de test", xlabel="Confianza media", ylabel="Exactitud observada")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    figure.suptitle(
        "Calidad y confiabilidad sobre test. Hallazgo: los números pequeños junto a cada punto indican soporte; no prueban causalidad.",
        fontsize=11,
    )
    figure.tight_layout()
    return figure


def _supervision_figure(document: Mapping[str, object]) -> Figure:
    """Muestra avance HITL y estados de gobierno sin decidir una promoción."""

    import matplotlib.pyplot as plt

    objectives = _mapping(document, "objectives")
    proxy = _mapping(objectives, "proxy_replacement")
    states = ("0", "1", "2")
    support = [int(_mapping(proxy, state)["human_support"]) for state in states]
    targets = [int(_mapping(proxy, state)["target"]) for state in states]
    proxy_weights = [float(_mapping(proxy, state)["proxy_memory_weight"]) for state in states]
    figure, (support_axis, governance_axis) = plt.subplots(1, 2, figsize=(13, 4.8))
    positions = np.arange(len(states))
    support_axis.bar(positions - 0.2, support, width=0.4, label="Soporte humano", color="#0072B2")
    support_axis.bar(positions + 0.2, targets, width=0.4, label="Objetivo", color="#BBBBBB")
    support_axis.set(
        title="Progreso HITL y memoria proxy",
        xticks=positions,
        xticklabels=[STATE_LABELS[int(state)] for state in states],
        ylabel="Registros humanos",
    )
    for position, weight in zip(positions, proxy_weights, strict=True):
        support_axis.text(position, max(support[position], targets[position]) + 1, f"proxy={weight:.2f}", ha="center", fontsize=9)
    support_axis.legend()
    support_axis.grid(axis="y", alpha=0.25)
    status_items = {
        "Holdout humano": _mapping(objectives, "frozen_human_holdout")["status"],
        "Calidad": _mapping(objectives, "candidate_quality")["status"],
        "Incidentes": _mapping(objectives, "incident_safety")["status"],
    }
    labels = list(status_items)
    colors = [_STATUS_COLORS.get(str(status), "#999999") for status in status_items.values()]
    governance_axis.barh(labels, [1] * len(labels), color=colors)
    for index, status in enumerate(status_items.values()):
        governance_axis.text(0.02, index, str(status), va="center", color="white", fontweight="bold")
    governance_axis.set(title="Gobernanza observacional", xlim=(0, 1), xticks=[])
    figure.suptitle(
        "Supervisión y gobernanza. Hallazgo: los estados informan evidencia disponible y nunca activan una promoción automática.",
        fontsize=11,
    )
    figure.tight_layout()
    return figure


def _plot_confusion(axis: Axes, matrix: object, title: str) -> None:
    values = np.asarray(matrix, dtype=int)
    if values.shape != (3, 3):
        raise ValueError("Training report confusion matrices must have shape (3, 3).")
    image = axis.imshow(values, cmap="Blues")
    labels = [STATE_LABELS[code] for code in (0, 1, 2)]
    axis.set(title=title, xlabel="Predicción", ylabel="Real", xticks=range(3), xticklabels=labels, yticks=range(3), yticklabels=labels)
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _save_figure_once(figure: Figure, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.png")
    try:
        figure.savefig(temporary, dpi=150, bbox_inches="tight")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    item = value.get(name)
    if not isinstance(item, Mapping):
        raise ValueError(f"Training report {name!r} is malformed.")
    return item


def _series(value: Mapping[str, object], name: str) -> np.ndarray:
    item = value.get(name)
    if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
        raise ValueError(f"Training report history {name!r} is malformed.")
    return np.asarray(item, dtype=float)

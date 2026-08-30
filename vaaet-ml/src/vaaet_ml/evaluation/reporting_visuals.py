# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptadores de gráficos opcionales para notebooks de entrenamiento e inferencia."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from vaaet.settings import STATE_LABELS


class _AxesLike(Protocol):
    """Superficie mínima de Matplotlib usada por los adaptadores de notebook."""

    def axis(self, *args: object, **kwargs: object) -> object: ...
    def axvline(self, *args: object, **kwargs: object) -> object: ...
    def bar(self, *args: object, **kwargs: object) -> object: ...
    def grid(self, *args: object, **kwargs: object) -> object: ...
    def hist(self, *args: object, **kwargs: object) -> object: ...
    def legend(self, *args: object, **kwargs: object) -> object: ...
    def plot(self, *args: object, **kwargs: object) -> object: ...
    def scatter(self, *args: object, **kwargs: object) -> object: ...
    def set(self, *args: object, **kwargs: object) -> object: ...
    def set_title(self, *args: object, **kwargs: object) -> object: ...
    def stackplot(self, *args: object, **kwargs: object) -> object: ...
    def text(self, *args: object, **kwargs: object) -> object: ...


def plot_training_history(history: Mapping[str, Sequence[float]]) -> None:
    """Renderiza curvas de pérdida y exactitud de entrenamiento y validación."""

    import matplotlib.pyplot as plt

    required = {"loss", "val_loss", "accuracy", "val_accuracy"}
    if missing := sorted(required - set(history)):
        raise ValueError(f"Training history is missing required series: {missing}")
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(14, 5))
    loss_axis.plot(history["loss"], label="Entrenamiento")
    loss_axis.plot(history["val_loss"], label="Validación")
    loss_axis.set(xlabel="Época", ylabel="Pérdida", title="Pérdida durante el entrenamiento")
    loss_axis.legend()
    loss_axis.grid(True, alpha=0.3)
    accuracy_axis.plot(history["accuracy"], label="Entrenamiento")
    accuracy_axis.plot(history["val_accuracy"], label="Validación")
    accuracy_axis.set(xlabel="Época", ylabel="Exactitud", title="Exactitud durante el entrenamiento")
    accuracy_axis.legend()
    accuracy_axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plt.show()


def plot_training_evaluation(
    y_true: Sequence[int],
    direct_predictions: Sequence[int],
    final_predictions: Sequence[int],
    probabilities: np.ndarray,
    *,
    state_labels: Mapping[int, str] = STATE_LABELS,
) -> None:
    """Renderiza matrices directas/finales y un diagrama de confiabilidad."""

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    truth = np.asarray(y_true, dtype=int)
    direct = np.asarray(direct_predictions, dtype=int)
    final = np.asarray(final_predictions, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    if truth.ndim != 1 or direct.shape != truth.shape or final.shape != truth.shape:
        raise ValueError("Predictions and targets must be one-dimensional and equally sized.")
    if proba.shape != (len(truth), 3):
        raise ValueError("probabilities must have shape (records, 3).")
    labels = [0, 1, 2]
    names = [state_labels[code] for code in labels]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, matrix, color, title in (
        (axes[0], confusion_matrix(truth, direct, labels=labels), "Greens", "Salida directa del MLP"),
        (axes[1], confusion_matrix(truth, final, labels=labels), "Blues", "Después de umbrales e histéresis"),
    ):
        sns.heatmap(matrix, annot=True, fmt="d", cmap=color, xticklabels=names, yticklabels=names, ax=axis)
        axis.set(xlabel="Predicción directa" if axis is axes[0] else "Estado final", ylabel="Estado esperado", title=title)
    figure.tight_layout()
    plt.show()
    _plot_reliability(truth, proba)


def _plot_reliability(truth: np.ndarray, probabilities: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    edges = np.linspace(0.0, 1.0, 11)
    pairs = [
        (float(confidence[mask].mean()), float(correct[mask].mean()))
        for lower, upper in zip(edges[:-1], edges[1:], strict=True)
        if (mask := (confidence > lower) & (confidence <= upper)).any()
    ]
    bin_confidence, bin_accuracy = zip(*pairs, strict=False) if pairs else ((), ())
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Calibración ideal")
    axis.plot(bin_confidence, bin_accuracy, marker="o", label="MLP calibrado")
    axis.set(xlabel="Confianza media", ylabel="Exactitud observada", title="Confiabilidad de la confianza")
    axis.legend()
    figure.tight_layout()
    plt.show()


def show_inference_dashboard(
    frame: pd.DataFrame,
    *,
    state_labels: Mapping[int, str] = STATE_LABELS,
) -> None:
    """Renderiza un tablero para minutos clasificados; no aplica reglas de negocio."""

    import matplotlib.pyplot as plt

    required = {"traffic_state", "avg_speed", "confidence"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Inference dashboard is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Inference dashboard requires at least one classified minute.")
    figure = plt.figure(figsize=(20, 10))
    state_colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c", 3: "#8e44ad"}
    minute_axis = range(len(frame))
    _plot_state_distribution(figure.add_subplot(2, 3, 1), frame, state_labels, state_colors)
    _plot_speed(figure.add_subplot(2, 3, 2), frame, minute_axis)
    _plot_confidence(figure.add_subplot(2, 3, 3), frame)
    _plot_counts(figure.add_subplot(2, 3, 4), frame, minute_axis)
    _plot_speed_volume(figure.add_subplot(2, 3, 5), frame, state_colors)
    _plot_summary(figure.add_subplot(2, 3, 6), frame)
    figure.tight_layout()
    plt.show()


def _plot_state_distribution(axis: _AxesLike, frame: pd.DataFrame, labels: Mapping[int, str], colors: Mapping[int, str]) -> None:
    distribution = frame["traffic_state"].value_counts().sort_index()
    axis.bar([labels[int(code)] for code in distribution.index], distribution.values, color=[colors.get(int(code), "#999999") for code in distribution.index])
    axis.set(title="Minutos por estado", ylabel="Minutos")


def _plot_speed(axis: _AxesLike, frame: pd.DataFrame, minute_axis: range) -> None:
    axis.plot(minute_axis, frame["avg_speed"], color="#3498db", linewidth=1.5)
    axis.set(title="Velocidad media a lo largo del clip", xlabel="Minuto clasificable", ylabel="Velocidad (km/h)")
    axis.grid(True, alpha=0.3)


def _plot_confidence(axis: _AxesLike, frame: pd.DataFrame) -> None:
    axis.hist(frame["confidence"], bins=20, color="#9b59b6", edgecolor="white")
    axis.axvline(0.8, color="#e74c3c", linestyle="--", linewidth=1)
    axis.set(title="Confianza del modelo", xlabel="Confianza", ylabel="Minutos")


def _plot_counts(axis: _AxesLike, frame: pd.DataFrame, minute_axis: range) -> None:
    colors = {"car": "#3498db", "truck": "#e67e22", "bus": "#e74c3c", "motorcycle": "#2ecc71", "bicycle": "#9b59b6"}
    columns = [column for column in ("count_car", "count_truck", "count_bus", "count_motorcycle", "count_bicycle") if column in frame.columns]
    if columns:
        counts = frame[columns].fillna(0)
        axis.stackplot(minute_axis, *[counts[column] for column in columns], labels=[column.removeprefix("count_") for column in columns], colors=[colors.get(column.removeprefix("count_"), "#999999") for column in columns], alpha=0.8)
        axis.legend(loc="upper left", fontsize=7)
    axis.set(title="Vehículos detectados por tipo", xlabel="Minuto", ylabel="Cantidad")


def _plot_speed_volume(axis: _AxesLike, frame: pd.DataFrame, colors: Mapping[int, str]) -> None:
    axis.scatter(frame.get("total_vehicles", pd.Series(0, index=frame.index)), frame["avg_speed"], c=[colors.get(int(code), "#999999") for code in frame["traffic_state"]], alpha=0.7, edgecolors="white", linewidth=0.5)
    axis.set(title="Velocidad frente al volumen", xlabel="Vehículos por minuto", ylabel="Velocidad media (km/h)")
    axis.grid(True, alpha=0.3)


def _plot_summary(axis: _AxesLike, frame: pd.DataFrame) -> None:
    axis.axis("off")
    summary = [f"Minutos clasificados: {len(frame)}", f"Velocidad media: {frame['avg_speed'].mean():.1f} km/h", f"Confianza media: {frame['confidence'].mean():.3f}", f"Minutos con confianza menor a 0.8: {int(frame['confidence'].lt(0.8).sum())}"]
    if "total_vehicles" in frame:
        summary.append(f"Vehículos contabilizados: {frame['total_vehicles'].sum():.0f}")
    axis.text(0.08, 0.5, "\n".join(summary), fontsize=11, va="center")
    axis.set_title("Resumen del clip")


__all__ = ["plot_training_evaluation", "plot_training_history", "show_inference_dashboard"]

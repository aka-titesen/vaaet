"""Keras model factories used by VAAET traffic-state training."""

from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input

from vaaet_ml.settings import FEATURE_COLS, N_MODEL_STATES

__all__ = ["build_traffic_state_mlp"]


def build_traffic_state_mlp(
    *,
    input_features: int = len(FEATURE_COLS),
    output_classes: int = N_MODEL_STATES,
) -> Sequential:
    """Build and compile the canonical three-state tabular MLP.

    The explicit contract checks prevent a notebook override from silently
    producing a model that cannot be served by the bundle v2 pipeline.
    """
    if input_features != len(FEATURE_COLS):
        raise ValueError(
            f"The traffic-state MLP requires {len(FEATURE_COLS)} features; "
            f"received {input_features}."
        )
    if output_classes != N_MODEL_STATES:
        raise ValueError(
            f"The traffic-state MLP requires {N_MODEL_STATES} outputs; "
            f"received {output_classes}."
        )

    model = Sequential(
        [
            Input(shape=(input_features,)),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(output_classes, activation="softmax"),
        ],
        name="traffic_state_classifier",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

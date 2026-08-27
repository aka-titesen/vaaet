"""Manifest-first loading for the VAAET traffic-state bundle v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from vaaet.artifacts import LABEL_MAPPING_FILE, MODEL_FILE, SCALER_FILE, validate_manifest
from vaaet.settings import FEATURE_COLS, STATE_LABELS


@dataclass(frozen=True)
class LoadedTrafficBundle:
    """A bundle that passed its portable contract before deserialization."""

    manifest: dict[str, object]
    model: object
    scaler: object
    label_mapping: dict[int, str]
    deployment_stage: str
    input_policy: str


def authorize_bundle(
    manifest: Mapping[str, object],
    *,
    allow_pilot: bool,
    allow_experimental: bool,
    persist_to_database: bool,
) -> tuple[str, str]:
    """Reject unauthorized deployment stages before loading model bytes."""

    lifecycle = manifest["training_lifecycle"]
    if not isinstance(lifecycle, Mapping):
        raise ValueError("The bundle training lifecycle is invalid.")
    stage = lifecycle.get("deployment_stage")
    input_policy = lifecycle.get("input_policy")
    if not isinstance(stage, str) or not isinstance(input_policy, str):
        raise ValueError("The bundle lifecycle lacks deployment stage or input policy.")
    if stage == "pilot" and not allow_pilot:
        raise RuntimeError("El bundle piloto no está autorizado. Usá ALLOW_PILOT_BUNDLE=True.")
    if stage == "candidate" and not allow_experimental:
        raise RuntimeError("El bundle candidato requiere autorización offline explícita.")
    if stage == "candidate" and persist_to_database:
        raise RuntimeError("Los bundles candidatos son sólo offline. Usá PERSIST_TO_DATABASE=False.")
    return stage, input_policy


def load_traffic_bundle(
    directory: Path,
    *,
    allow_pilot: bool,
    allow_experimental: bool,
    persist_to_database: bool,
) -> LoadedTrafficBundle:
    """Validate the v2 manifest, authorize lifecycle, then deserialize its files."""

    manifest = validate_manifest(directory)
    stage, input_policy = authorize_bundle(
        manifest,
        allow_pilot=allow_pilot,
        allow_experimental=allow_experimental,
        persist_to_database=persist_to_database,
    )
    import joblib
    import tensorflow as tf

    model = tf.keras.models.load_model(directory / MODEL_FILE)
    scaler = joblib.load(directory / SCALER_FILE)
    label_mapping = joblib.load(directory / LABEL_MAPPING_FILE)
    if dict(label_mapping) != dict(STATE_LABELS):
        raise RuntimeError("label_mapping.joblib no coincide con los cuatro estados públicos.")
    if int(model.output_shape[-1]) != 3:
        raise RuntimeError("El MLP del bundle debe exponer exactamente tres estados estables.")
    if int(getattr(scaler, "n_features_in_", -1)) != len(FEATURE_COLS):
        raise RuntimeError("El scaler del bundle no coincide con el contrato de 19 features.")
    return LoadedTrafficBundle(
        manifest=manifest,
        model=model,
        scaler=scaler,
        label_mapping=dict(label_mapping),
        deployment_stage=stage,
        input_policy=input_policy,
    )

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from vaaet_ml.settings import FEATURE_COLS, N_MODEL_STATES  # noqa: E402
from vaaet_ml.training.modeling import build_traffic_state_mlp  # noqa: E402


def test_build_traffic_state_mlp_preserves_canonical_contract() -> None:
    model = build_traffic_state_mlp()

    assert model.input_shape[-1] == len(FEATURE_COLS)
    assert model.output_shape[-1] == N_MODEL_STATES
    assert model.name == "traffic_state_classifier"


@pytest.mark.parametrize(
    ("input_features", "output_classes"),
    [(len(FEATURE_COLS) - 1, N_MODEL_STATES), (len(FEATURE_COLS), 4)],
)
def test_build_traffic_state_mlp_rejects_contract_drift(
    input_features: int, output_classes: int
) -> None:
    with pytest.raises(ValueError):
        build_traffic_state_mlp(
            input_features=input_features,
            output_classes=output_classes,
        )

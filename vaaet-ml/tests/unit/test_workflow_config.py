# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from vaaet.exceptions import RuntimeConfigurationError

from vaaet_ml.workflow_config import (
    EvaluationWorkflowConfig,
    InferenceWorkflowConfig,
    TrainingWorkflowConfig,
)


def test_inference_config_rejects_unknown_review_mode() -> None:
    with pytest.raises(RuntimeConfigurationError, match="review_mode"):
        InferenceWorkflowConfig(False, False, False, False, "random", False, False, False)  # type: ignore[arg-type]


def test_training_config_requires_reason_for_versioned_holdout() -> None:
    with pytest.raises(RuntimeConfigurationError, match="human_holdout_update_reason"):
        TrainingWorkflowConfig(
            "hitl_retraining", False, False, True, "create_new_version", None,
            "reuse_or_create", None,
        )


def test_evaluation_config_requires_exact_holdout() -> None:
    with pytest.raises(RuntimeConfigurationError, match="must not be current.json"):
        EvaluationWorkflowConfig(
            True, "champion", "challenger", "current.json", 1, False, "", "", False, "", ""
        )

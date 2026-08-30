# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Fachada compatible para los holdouts humanos inmutables de VAAET 4.x."""

from vaaet_ml.training.holdout_contract import (
    CURRENT_POINTER_FILE,
    HOLDOUT_RECORD_COLUMNS,
    HUMAN_HOLDOUT_CONTRACT,
    HumanHoldoutAction,
    HumanHoldoutConfig,
    HumanHoldoutSnapshot,
)
from vaaet_ml.training.holdout_resolution import (
    require_comparable_holdouts,
    resolve_human_holdout,
)
from vaaet_ml.training.holdout_storage import FileSystemHoldoutStore

__all__ = [
    "CURRENT_POINTER_FILE",
    "FileSystemHoldoutStore",
    "HOLDOUT_RECORD_COLUMNS",
    "HUMAN_HOLDOUT_CONTRACT",
    "HumanHoldoutAction",
    "HumanHoldoutConfig",
    "HumanHoldoutSnapshot",
    "require_comparable_holdouts",
    "resolve_human_holdout",
]

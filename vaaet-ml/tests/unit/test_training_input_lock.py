# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Caracterización del lock inmutable de inputs de entrenamiento."""

from __future__ import annotations

import uuid
from pathlib import Path

from vaaet_ml.data.training_input_lock import create_training_input_lock


def test_training_input_lock_is_reproducible_for_exact_inputs(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    kwargs = {
        "training_pipeline_run_id": run_id,
        "training_mode": "hitl-retraining",
        "seed_snapshot": {"fingerprint": "a" * 64},
        "hitl_catalog": {"revision": 2, "package_ids": [str(uuid.uuid4())]},
        "human_holdout": {"fingerprint": "b" * 64},
        "result_rows": {"train": 100, "validation": 20, "test": 20},
        "resolution": {"duplicates": 2, "corrections": 1},
    }
    first = create_training_input_lock(tmp_path, **kwargs)
    second = create_training_input_lock(tmp_path, **kwargs)

    assert first.descriptor == second.descriptor
    assert first.path == second.path
    assert first.path.is_file()

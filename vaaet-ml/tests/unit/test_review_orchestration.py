# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import logging

import pandas as pd
from pytest import LogCaptureFixture

from vaaet_ml.data.review import prepare_inference_review


def test_disabled_review_has_no_queue_or_persistence(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="vaaet_ml.data.review")
    session = prepare_inference_review(
        enabled=False,
        classified=pd.DataFrame(),
        inference_pipeline_run_id=None,
        reviewer_id=None,
        settings=None,
        mode="priority",
    )

    assert session.export_frame is None
    assert session.validations == []
    assert "Revisión humana desactivada" in caplog.text

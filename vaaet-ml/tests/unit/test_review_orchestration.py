from __future__ import annotations

import pandas as pd

from vaaet_ml.data.review import prepare_inference_review


def test_disabled_review_has_no_queue_or_persistence(capsys) -> None:
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
    assert "Revisión humana desactivada" in capsys.readouterr().out

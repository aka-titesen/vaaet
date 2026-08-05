"""PostgreSQL 17 integration checks for vaaet-db-v2."""

from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from vaaet.data.persistence import persist_classified_telemetry
from vaaet.data.review import HumanValidation, persist_human_validation

ADMIN_URL = os.getenv("VAAET_DATABASE_ADMIN_URL")
pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def engine():
    if not ADMIN_URL:
        pytest.skip("VAAET_DATABASE_ADMIN_URL is not configured")
    active = create_engine(ADMIN_URL)
    try:
        yield active
    finally:
        active.dispose()


def test_migrated_schemas_tables_and_views_exist(engine) -> None:
    expected = {
        "vaaet_raw.traffic_data",
        "vaaet_ml.telemetry_features",
        "vaaet_ml.traffic_predictions",
        "vaaet_feedback.human_validations",
        "vaaet_feedback.review_queue",
        "vaaet_feedback.effective_human_labels",
        "public.traffic_data",
        "public.telemetry_raw",
        "public.traffic_classifications",
    }
    with engine.connect() as connection:
        existing = {
            name
            for name in expected
            if connection.execute(text("SELECT to_regclass(:name)"), {"name": name}).scalar()
        }
    assert existing == expected


def test_legacy_hitl_and_buenos_aires_timestamps_are_migrated(engine) -> None:
    if os.getenv("VAAET_EXPECT_LEGACY_FIXTURE") != "1":
        pytest.skip("Legacy fixture is not expected in this database")
    with engine.connect() as connection:
        raw_time = connection.execute(
            text(
                "SELECT record_time FROM vaaet_raw.traffic_data "
                "WHERE clip_id='legacy-clip'"
            )
        ).scalar_one()
        validation = connection.execute(
            text(
                "SELECT validated_state, incident_context_reviewed, notes, reviewed_at "
                "FROM vaaet_feedback.human_validations "
                "WHERE reviewer_id='legacy-migration'"
            )
        ).one()
        automatic_state = connection.execute(
            text(
                "SELECT traffic_state FROM vaaet_ml.traffic_predictions "
                "WHERE model_version='mlp-v1.1'"
            )
        ).scalar_one()
    assert raw_time.hour == 11
    assert validation.validated_state == 3
    assert validation.incident_context_reviewed
    assert validation.notes
    assert validation.reviewed_at.hour == 11
    assert automatic_state == 2


def test_automatic_accident_is_rejected_by_database(engine) -> None:
    with engine.begin() as connection:
        feature_id = connection.execute(
            text(
                """
                INSERT INTO vaaet_ml.telemetry_features
                  (pipeline_run_id, clip_id, record_time, feature_schema_version)
                VALUES
                  ('00000000-0000-0000-0000-000000000001', 'constraint-test',
                   '2026-08-04T12:00:00Z', 'traffic-features-v2')
                ON CONFLICT (clip_id, record_time, feature_schema_version)
                DO UPDATE SET clip_id=EXCLUDED.clip_id RETURNING id
                """
            )
        ).scalar_one()
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO vaaet_ml.traffic_predictions
                      (telemetry_feature_id, pipeline_run_id, traffic_state, state_label,
                       confidence, model_version)
                    VALUES (:feature_id, '00000000-0000-0000-0000-000000000001',
                            3, 'Accident', 1, 'constraint-test')
                    """
                ),
                {"feature_id": feature_id},
            )


def test_group_roles_follow_least_privilege(engine) -> None:
    checks = {
        "vaaet_collection_role": {
            "vaaet_raw.traffic_data": ("SELECT", "INSERT"),
        },
        "vaaet_inference_role": {
            "vaaet_ml.telemetry_features": ("SELECT", "INSERT", "UPDATE"),
            "vaaet_ml.traffic_predictions": ("SELECT", "INSERT", "UPDATE"),
        },
        "vaaet_training_role": {
            "vaaet_raw.traffic_data": ("SELECT",),
            "vaaet_ml.traffic_predictions": ("SELECT",),
            "vaaet_feedback.human_validations": ("SELECT",),
        },
        "vaaet_reviewer_role": {
            "vaaet_ml.traffic_predictions": ("SELECT",),
            "vaaet_feedback.human_validations": ("SELECT", "INSERT"),
        },
    }
    with engine.connect() as connection:
        for role, tables in checks.items():
            for table, privileges in tables.items():
                for privilege in privileges:
                    assert connection.execute(
                        text("SELECT has_table_privilege(:role, :table, :privilege)"),
                        {"role": role, "table": table, "privilege": privilege},
                    ).scalar()
        assert not connection.execute(
            text(
                "SELECT has_table_privilege('vaaet_reviewer_role', "
                "'vaaet_feedback.human_validations', 'UPDATE')"
            )
        ).scalar()
        assert not connection.execute(
            text(
                "SELECT has_table_privilege('vaaet_training_role', "
                "'vaaet_raw.traffic_data', 'INSERT')"
            )
        ).scalar()


def test_reinference_preserves_append_only_human_validation(engine) -> None:
    frame = pd.DataFrame(
        [
            {
                "clip_id": "reinference-test",
                "record_time": "2026-08-04T13:00:00Z",
                "traffic_state": 2,
                "state_label": "Congested",
                "confidence": 0.81,
                "model_version": "mlp-v2.0-test",
            }
        ]
    )
    persist_classified_telemetry(frame, engine=engine, model_version="mlp-v2.0-test")
    with engine.connect() as connection:
        prediction_id = connection.execute(
            text(
                """
                SELECT p.id FROM vaaet_ml.traffic_predictions p
                JOIN vaaet_ml.telemetry_features f ON f.id=p.telemetry_feature_id
                WHERE f.clip_id='reinference-test' AND p.model_version='mlp-v2.0-test'
                """
            )
        ).scalar_one()
    first_validation_id = persist_human_validation(
        HumanValidation(prediction_id, 1, "integration-reviewer"), engine=engine
    )
    frame.loc[0, "confidence"] = 0.92
    persist_classified_telemetry(frame, engine=engine, model_version="mlp-v2.0-test")
    with engine.connect() as connection:
        count = connection.execute(
            text(
                "SELECT count(*) FROM vaaet_feedback.human_validations "
                "WHERE prediction_id=:prediction_id"
            ),
            {"prediction_id": prediction_id},
        ).scalar_one()
        effective = connection.execute(
            text(
                "SELECT traffic_state FROM vaaet_feedback.effective_human_labels "
                "WHERE prediction_id=:prediction_id"
            ),
            {"prediction_id": prediction_id},
        ).scalar_one()
    assert count == 1
    assert effective == 1
    persist_human_validation(
        HumanValidation(
            prediction_id,
            0,
            "integration-reviewer",
            notes="Correction after reviewing adjacent minutes.",
            supersedes_validation_id=first_validation_id,
        ),
        engine=engine,
    )
    with engine.connect() as connection:
        count, effective = connection.execute(
            text(
                "SELECT count(*), max(e.traffic_state) FROM vaaet_feedback.human_validations v "
                "JOIN vaaet_feedback.effective_human_labels e ON e.prediction_id=v.prediction_id "
                "WHERE v.prediction_id=:prediction_id GROUP BY e.traffic_state"
            ),
            {"prediction_id": prediction_id},
        ).one()
    assert count == 2
    assert effective == 0

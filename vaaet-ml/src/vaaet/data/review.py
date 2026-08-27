"""Append-only human review for stable traffic states and incidents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from vaaet.data.database import DatabaseSettings, get_engine
from vaaet.data.dataset_artifacts import finalize_review_session
from vaaet.data.ingestion import create_dataset_package
from vaaet.data.pipeline_runs import PipelineRunMetadata, PipelineWorkflow, pipeline_run
from vaaet.settings import STATE_LABELS

REVIEW_QUEUE_QUERY = """
SELECT prediction_id, pipeline_run_id, clip_id, record_time, traffic_state,
       state_label, confidence, model_version, probability_margin,
       decision_abstained, measurement_reliable, accident_rule_triggered,
       accident_alert_started, accident_evidence_score, latest_validation_id,
       current_validated_state, current_reviewer_id, current_reviewed_at
FROM vaaet_feedback.review_queue
WHERE (:pipeline_run_id IS NULL OR pipeline_run_id = CAST(:pipeline_run_id AS UUID))
ORDER BY record_time
"""

INSERT_VALIDATION_QUERY = """
INSERT INTO vaaet_feedback.human_validations (
    id, prediction_id, validated_state, reviewer_id, reviewed_at, notes,
    review_source, incident_context_reviewed, supersedes_validation_id,
    pipeline_run_id
) VALUES (
    :id, :prediction_id, :validated_state, :reviewer_id, CURRENT_TIMESTAMP, :notes,
    :review_source, :incident_context_reviewed, :supersedes_validation_id,
    CAST(:pipeline_run_id AS UUID)
)
"""


@dataclass(frozen=True)
class HumanValidation:
    prediction_id: int
    validated_state: int
    reviewer_id: str
    notes: str | None = None
    incident_context_reviewed: bool = False
    supersedes_validation_id: UUID | None = None
    validation_id: UUID | None = None
    review_source: str = "colab"

    def __post_init__(self) -> None:
        if self.validated_state not in STATE_LABELS:
            raise ValueError("validated_state must be one of the four public traffic states.")
        if not self.reviewer_id.strip():
            raise ValueError("A stable reviewer identifier is required.")
        if self.validated_state == 3:
            if not self.incident_context_reviewed:
                raise ValueError("Accident requires explicit temporal-context confirmation.")
            if not self.notes or not self.notes.strip():
                raise ValueError("Accident confirmation requires a non-empty review note.")


@dataclass
class InferenceReviewSession:
    """Mutable user decisions paired with the immutable frame they review."""

    export_frame: pd.DataFrame | None
    validations: list[HumanValidation]


def select_review_queue(frame: pd.DataFrame, *, mode: str = "priority") -> pd.DataFrame:
    if mode not in {"priority", "all"}:
        raise ValueError("Review mode must be 'priority' or 'all'.")
    if mode == "all" or frame.empty:
        return frame.copy().reset_index(drop=True)
    if "latest_validation_id" in frame:
        frame = frame.loc[frame["latest_validation_id"].isna()].copy()
    priority = pd.Series(False, index=frame.index)
    for column in ("accident_rule_triggered", "decision_abstained"):
        if column in frame:
            priority |= frame[column].fillna(False).astype(bool)
    if "probability_margin" in frame:
        priority |= pd.to_numeric(frame["probability_margin"], errors="coerce").fillna(0).lt(0.15)
    if "confidence" in frame:
        priority |= pd.to_numeric(frame["confidence"], errors="coerce").fillna(0).lt(0.75)
    if "traffic_state" in frame:
        groups = frame.get("clip_id", pd.Series("all", index=frame.index))
        priority |= frame.groupby(groups)["traffic_state"].diff().fillna(0).ne(0)
    return frame.loc[priority].reset_index(drop=True)


def prepare_inference_review(
    *,
    enabled: bool,
    classified: pd.DataFrame | None,
    inference_pipeline_run_id: str | None,
    reviewer_id: str | None,
    settings: DatabaseSettings | Mapping[str, str] | None,
    mode: str,
) -> InferenceReviewSession:
    """Build the explicit PostgreSQL or portable review flow without hidden writes."""

    session = InferenceReviewSession(export_frame=None, validations=[])
    if not enabled:
        print("ℹ️ Revisión humana desactivada. Para el próximo video usá ENABLE_HUMAN_REVIEW=True.")
        return session
    if reviewer_id is None:
        raise ValueError("A stable reviewer identifier is required for human review.")
    if settings is not None and inference_pipeline_run_id is not None:
        full_queue = load_review_queue(
            settings=settings, pipeline_run_id=inference_pipeline_run_id, mode="all"
        )
        review_queue = select_review_queue(full_queue, mode=mode)
        print(f"✅ Cola PostgreSQL: {len(review_queue)} de {len(full_queue)} filas seleccionadas ({mode}).")
        prediction_keys = full_queue[["clip_id", "record_time", "prediction_id"]].copy()
        prediction_keys["record_time"] = pd.to_datetime(prediction_keys["record_time"], utc=True)
        session.export_frame = classified.copy()
        session.export_frame["record_time"] = pd.to_datetime(
            session.export_frame["record_time"], utc=True
        )
        session.export_frame = session.export_frame.merge(
            prediction_keys, on=["clip_id", "record_time"], how="left", validate="one_to_one"
        )
        if session.export_frame["prediction_id"].isna().any():
            raise RuntimeError("La cola de revisión PostgreSQL no cubre todas las filas inferidas.")

        def persist_and_accumulate(decision: HumanValidation) -> None:
            persist_human_validation(decision, settings=settings)
            session.validations.append(decision)

        build_review_widget(review_queue, reviewer_id=reviewer_id, on_submit=persist_and_accumulate)
        return session
    if classified is None or classified.empty:
        print("ℹ️ Revisión HITL omitida: no hay minutos clasificables.")
        return session

    reason = (
        "la inferencia no se persistió"
        if inference_pipeline_run_id is None
        else "el perfil review no está disponible"
    )
    print(f"ℹ️ Revisión portable: {reason}.")
    print("   Las decisiones no modifican PostgreSQL y quedan en el paquete inmutable.")
    session.export_frame = classified.copy().reset_index(drop=True)
    session.export_frame["prediction_id"] = session.export_frame.index + 1
    review_queue = select_review_queue(session.export_frame, mode=mode)
    print(f"✅ Cola portable: {len(review_queue)} de {len(session.export_frame)} filas seleccionadas ({mode}).")
    build_review_widget(review_queue, reviewer_id=reviewer_id, on_submit=session.validations.append)
    return session


def load_review_queue(
    *,
    settings: DatabaseSettings | Mapping[str, str] | None = None,
    engine: Engine | None = None,
    pipeline_run_id: UUID | str | None = None,
    mode: str = "priority",
) -> pd.DataFrame:
    owns_engine = engine is None
    active_engine = engine or get_engine(settings)
    try:
        frame = pd.read_sql(
            text(REVIEW_QUEUE_QUERY),
            active_engine,
            params={"pipeline_run_id": str(pipeline_run_id) if pipeline_run_id else None},
        )
    finally:
        if owns_engine:
            active_engine.dispose()
    return select_review_queue(frame, mode=mode)


def persist_human_validation(
    decision: HumanValidation,
    *,
    settings: DatabaseSettings | Mapping[str, str] | None = None,
    engine: Engine | None = None,
    pipeline_run_id: UUID | str | None = None,
) -> UUID:
    owns_engine = engine is None
    active_engine = engine or get_engine(settings)
    if pipeline_run_id is None:
        try:
            metadata = PipelineRunMetadata(
                workflow=PipelineWorkflow.REVIEW,
                source_kind=decision.review_source,
                input_rows=1,
                telemetry_schema_version=None,
                feature_schema_version=None,
                model_version=None,
            )
            with pipeline_run(metadata, engine=active_engine) as run:
                validation_id = persist_human_validation(
                    decision,
                    engine=active_engine,
                    pipeline_run_id=run.id,
                )
                run.set_output_rows(1)
            return validation_id
        finally:
            if owns_engine:
                active_engine.dispose()

    validation_id = decision.validation_id or uuid4()
    payload = {
        "id": str(validation_id),
        "prediction_id": decision.prediction_id,
        "validated_state": decision.validated_state,
        "reviewer_id": decision.reviewer_id,
        "notes": decision.notes,
        "review_source": decision.review_source,
        "incident_context_reviewed": decision.incident_context_reviewed,
        "supersedes_validation_id": (
            str(decision.supersedes_validation_id) if decision.supersedes_validation_id else None
        ),
        "pipeline_run_id": str(pipeline_run_id),
    }
    try:
        with active_engine.begin() as connection:
            connection.execute(text(INSERT_VALIDATION_QUERY), payload)
    finally:
        if owns_engine:
            active_engine.dispose()
    return validation_id


def export_offline_review_package(
    output_path: str | Path,
    *,
    classified: pd.DataFrame,
    validations: pd.DataFrame | Sequence[HumanValidation],
) -> Path:
    features = classified.copy()
    if "id" not in features:
        features["id"] = range(1, len(features) + 1)
    prediction_ids = (
        features["prediction_id"].tolist()
        if "prediction_id" in features
        else list(range(1, len(features) + 1))
    )
    predictions = pd.DataFrame(
        {
            "id": prediction_ids,
            "telemetry_feature_id": features["id"],
            "model_version": features.get("model_version", "unknown"),
        }
    )
    features = features.drop(columns=["prediction_id"], errors="ignore")
    if isinstance(validations, pd.DataFrame):
        validation_frame = validations.copy()
    else:
        validation_frame = pd.DataFrame([asdict(decision) for decision in validations])
        validation_frame = validation_frame.rename(columns={"validation_id": "id"})
    if validation_frame.empty:
        raise ValueError("Complete at least one human validation before exporting feedback.")
    if "id" not in validation_frame:
        validation_frame["id"] = [str(uuid4()) for _ in range(len(validation_frame))]
    else:
        validation_frame["id"] = validation_frame["id"].map(
            lambda value: str(uuid4() if value is None or pd.isna(value) else value)
        )
    validation_frame["validated_state"] = validation_frame.pop("validated_state")
    exported_at = pd.Timestamp.now(tz="UTC")
    validation_frame["reviewed_at"] = [
        exported_at + pd.Timedelta(microseconds=index)
        for index in range(len(validation_frame))
    ]
    return create_dataset_package(
        output_path,
        features=features,
        predictions=predictions,
        validations=validation_frame,
        provenance={"origin": "inference-colab-human-review"},
    )


def build_review_widget(queue: pd.DataFrame, *, reviewer_id: str, on_submit):
    """Build an explicit Colab form; widgets remain an optional lazy import."""
    import ipywidgets as widgets
    from IPython.display import display

    if queue.empty:
        print("No pending rows match the selected review mode.")
        return None
    position = {"value": 0}
    heading = widgets.HTML()
    state = widgets.Dropdown(
        options=[(label, code) for code, label in STATE_LABELS.items()],
        description="Validated:",
    )
    notes = widgets.Textarea(description="Notes:")
    context = widgets.Checkbox(description="I reviewed temporal context")
    submit = widgets.Button(description="Save validation", button_style="success")
    skip = widgets.Button(description="Skip")
    output = widgets.Output()

    def render() -> None:
        row = queue.iloc[position["value"]]
        current_state = row.get("current_validated_state")
        state.value = int(
            row["traffic_state"] if current_state is None or pd.isna(current_state) else current_state
        )
        notes.value = ""
        context.value = False
        heading.value = (
            f"<b>{position['value'] + 1}/{len(queue)}</b> — {row.get('clip_id')} "
            f"{row.get('record_time')} — predicted {row.get('state_label')} "
            f"(confidence={float(row.get('confidence', 0)):.3f}, "
            f"incident={bool(row.get('accident_rule_triggered', False))})"
        )

    def advance() -> None:
        if position["value"] + 1 < len(queue):
            position["value"] += 1
            render()
        else:
            heading.value = "<b>Review queue completed.</b>"
            submit.disabled = True
            skip.disabled = True

    def submit_row(_button) -> None:
        row = queue.iloc[position["value"]]
        decision = HumanValidation(
            prediction_id=int(row["prediction_id"]),
            validated_state=int(state.value),
            reviewer_id=reviewer_id,
            notes=notes.value.strip() or None,
            incident_context_reviewed=bool(context.value),
            supersedes_validation_id=(
                UUID(str(row["latest_validation_id"]))
                if pd.notna(row.get("latest_validation_id"))
                else None
            ),
        )
        with output:
            on_submit(decision)
            print(f"Saved {STATE_LABELS[decision.validated_state]} for {row.get('record_time')}")
        advance()

    submit.on_click(submit_row)
    skip.on_click(lambda _button: advance())
    render()
    widget = widgets.VBox(
        [heading, state, notes, context, widgets.HBox([submit, skip]), output]
    )
    display(widget)
    return widget


__all__ = [
    "HumanValidation",
    "InferenceReviewSession",
    "build_review_widget",
    "export_offline_review_package",
    "finalize_review_session",
    "load_review_queue",
    "prepare_inference_review",
    "persist_human_validation",
    "select_review_queue",
]

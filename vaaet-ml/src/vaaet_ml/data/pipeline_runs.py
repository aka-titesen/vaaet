# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Redacted, provider-neutral lifecycle records for VAAET workflows."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine
from vaaet.artifacts import FEATURE_SCHEMA_VERSION
from vaaet.logging import get_logger
from vaaet.settings import MODEL_VERSION, TELEMETRY_SCHEMA_VERSION

from vaaet_ml import __version__

logger = get_logger(__name__)

_START_RUN_SQL = """
SELECT vaaet_ops.start_pipeline_run(
    CAST(:id AS UUID), :workflow, :application_version, :git_commit,
    :telemetry_schema_version, :feature_schema_version, :model_version,
    :source_kind, :clip_id, :input_rows
)
"""

_FINISH_RUN_SQL = """
SELECT vaaet_ops.finish_pipeline_run(
    CAST(:id AS UUID), :status, :output_rows, :error_category
)
"""


class PipelineWorkflow(str, Enum):
    COLLECTION = "collection"
    INFERENCE = "inference"
    TRAINING = "training"
    REVIEW = "review"


@dataclass(frozen=True)
class PipelineRunMetadata:
    workflow: PipelineWorkflow
    git_commit: str | None = None
    source_kind: str | None = None
    clip_id: str | None = None
    input_rows: int | None = None
    telemetry_schema_version: str | None = TELEMETRY_SCHEMA_VERSION
    feature_schema_version: str | None = FEATURE_SCHEMA_VERSION
    model_version: str | None = MODEL_VERSION

    def __post_init__(self) -> None:
        if self.input_rows is not None and self.input_rows < 0:
            raise ValueError("Pipeline input_rows cannot be negative.")
        if self.git_commit is not None and not re.fullmatch(
            r"[0-9a-fA-F]{7,40}", self.git_commit
        ):
            raise ValueError("git_commit must be a 7-40 character hexadecimal revision.")
        for value, label in (
            (self.git_commit, "git_commit"),
            (self.source_kind, "source_kind"),
            (self.clip_id, "clip_id"),
        ):
            if value is not None and any(token in value.lower() for token in ("password=", "://")):
                raise ValueError(f"{label} may not contain credentials or a connection URL.")
            if value is not None and label in {"source_kind", "clip_id"} and any(
                separator in value for separator in ("/", "\\")
            ):
                raise ValueError(f"{label} must be an identifier, not a filesystem path.")


@dataclass
class PipelineRunHandle:
    id: UUID
    metadata: PipelineRunMetadata
    output_rows: int | None = None

    def set_output_rows(self, rows: int) -> None:
        if rows < 0:
            raise ValueError("Pipeline output_rows cannot be negative.")
        self.output_rows = rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_payload(
    handle: PipelineRunHandle,
    *,
    status: str,
    started_at: str,
    error_category: str | None,
) -> dict[str, object]:
    metadata = asdict(handle.metadata)
    metadata["workflow"] = handle.metadata.workflow.value
    return {
        "id": str(handle.id),
        "status": status,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "application_version": __version__,
        "output_rows": handle.output_rows,
        "error_category": error_category,
        **metadata,
    }


def _write_local_manifest(directory: Path, payload: dict[str, object]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{payload['id']}.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def start_pipeline_run(
    metadata: PipelineRunMetadata,
    *,
    engine: Engine | None = None,
    local_manifest_directory: str | Path | None = None,
) -> tuple[PipelineRunHandle, str]:
    """Start a database-backed run or prepare its redacted local fallback."""
    handle = PipelineRunHandle(uuid4(), metadata)
    started_at = _utc_now()
    if engine is not None:
        payload = {
            "id": str(handle.id),
            "workflow": metadata.workflow.value,
            "application_version": __version__,
            "git_commit": metadata.git_commit,
            "telemetry_schema_version": metadata.telemetry_schema_version,
            "feature_schema_version": metadata.feature_schema_version,
            "model_version": metadata.model_version,
            "source_kind": metadata.source_kind,
            "clip_id": metadata.clip_id,
            "input_rows": metadata.input_rows,
        }
        with engine.begin() as connection:
            connection.execute(text(_START_RUN_SQL), payload).scalar_one()
    elif local_manifest_directory is None:
        raise ValueError("A local manifest directory is required when PostgreSQL is unavailable.")
    return handle, started_at


def finish_pipeline_run(
    handle: PipelineRunHandle,
    *,
    status: str,
    started_at: str,
    engine: Engine | None = None,
    local_manifest_directory: str | Path | None = None,
    error_category: str | None = None,
) -> Path | None:
    """Finish a run without storing exception messages or other arbitrary metadata."""
    if status not in {"succeeded", "failed"}:
        raise ValueError("Final pipeline status must be succeeded or failed.")
    if status == "succeeded":
        error_category = None
    if engine is not None:
        with engine.begin() as connection:
            connection.execute(
                text(_FINISH_RUN_SQL),
                {
                    "id": str(handle.id),
                    "status": status,
                    "output_rows": handle.output_rows,
                    "error_category": error_category,
                },
            )
        return None
    if local_manifest_directory is None:
        raise ValueError("A local manifest directory is required when PostgreSQL is unavailable.")
    return _write_local_manifest(
        Path(local_manifest_directory),
        _local_payload(
            handle,
            status=status,
            started_at=started_at,
            error_category=error_category,
        ),
    )


@contextmanager
def pipeline_run(
    metadata: PipelineRunMetadata,
    *,
    engine: Engine | None = None,
    local_manifest_directory: str | Path | None = None,
) -> Iterator[PipelineRunHandle]:
    """Record a complete workflow lifecycle and re-raise the original failure."""
    handle, started_at = start_pipeline_run(
        metadata,
        engine=engine,
        local_manifest_directory=local_manifest_directory,
    )
    try:
        yield handle
    except Exception as error:
        try:
            finish_pipeline_run(
                handle,
                status="failed",
                started_at=started_at,
                engine=engine,
                local_manifest_directory=local_manifest_directory,
                error_category=type(error).__name__,
            )
        except Exception as audit_error:  # never mask the workflow failure
            logger.warning(
                "Pipeline failure audit could not be finalized: %s",
                type(audit_error).__name__,
            )
        raise
    else:
        finish_pipeline_run(
            handle,
            status="succeeded",
            started_at=started_at,
            engine=engine,
            local_manifest_directory=local_manifest_directory,
        )


__all__ = [
    "PipelineRunHandle",
    "PipelineRunMetadata",
    "PipelineWorkflow",
    "finish_pipeline_run",
    "pipeline_run",
    "start_pipeline_run",
]

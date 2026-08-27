# SPDX-FileCopyrightText: 2026 VAAET Contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Contrato portable de telemetría cruda producido por la visión."""

BASE_RAW_TELEMETRY_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "record_time",
    "avg_speed",
    "count_car",
    "count_truck",
    "count_bus",
    "count_motorcycle",
    "count_bicycle",
    "total_vehicles",
)

TELEMETRY_QUALITY_COLUMNS: tuple[str, ...] = (
    "near_zero_motion_count",
    "stationary_confirmed_count",
    "rejected_speed_count",
    "recovered_track_count",
    "speed_sample_count",
    "speed_measurement_quality",
    "optical_flow_tracking_ratio",
)

TELEMETRY_METADATA_COLUMNS: tuple[str, ...] = ("telemetry_schema_version",)

CANONICAL_RAW_TELEMETRY_COLUMNS: tuple[str, ...] = (
    *BASE_RAW_TELEMETRY_COLUMNS,
    *TELEMETRY_QUALITY_COLUMNS,
    *TELEMETRY_METADATA_COLUMNS,
)

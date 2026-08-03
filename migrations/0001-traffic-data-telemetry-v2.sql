-- VAAET ML 4.0.0 — nullable, backward-compatible telemetry v2 fields.
-- Historical rows intentionally remain NULL and therefore keep schema v1 semantics.

BEGIN;

ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS near_zero_motion_count INTEGER;
ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS stationary_confirmed_count INTEGER;
ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS rejected_speed_count INTEGER;
ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS recovered_track_count INTEGER;
ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS speed_sample_count INTEGER;
ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS speed_measurement_quality NUMERIC(8,4);
ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS optical_flow_tracking_ratio NUMERIC(8,4);
ALTER TABLE traffic_data ADD COLUMN IF NOT EXISTS telemetry_schema_version TEXT;

ALTER TABLE IF EXISTS telemetry_raw ADD COLUMN IF NOT EXISTS optical_flow_tracking_ratio NUMERIC(8,4);
ALTER TABLE IF EXISTS telemetry_raw ADD COLUMN IF NOT EXISTS telemetry_schema_version TEXT;

ALTER TABLE IF EXISTS traffic_classifications ADD COLUMN IF NOT EXISTS probability_margin NUMERIC(8,4);
ALTER TABLE IF EXISTS traffic_classifications ADD COLUMN IF NOT EXISTS decision_abstained BOOLEAN DEFAULT FALSE;
ALTER TABLE IF EXISTS traffic_classifications ADD COLUMN IF NOT EXISTS measurement_reliable BOOLEAN;
ALTER TABLE IF EXISTS traffic_classifications ADD COLUMN IF NOT EXISTS accident_alert_started BOOLEAN DEFAULT FALSE;

COMMIT;

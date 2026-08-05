CREATE TABLE public.traffic_data (
  id SERIAL PRIMARY KEY,
  clip_id TEXT NOT NULL,
  record_time TIMESTAMP NOT NULL,
  avg_speed NUMERIC(8,2) NOT NULL,
  count_car INTEGER NOT NULL,
  count_truck INTEGER NOT NULL,
  count_bus INTEGER NOT NULL,
  count_motorcycle INTEGER NOT NULL,
  count_bicycle INTEGER NOT NULL,
  total_vehicles INTEGER NOT NULL,
  UNIQUE (clip_id, record_time)
);

CREATE TABLE public.telemetry_raw (
  id SERIAL PRIMARY KEY,
  source_record_id INTEGER REFERENCES public.traffic_data(id),
  clip_id TEXT NOT NULL,
  record_time TIMESTAMP NOT NULL,
  avg_speed NUMERIC(8,2), total_vehicles INTEGER,
  count_car INTEGER, count_truck INTEGER, count_bus INTEGER,
  count_motorcycle INTEGER, count_bicycle INTEGER,
  heavy_vehicle_ratio NUMERIC(8,4), delta_speed NUMERIC(8,2),
  delta_count INTEGER, transition_flag SMALLINT DEFAULT 0,
  speed_variance NUMERIC(8,4), cumulative_delta_speed NUMERIC(8,2),
  low_speed_persistence NUMERIC(8,4)
);

CREATE TABLE public.traffic_classifications (
  id SERIAL PRIMARY KEY,
  telemetry_id INTEGER NOT NULL REFERENCES public.telemetry_raw(id),
  classified_at TIMESTAMP NOT NULL,
  traffic_state SMALLINT NOT NULL CHECK (traffic_state BETWEEN 0 AND 3),
  state_label TEXT NOT NULL,
  confidence NUMERIC(8,4) NOT NULL,
  model_version TEXT NOT NULL,
  model_traffic_state SMALLINT,
  model_state_label TEXT,
  model_confidence NUMERIC(8,4),
  accident_gate_applied BOOLEAN DEFAULT FALSE,
  accident_evidence_score NUMERIC(8,4),
  is_human_validated BOOLEAN DEFAULT FALSE,
  human_override_state SMALLINT,
  validated_at TIMESTAMP,
  UNIQUE (telemetry_id, model_version)
);

INSERT INTO public.traffic_data
  (clip_id, record_time, avg_speed, count_car, count_truck, count_bus,
   count_motorcycle, count_bicycle, total_vehicles)
VALUES ('legacy-clip', '2025-04-21 08:00:00', 1.5, 4, 1, 0, 0, 0, 5);

INSERT INTO public.telemetry_raw
  (source_record_id, clip_id, record_time, avg_speed, total_vehicles,
   count_car, count_truck, count_bus, count_motorcycle, count_bicycle,
   heavy_vehicle_ratio, delta_speed, delta_count, transition_flag,
   speed_variance, cumulative_delta_speed, low_speed_persistence)
VALUES
  (1, 'legacy-clip', '2025-04-21 08:00:00', 1.5, 5,
   4, 1, 0, 0, 0, 0.2, -4, 2, 1, 2, -4, 1);

INSERT INTO public.traffic_classifications
  (telemetry_id, classified_at, traffic_state, state_label, confidence,
   model_version, model_traffic_state, model_state_label, model_confidence,
   accident_gate_applied, accident_evidence_score, is_human_validated,
   human_override_state, validated_at)
VALUES
  (1, '2025-04-21 08:01:00', 3, 'Accident', 0.9,
   'mlp-v1.1', 2, 'Congested', 0.7, TRUE, 0.9, TRUE, 3,
   '2025-04-21 08:02:00');

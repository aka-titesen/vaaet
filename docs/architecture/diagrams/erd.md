<!-- context: VAAET/docs/diagrams/erd.md — Entity-Relationship diagram.
Referenced by SAD.md, ADR-005, README.md. -->

# Database Schema (ERD)

Table `traffic_data` (Module 0 — legacy) with Module 1 extensions: `telemetry_raw` and `traffic_classifications`.

For the complete ERD with all 3 tables, see [erd-phase2.md](erd-phase2.md).

```mermaid
erDiagram
    TRAFFIC_DATA {
        serial id PK "Auto-increment"
        text clip_id "Processed video identifier"
        timestamp record_time "Timestamp of recorded minute"
        numeric avg_speed "Average speed km/h (5,2)"
        integer count_car "Car count"
        integer count_truck "Truck count"
        integer count_bus "Bus count"
        integer count_motorcycle "Motorcycle count"
        integer count_bicycle "Bicycle count"
        integer total_vehicles "Total vehicles"
    }

    TELEMETRY_RAW {
        serial id PK "Auto-increment"
        integer source_record_id FK "FK to traffic_data.id"
        timestamp record_time "Record timestamp"
        numeric heavy_vehicle_ratio "Heavy vehicle ratio"
        numeric delta_speed "Speed delta"
        integer delta_count "Volume delta"
        smallint transition_flag "Transition flag"
        numeric speed_variance "Speed variance"
        smallint hour_of_day "Hour of day"
        smallint weather_condition "Weather condition"
    }

    TRAFFIC_CLASSIFICATIONS {
        serial id PK "Auto-increment"
        integer telemetry_id FK "FK to telemetry_raw.id"
        timestamp classified_at "Classification timestamp"
        smallint traffic_state "State code 0-3"
        text state_label "State name"
        numeric confidence "Model confidence"
        text model_version "Model version"
        boolean is_human_validated "HITL validated"
    }

    TRAFFIC_DATA ||--o{ TELEMETRY_RAW : "source_record_id"
    TELEMETRY_RAW ||--o{ TRAFFIC_CLASSIFICATIONS : "telemetry_id"
```

## Constraints

- **Primary Key**: `id` (auto-increment)
- **Unique**: `(clip_id, record_time)` — prevents duplicates if the same video is re-processed
- **Not Null**: `clip_id`, `record_time`, `avg_speed`, all counts

## Write Pattern

- **Frequency**: One INSERT every 60 seconds of processed video
- **Connection**: Opens and closes per write (no connection pooling)
- **Failure**: If the connection fails, the system continues without persistence (silent degradation)

## `clip_id` Format

Derived from the video filename:
```
bridge_2026-03-06_08-00-00_to_09-00-00
```

<!-- context: VAAET/docs/diagrams/erd-phase2.md — Complete ERD with all 3 tables.
Referenced by DDS.md, ADR-008, DATA_LINEAGE.md. -->

# Database Schema — Module 0 + Module 1

Three tables with FK chain: `traffic_data` → `telemetry_raw` → `traffic_classifications`.

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
        numeric avg_speed "Average speed (5,2)"
        integer total_vehicles "Total vehicles"
        integer count_car "Car count"
        integer count_truck "Truck count"
        integer count_bus "Bus count"
        integer count_motorcycle "Motorcycle count"
        integer count_bicycle "Bicycle count"
        numeric heavy_vehicle_ratio "Heavy ratio (5,4)"
        numeric delta_speed "Speed delta (6,2)"
        integer delta_count "Volume delta"
        smallint transition_flag "Transition flag 0/1"
        numeric speed_variance "Speed variance (6,2)"
        smallint hour_of_day "Hour of day 0-23"
        smallint weather_condition "Weather condition 0/1"
    }

    TRAFFIC_CLASSIFICATIONS {
        serial id PK "Auto-increment"
        integer telemetry_id FK "FK to telemetry_raw.id"
        timestamp classified_at "Classification timestamp"
        smallint traffic_state "State code 0-3"
        text state_label "State name"
        numeric confidence "Model confidence (5,4)"
        text model_version "Model version"
        boolean is_human_validated "Validated by HITL operator"
        smallint human_override_state "Human-corrected state"
        timestamp validated_at "HITL validation timestamp"
    }

    TRAFFIC_DATA ||--o{ TELEMETRY_RAW : "source_record_id"
    TELEMETRY_RAW ||--o{ TRAFFIC_CLASSIFICATIONS : "telemetry_id"
```

## Constraints

### telemetry_raw
- **PK**: `id` (auto-increment)
- **FK**: `source_record_id` → `traffic_data(id)`
- **Unique**: `(source_record_id)` — one feature set per telemetry record

### traffic_classifications
- **PK**: `id` (auto-increment)
- **FK**: `telemetry_id` → `telemetry_raw(id)`
- **Unique**: `(telemetry_id, model_version)` — one classification per model per record

## HITL (Human-in-the-Loop) Fields

Designed for Module 2 feedback loop. Initially all records have:
- `is_human_validated = FALSE`
- `human_override_state = NULL`
- `validated_at = NULL`

When a SISE operator validates a classification, these fields will be updated.

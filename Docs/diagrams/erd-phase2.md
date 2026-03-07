<!-- context: VAAET/Docs/diagrams/erd-phase2.md — Diagrama ERD completo con las 3 tablas.
Referenciado por DDS.md §8, ADR-008, DATA_LINEAGE.md §6. -->

# Esquema de Base de Datos — Etapa 1 + Etapa 2

Tres tablas con FK chain: `traffic_data` → `telemetry_raw` → `traffic_classifications`.

```mermaid
erDiagram
    TRAFFIC_DATA {
        serial id PK "Auto-incremento"
        text clip_id "Identificador del video procesado"
        timestamp record_time "Timestamp del minuto registrado"
        numeric avg_speed "Velocidad promedio km/h (5,2)"
        integer count_car "Conteo de autos"
        integer count_truck "Conteo de camiones"
        integer count_bus "Conteo de buses"
        integer count_motorcycle "Conteo de motos"
        integer count_bicycle "Conteo de bicicletas"
        integer total_vehicles "Total de vehículos"
    }

    TELEMETRY_RAW {
        serial id PK "Auto-incremento"
        integer source_record_id FK "FK a traffic_data.id"
        timestamp record_time "Timestamp del registro"
        numeric avg_speed "Velocidad promedio (5,2)"
        integer total_vehicles "Total vehículos"
        integer count_car "Conteo de autos"
        integer count_truck "Conteo de camiones"
        integer count_bus "Conteo de buses"
        integer count_motorcycle "Conteo de motos"
        integer count_bicycle "Conteo de bicicletas"
        numeric heavy_vehicle_ratio "Ratio pesados (5,4)"
        numeric delta_speed "Cambio de velocidad (6,2)"
        integer delta_count "Cambio de volumen"
        smallint transition_flag "Flag de transicion 0/1"
        numeric speed_variance "Var velocidad (6,2)"
        smallint hour_of_day "Hora del dia 0-23"
        smallint weather_condition "Condicion meteorologica 0/1"
    }

    TRAFFIC_CLASSIFICATIONS {
        serial id PK "Auto-incremento"
        integer telemetry_id FK "FK a telemetry_raw.id"
        timestamp classified_at "Timestamp de clasificacion"
        smallint traffic_state "Codigo de estado 0-3"
        text state_label "Nombre del estado"
        numeric confidence "Confianza del modelo (5,4)"
        text model_version "Version del modelo"
        boolean is_human_validated "Validado por operador HITL"
        smallint human_override_state "Estado corregido por humano"
        timestamp validated_at "Timestamp de validacion HITL"
    }

    TRAFFIC_DATA ||--o{ TELEMETRY_RAW : "source_record_id"
    TELEMETRY_RAW ||--o{ TRAFFIC_CLASSIFICATIONS : "telemetry_id"
```

## Constraints

### telemetry_raw
- **PK**: `id` (auto-increment)
- **FK**: `source_record_id` → `traffic_data(id)`
- **Unique**: `(source_record_id)` — un feature set por registro de telemetría

### traffic_classifications
- **PK**: `id` (auto-increment)
- **FK**: `telemetry_id` → `telemetry_raw(id)`
- **Unique**: `(telemetry_id, model_version)` — una clasificación por modelo por registro

## Campos HITL (Human-in-the-Loop)

Diseñados para Fase 2 futura. En Fase 1 todos los registros tienen:
- `is_human_validated = FALSE`
- `human_override_state = NULL`
- `validated_at = NULL`

Cuando un operador SISE valide una clasificación, se actualizarán estos campos.

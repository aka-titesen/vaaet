<!-- context: VAAET/Docs/diagrams/erd.md — Diagrama entidad-relación de la BD.
Referenciado por DDS.md, ADR-005, README.md. -->

# Esquema de Base de Datos (ERD)

Tabla `traffic_data` (Etapa 1) con extensión Phase 2: `telemetry_raw` y `traffic_classifications`.

Para el ERD completo con las 3 tablas, ver [erd-phase2.md](erd-phase2.md).

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
        numeric heavy_vehicle_ratio "Ratio pesados"
        numeric delta_speed "Cambio de velocidad"
        integer delta_count "Cambio de volumen"
        smallint transition_flag "Flag de transicion"
        numeric speed_variance "Variabilidad velocidad"
        smallint hour_of_day "Hora del dia"
        smallint weather_condition "Condicion meteorologica"
    }

    TRAFFIC_CLASSIFICATIONS {
        serial id PK "Auto-incremento"
        integer telemetry_id FK "FK a telemetry_raw.id"
        timestamp classified_at "Timestamp clasificacion"
        smallint traffic_state "Codigo estado 0-3"
        text state_label "Nombre del estado"
        numeric confidence "Confianza del modelo"
        text model_version "Version del modelo"
        boolean is_human_validated "Validado HITL"
    }

    TRAFFIC_DATA ||--o{ TELEMETRY_RAW : "source_record_id"
    TELEMETRY_RAW ||--o{ TRAFFIC_CLASSIFICATIONS : "telemetry_id"
```

## Constraints

- **Primary Key**: `id` (auto-increment)
- **Unique**: `(clip_id, record_time)` — previene duplicados si se re-procesa el mismo video
- **Not Null**: `clip_id`, `record_time`, `avg_speed`, todos los conteos

## Patrón de Escritura

- **Frecuencia**: Un INSERT cada 60 segundos de video procesado
- **Conexión**: Abre y cierra por cada escritura (sin connection pooling)
- **Fallo**: Si la conexión falla, el sistema continúa sin persistencia (degradación silenciosa)

## Formato de `clip_id`

Derivado del nombre del archivo de video:
```
bridge_2026-03-06_08-00-00_to_09-00-00
```

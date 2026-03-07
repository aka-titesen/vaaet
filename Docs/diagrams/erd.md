<!-- context: VAAET/Docs/diagrams/erd.md — Diagrama entidad-relación de la BD.
Referenciado por DDS.md, ADR-005, README.md. -->

# Esquema de Base de Datos (ERD)

Tabla única `traffic_data` en PostgreSQL (AWS RDS).

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

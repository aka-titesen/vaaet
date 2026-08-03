# Directorios de datos

- `raw/`: backups y exportaciones originales; nunca versionar datos sensibles.
- `processed/`: datasets generados para entrenamiento.
- `sample/`: ejemplos pequeños, anónimos y no sensibles.

`traffic_data`/CSV v2 añade contadores de tracks únicos, calidad de velocidad,
flujo óptico y `telemetry_schema_version`. Las filas históricas v1 conservan
esos campos como `NULL`; no equivalen a calidad perfecta. La plantilla de
ground truth está en `sample/traffic-state-annotation-template.csv` y su uso se
describe en [el protocolo humano](../docs/ml/human-annotation-protocol.md).

Los datos operativos permanecen ignorados por Git. DVC se reserva para el bundle
de modelo aprobado, no para videos ni backups de bases de datos.

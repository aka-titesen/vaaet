# Utilidades manuales

Este directorio contiene herramientas auxiliares; no forma parte del runtime de
los notebooks.

- `convert-postgres-backup.py`: convierte un backup PostgreSQL a CSV para Colab.
- `export-training-dataset.py`: crea y verifica `vaaet-training-dataset-v1.zip` desde exports CSV administrativos.
- `evaluate-telemetry-exports.py`: compara resultados exportados de telemetría.
- `audit-postgres-database.py`: audita en modo read-only migración, permisos,
  comentarios, constraints, índices, tamaños, integridad y planes PostgreSQL.
- `setup-dvc.sh`: prepara DVC y describe cómo registrar el bundle completo.

Requieren instalación editable del proyecto. No se admiten scripts temporales,
parches ni modificaciones de `sys.path`.

Las migraciones y roles viven en `migrations/`; no son utilidades de notebook.

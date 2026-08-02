# Utilidades manuales

Este directorio contiene herramientas auxiliares; no forma parte del runtime de
los notebooks.

- `convert-postgres-backup.py`: convierte un backup PostgreSQL a CSV para Colab.
- `evaluate-telemetry-exports.py`: compara resultados exportados de telemetría.
- `setup-dvc.sh`: prepara DVC y describe cómo registrar el bundle completo.

Requieren instalación editable del proyecto. No se admiten scripts temporales,
parches ni modificaciones de `sys.path`.

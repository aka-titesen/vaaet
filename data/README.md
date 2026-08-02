# Directorios de datos

- `raw/`: backups y exportaciones originales; nunca versionar datos sensibles.
- `processed/`: datasets generados para entrenamiento.
- `sample/`: ejemplos pequeños, anónimos y no sensibles.

Los datos operativos permanecen ignorados por Git. DVC se reserva para el bundle
de modelo aprobado, no para videos ni backups de bases de datos.

# VAAET Core

Biblioteca portable de percepción, telemetría e inferencia para notebooks y
workers de API. No accede a DVC, Google Drive, PostgreSQL ni sistemas de tareas.

El paquete importable es `vaaet`; la validación del bundle v2 ocurre antes de
deserializarlo mediante `vaaet.artifacts.validate_manifest()`.

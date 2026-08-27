# VAAET Core

Biblioteca portable de percepción, telemetría e inferencia para notebooks y
workers de API. No accede a DVC, Google Drive, PostgreSQL ni sistemas de tareas.

El paquete importable es `vaaet`; la validación del bundle v2 ocurre antes de
deserializarlo mediante `vaaet.artifacts.validate_manifest()`.

## Instalación local

Desde la raíz del monorepo, creá y activá una `.venv` con Python 3.10–3.13 y
elegí sólo los extras necesarios:

```bash
python -m venv .venv
# Activá .venv con tu shell.
python -m pip install --upgrade pip
python -m pip install -e "./vaaet-core[vision,inference,dev]"
python -m pip check
```

`vision` incorpora YOLO para detección. El repositorio conserva licencia MIT,
pero una API o Web App que use ese extra no puede desplegarse hasta verificar
una licencia Ultralytics Enterprise fuera de Git. Consultá el
[registro de licencias de terceros](../docs/governance/third-party-licenses.md).

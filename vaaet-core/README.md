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

`vision` incorpora YOLO para detección. VAAET se distribuye bajo AGPL-3.0-only:
una demo web pública que use este extra debe publicar su código correspondiente
y cumplir el gate de activos; una aplicación privada o comercial requiere una
licencia Ultralytics Enterprise fuera de Git. Consultá el
[registro de licencias de terceros](../docs/governance/third-party-licenses.md).

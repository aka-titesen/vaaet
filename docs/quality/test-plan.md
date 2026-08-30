# Plan de pruebas — VAAET

## Contexto

VAAET es un monorepo con `vaaet-core==0.1.0` (import `vaaet`) y
`vaaet-ml==4.5.3` (import `vaaet_ml`). [ADR-0021](../architecture/decisions/0021-portable-core-and-ml-laboratory-boundary.md)
define los límites; las reglas para agentes están en
[AGENTS.md](../../AGENTS.md) y [llms.txt](../../llms.txt). Este plan cubre
validación automática y evidencia manual; no reemplaza contratos de datos,
bundles o licencias.

## Objetivos

- Preservar los contratos del core: visión, telemetría, 19 features, estados,
  bundle v2 e inferencia manifest-first.
- Mantener aislado el laboratorio: datos, entrenamiento, evaluación, Colab,
  PostgreSQL, DVC y utilidades bajo `vaaet_ml`.
- Comprobar que los cuatro notebooks sean orquestadores delgados y sintácticamente
  válidos.
- Detectar regresiones de estructura, documentación, licencias y límites de
  componentes antes de integrar cambios.

## Estrategia

| Nivel | Propiedad | Ubicación principal |
| --- | --- | --- |
| Unitario y contractual | Core portable, paquetes del pipeline, bundle e inferencia | `vaaet-core/tests/` |
| Unitario y de integración local | Datos, entrenamiento, evaluación, runtime y persistencia de laboratorio | `vaaet-ml/tests/` |
| PostgreSQL | Migraciones, roles, grants y contratos reales | `vaaet-ml/tests/integration/` con PostgreSQL 17 en CI |
| Repositorio | Imports separados, contexto de agentes, notebooks, enlaces, licencias y layout | `vaaet-ml/tests/repository/` |
| Sintaxis | Código Python y celdas de los cuatro notebooks | `compileall` y `ast.parse()` |

Los notebooks activos son adquisición, entrenamiento, inferencia y evaluación
Champion--Challenger. Sólo los tres primeros son workflows operacionales;
evaluación es read-only y no crea `pipeline_run` ni persiste datos.

## Ejecución local

Desde la raíz, instalar el core antes del laboratorio y elegir los extras del
workflow. No existe un paquete instalable raíz.

```bash
python -m pip install -e "./vaaet-core[vision,inference,dev]"
python -m pip install -e "./vaaet-ml[training,visualization,database,dev]"
python -m pip check
```

Luego, desde cada componente:

```bash
# vaaet-core/
ruff check src tests
pytest tests -v --tb=short
python -m compileall -q src tests

# vaaet-ml/
ruff check src tests scripts migrations
pytest tests -v --tb=short
python -m compileall -q src tests scripts migrations
```

También se deben compilar con `ast.parse()` las celdas de los cuatro notebooks,
resolver enlaces Markdown y ejecutar `git diff --check`.

## CI y evidencia manual

GitHub Actions ejecuta jobs separados de core, ML, integración del workspace,
PostgreSQL, enlaces y DVC. DVC se invoca desde la raíz y obtiene la CLI base y
plugins declarados de `vaaet-ml[dvc,dvc-gdrive,dvc-s3]`, sin autenticarse ni
transferir datos.

La evidencia local no sustituye las comprobaciones manuales en Colab: GPU,
Google Drive, DVC remoto, descarga/ejecución YOLO y PostgreSQL con Secrets.
Una futura demo web debe completar además los requisitos de
[ADR-0022](../architecture/decisions/0022-agpl-public-demo-path.md) y el
[checklist AGPL](../governance/agpl-demo-release-checklist.md).

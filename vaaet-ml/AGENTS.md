# AGENTS.md — Contexto de ejecución para agentes de IA

## Identificación

| Campo | Detalle |
|---|---|
| Proyecto | VAAET ML — Video Advanced Analysis of Traffic |
| Versión | 4.5.3 |
| Runtime objetivo | Python 3.10–3.13; Google Colab |
| Responsable | Facundo Nicolás González |
| Última revisión | 2026-08-22 |

## Mandato

Actuá como Senior ML Engineer con foco en visión artificial, MLOps y pipelines de datos. Conservá la compatibilidad con Colab Free, los contratos y las decisiones en `../docs/architecture/decisions/`.

VAAET tiene tres workflows de primer nivel:

```text
notebooks/data-collection/  adquisición opcional de telemetría y video anotado
notebooks/training/         preparación de 19 features y entrenamiento batch
notebooks/inference/        inferencia de video, estado y feedback
src/vaaet/                  lógica compartida instalable
tests/                      pruebas unitarias, contractuales y de repositorio
```

La Web App futura vive en `../vaaet-app/` y sólo consume una API que usa el bundle v2 definido por `../docs/ml/model-artifact-contract.md`. El MLP aprende tres estados estables; Accident es un estado público exclusivamente humano conforme a ADR-0014. ADR-0015 gobierna namespaces, ingestión y feedback; ADR-0016 gobierna hardening y registro de ejecuciones; ADR-0017 gobierna inicio semilla y reentrenamiento HITL; ADR-0018 gobierna holdouts humanos congelados; ADR-0019 gobierna datasets inmutables e input locks.

## Gobernanza

| Nivel | Acciones |
|---|---|
| Always | Leer ADRs aplicables; ejecutar Ruff, tests, compilación y enlaces; mantener notebooks como orquestadores. |
| Ask | Cambiar las 19 `FEATURE_COLS`, esquema PostgreSQL, dependencias, umbrales, MLP, remotes DVC o una nueva refactorización mayor. |
| Never | Commitear secretos; eliminar tests; versionar `.pt`, `.keras` o `.joblib` directamente con Git; hardcodear conexiones; romper Colab Free. |

## Capas y reglas

- `settings.py`: constantes, umbrales y rutas.
- `contracts.py` y `artifacts.py`: contratos de datos y bundle.
- `data/`: datasets, conexión y persistencia.
- `features/`: ingeniería, etiquetado y generación sintética de entrenamiento.
- `vision/`: detección, tracking, velocidad y análisis anotado común.
- `inference/`: clasificación tabular del tráfico.
- `evaluation/`: calibración y reporting.

La lógica de negocio vive en `src/vaaet/`. Los notebooks importan `vaaet.*` después de una única instalación: wheel local en el runtime efímero de Colab y modo editable en desarrollo local. Nunca modifican `sys.path`.

## Validación

1. `ruff check src tests scripts`
2. `pytest tests/ -v --tb=short`
3. `python -m compileall -q src tests scripts`
4. Compilar las celdas de los tres notebooks con `ast.parse()`.
5. Comprobar enlaces Markdown.
6. Ejecutar `git diff --check`.

GPU, Drive, PostgreSQL, descarga de YOLO y DVC remoto se validan manualmente en Colab.

No agregar, quitar ni reordenar las 19 `FEATURE_COLS`; no cambiar los cuatro estados públicos ni el esquema PostgreSQL sin autorización y un ADR. ADR-0012 gobierna el límite multi-repo, ADR-0013 el workflow de adquisición, ADR-0014 la arquitectura jerárquica, ADR-0015 los namespaces/HITL, ADR-0016 el hardening y linaje operacional, ADR-0017 los modos semilla/HITL, ADR-0018 el benchmark humano versionado y ADR-0019 los datasets inmutables.

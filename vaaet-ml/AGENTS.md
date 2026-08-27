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
src/vaaet_ml/               laboratorio instalable: datos, entrenamiento y evaluación
tests/                      pruebas unitarias, contractuales y de repositorio
```

La lógica operativa reutilizable vive en `../vaaet-core/src/vaaet/`; este
componente la consume como `vaaet`. La Web App futura sólo consumirá una API
que usará el core y validará el bundle v2. El MLP aprende tres estados estables;
Accident es un estado público exclusivamente humano conforme a ADR-0014.
ADR-0021 gobierna esta frontera junto con los ADRs de datos, HITL y holdouts.

## Gobernanza

| Nivel | Acciones |
|---|---|
| Always | Leer ADRs aplicables; ejecutar Ruff, tests, compilación y enlaces; mantener notebooks como orquestadores. |
| Ask | Cambiar las 19 `FEATURE_COLS`, esquema PostgreSQL, dependencias, umbrales, MLP, remotes DVC o una nueva refactorización mayor. |
| Never | Commitear secretos; eliminar tests; versionar `.pt`, `.keras` o `.joblib` directamente con Git; hardcodear conexiones; romper Colab Free. |

## Capas y reglas

- `../vaaet-core/src/vaaet/`: contratos, umbrales algorítmicos, percepción,
  features, política de estados e inferencia portable.
- `settings.py`: rutas de laboratorio, DVC/Drive, configuración de datos y DB.
- `data/`: datasets, conexión y persistencia.
- `features/`: ingeniería, etiquetado y generación sintética de entrenamiento.
- `evaluation/`: comparación, drift y reporting de laboratorio.

Los notebooks instalan primero `vaaet-core` y luego `vaaet-ml`: importan
`vaaet.*` para operaciones y `vaaet_ml.*` para laboratorio. Nunca modifican
`sys.path`. El core no puede importar este paquete ni PostgreSQL, DVC o Drive.

## Validación

1. Instalar `../vaaet-core` y luego este componente local.
2. `ruff check src tests scripts`
3. `pytest tests/ -v --tb=short`
4. `python -m compileall -q src tests scripts`
4. Compilar las celdas de los tres notebooks con `ast.parse()`.
5. Comprobar enlaces Markdown.
6. Ejecutar `git diff --check`.

GPU, Drive, PostgreSQL, descarga de YOLO y DVC remoto se validan manualmente en Colab.

No agregar, quitar ni reordenar las 19 `FEATURE_COLS`; no cambiar los cuatro estados públicos ni el esquema PostgreSQL sin autorización y un ADR. ADR-0021 gobierna el core/laboratorio; ADR-0013 el workflow de adquisición, ADR-0014 la arquitectura jerárquica, ADR-0015 los namespaces/HITL, ADR-0016 el hardening y linaje operacional, ADR-0017 los modos semilla/HITL, ADR-0018 el benchmark humano versionado y ADR-0019 los datasets inmutables.

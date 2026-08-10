# ADR-0018 — Holdout humano congelado y versionado

- Estado: aceptado
- Fecha: 2026-08-10
- Versión: VAAET ML 4.4.0
- Complementa: [ADR-0014](0014-hierarchical-traffic-state-and-incident-policy.md) y [ADR-0017](0017-seed-bootstrap-and-hitl-retraining.md)

## Contexto

Hasta VAAET 4.3.0 `HUMAN_HOLDOUT_FROZEN` sólo declaraba una intención de
promoción. Validation y test se volvían a seleccionar en cada reentrenamiento;
por lo tanto, activar el booleano no creaba un benchmark inmutable ni permitía
comparar candidatos bajo las mismas condiciones.

## Decisión

En `TrainingMode.HITL_RETRAINING`, un holdout congelado es un snapshot portable
`vaaet-human-holdout-v1` que contiene las 19 features y etiquetas humanas
efectivas de validation y test. Conserva clips completos, excluye Accident,
proxy y sintéticos, y se almacena en Google Drive mediante un filesystem
montado. `current.json` señala la generación activa.

La primera ejecución con `REUSE_OR_CREATE` crea el snapshot si existen al menos
tres grupos independientes por estado estable. Las siguientes reutilizan sus
filas exactas y excluyen todos sus grupos del entrenamiento. Un cambio en una
etiqueta o feature congelada exige una actualización explícita.

`CREATE_NEW_VERSION` nunca sobrescribe: conserva la versión anterior, refresca
las validaciones efectivas de grupos reservados e incorpora una fracción de
grupos nuevos. El motivo es obligatorio y la fotografía completa de la fuente
hace la operación idempotente. Las generaciones con fingerprints diferentes
son benchmarks distintos y no se comparan automáticamente.

El manifiesto del modelo incorpora el descriptor del snapshot. Un booleano
`human_holdout=true` sin contrato, UUID, generación, fingerprint y soportes deja
de ser evidencia suficiente para promoción.

## Consecuencias

- PostgreSQL continúa siendo la autoridad del feedback; el ZIP es una fotografía
  de evaluación, no una copia de la base ni memoria de pesos.
- Google Drive es obligatorio en Colab cuando el holdout congelado está activo;
  no existe fallback aleatorio o efímero.
- Ningún grupo reservado vuelve silenciosamente a train.
- Corregir o ampliar el benchmark crea una nueva generación auditable.
- El contrato del bundle permanece en v2 y acepta el descriptor adicional;
  bundles sin holdout congelado continúan usando `human_holdout=false`.

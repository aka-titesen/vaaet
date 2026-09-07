# ADR-0026 — Continuidad temporal y revisiones inmutables del modelo

- Estado: aceptada
- Fecha: 2026-09-05
- Decisores: Facundo Nicolás González
- Actualiza: ADR-0014, ADR-0018 y ADR-0025
- Complementa: ADR-0021, ADR-0023 y ADR-0024

## Contexto

Las features temporales, la histéresis y el detector de posibles incidentes
conservaban memoria por `clip_id`. Ese límite no representa necesariamente una
secuencia física continua: un clip puede cambiar de cámara o contener huecos.
Además, `model_version` era una etiqueta semántica reutilizable y, por sí sola,
no identificaba los pesos exactos que originaron una predicción. Ambas
ambigüedades podían contaminar estados o reemplazar historia operacional.

Las métricas también necesitaban distinguir la salida inmediata del MLP del
estado final estabilizado y tratar clips completos, no minutos correlacionados,
como unidades estadísticas.

## Decisión

Se publican `traffic-telemetry-v3`, `traffic-features-v3`, bundle v3,
`mlp-v3.0` y `vaaet-db-v3`. Las 19 features mantienen sus nombres y orden.

`continuity_id` identifica un tramo continuo de una vista. La continuidad se
reinicia ante un nuevo clip, una transición de `VideoViewPlan` o un hueco mayor
a 90 segundos. Ingeniería de features, weak labels, histéresis y detección de
incidentes consumen la misma clave `clip_id + continuity_id`. Timestamps
duplicados, regresivos o inválidos se rechazan.

`model_version` permanece como nombre semántico. `model_revision` es un SHA-256
derivado de los hashes de modelo, scaler y mapping, la política de decisión, el
schema de features y el training input lock. PostgreSQL conserva predicciones
por `(telemetry_feature_id, model_revision)` y las features por ejecución. Los
conflictos idempotentes con contenido diferente fallan; nunca se actualiza una
predicción histórica revisada.

La evaluación publica métricas `direct_*` para la salida del MLP y `final_*`
para la salida calibrada y estabilizada. Los intervalos del 95 % se calculan
con bootstrap de clips completos. Los gates de promoción se aplican sobre sus
límites conservadores, no sólo sobre estimaciones puntuales.

Los holdouts v2 incorporan `continuity_id`. Un bundle v3 sólo se compara para
promoción con el mismo fingerprint de holdout v2. Los formatos históricos se
mantienen para evaluación legacy, sin habilitar promoción v3.

## Consecuencias

- Un estado o candidato de incidente no atraviesa silenciosamente cambios de
  vista ni huecos temporales.
- Dos entrenamientos con el mismo nombre de modelo conservan identidad e
  historia independientes.
- HUD progresivo e inferencia batch deben coincidir antes de persistir.
- Las métricas son más conservadoras y pueden quedar como insuficientes cuando
  pocos clips representan una clase.
- Los bundles y snapshots v2 siguen siendo históricos; no se reescriben.
- Alembic `20260905_0003` es irreversible de forma intencional. El rollback
  operativo requiere restaurar un backup anterior.

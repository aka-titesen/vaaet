# ADR-0015 — PostgreSQL modular, mínimo privilegio e ingestión HITL

- Estado: aceptado
- Fecha: 2026-08-04
- Versión: VAAET ML 4.1.0
- Sustituye la parte vigente de [ADR-0005](0005-postgresql-aws-rds.md) sobre conexión, credenciales y schema.

## Contexto

Los tres workflows necesitan compartir datos sin compartir una identidad con
permisos excesivos. La estructura histórica en `public` mezclaba telemetría,
features, predicciones y correcciones humanas; además, una reinferencia podía
modificar el registro que representaba feedback. El entrenamiento también
utilizaba fallback implícito y podía confundir datos raw con datos enriquecidos.

## Decisión

Una base PostgreSQL 14+ se divide en:

- `vaaet_raw.traffic_data`: telemetría v2 adquirida.
- `vaaet_ml.telemetry_features`: las 19 features contractuales.
- `vaaet_ml.traffic_predictions`: salidas automáticas limitadas a estados 0–2.
- `vaaet_feedback.human_validations`: decisiones humanas append-only 0–3.

Las vistas `review_queue` y `effective_human_labels` exponen respectivamente las
predicciones con contexto de revisión y la última validación. `priority` toma sólo
pendientes; `all` permite agregar correcciones. `Accident` requiere contexto temporal revisado
y nota. Las vistas de compatibilidad en `public` duran sólo durante VAAET 4.x.

Cuatro perfiles (`collection`, `inference`, `training`, `review`) cargan
credenciales independientes mediante Colab Secrets o entorno local. Se usa
`sqlalchemy.URL.create()`, TLS, conexiones pequeñas y nombres cualificados. El
administrador sólo ejecuta Alembic y grants fuera de los notebooks.

El entrenamiento usa `TrainingIngestionPlan` y fuentes tipadas. Puede combinar
raw y feedback desde PostgreSQL, backup o paquete CSV contractual; raw pasa por
feature engineering, mientras las 19 features validadas no se recalculan.
Predicciones sin validación jamás son etiquetas y Accident confirmado queda fuera
del target del MLP.

## Consecuencias

- La migración a `vaaet-db-v2` es administrativa e intencionalmente irreversible;
  un rollback exige restaurar el backup previo.
- Proveedores distintos de AWS son compatibles si exponen PostgreSQL estándar.
- Se necesitan usuarios o identidades por workflow y rotación independiente.
- Los notebooks fallan claramente si falta la migración o un permiso, pero
  conservan sus salidas locales.
- CI valida migración, constraints y grants sobre PostgreSQL 17 real.

## Compatibilidad

Los nombres `DB_*` y las vistas `public.*` quedan deprecados durante VAAET 4.x y
se eliminarán en 5.0.0. Los backups legacy con `public.traffic_data` continúan
siendo fuentes raw explícitas.

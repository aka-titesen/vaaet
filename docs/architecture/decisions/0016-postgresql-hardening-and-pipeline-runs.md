# ADR-0016 — Hardening PostgreSQL y registro de ejecuciones

- Estado: aceptado
- Fecha: 2026-08-06
- Versión: VAAET ML 4.2.0
- Complementa: [ADR-0015](0015-postgresql-namespaces-security-and-hitl.md)

## Contexto

El contrato `vaaet-db-v2` ya separaba adquisición, inferencia y feedback, pero
los UUID de ejecución no tenían una entidad referenciada, las vistas de
compatibilidad dependían de `SELECT *`, algunos índices duplicaban constraints
`UNIQUE` y la documentación del esquema sólo existía fuera del motor.

En Python, `RAW_TELEMETRY_V2_COLUMNS` mezclaba identidad del contrato y versión.
La versión es necesaria en datos persistidos, pero el nombre del símbolo canónico
debe describir su responsabilidad y continuar siendo válido en la siguiente
revisión compatible.

## Decisión

- El esquema canónico se expresa mediante `BASE_RAW_TELEMETRY_COLUMNS`,
  `TELEMETRY_QUALITY_COLUMNS`, `TELEMETRY_METADATA_COLUMNS` y
  `CANONICAL_RAW_TELEMETRY_COLUMNS`.
- `traffic-telemetry-v2`, `traffic-features-v2`, bundle v2 y `vaaet-db-v2`
  permanecen como identificadores versionados de contratos persistidos.
- `vaaet_ops.pipeline_runs` registra ciclos collection, inference, training y
  review. Sólo guarda metadata tipada y redactada; nunca acepta DSN, secretos,
  certificados, rutas privadas ni mensajes de excepción.
- Los workflows escriben el ciclo mediante funciones `SECURITY DEFINER` con
  autorización por rol. Sin PostgreSQL producen un manifiesto JSON local.
- Raw, features y predicciones referencian una ejecución; feedback nuevo puede
  referenciar la sesión de review. La migración preserva UUID históricos antes
  de añadir las FK.
- Las vistas activas proyectan columnas explícitas. Códigos y etiquetas se
  validan en el motor y las correcciones HITL forman cadenas append-only lineales.
- Las 19 features permanecen deliberadamente denormalizadas como snapshot
  versionado. Aplicar 3FN a cada feature perjudicaría reproducibilidad y paridad
  train/serve sin eliminar una anomalía operacional real.

## Consecuencias

- Alembic `20260806_0002` es obligatorio antes de persistir con VAAET 4.2.0.
- La base es exclusiva de VAAET y `public` pierde el permiso global de creación;
  las vistas legacy continúan read-only durante 4.x.
- La constraint de suma de vehículos queda `NOT VALID`: protege escrituras
  nuevas y permite auditar históricos antes de su validación administrativa.
- No se particiona con el volumen actual. Se reevalúa a partir de diez millones
  de filas o evidencia de degradación en planes y tamaños.
- El rollback exige restaurar el backup previo porque las relaciones de linaje y
  las garantías append-only no deben degradarse automáticamente.

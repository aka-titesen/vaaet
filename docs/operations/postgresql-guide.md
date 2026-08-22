# Operación PostgreSQL — VAAET ML 4.5.3

Esta guía aplica a PostgreSQL 14+ en AWS RDS, Neon, Supabase o un servidor
propio. El endpoint debe ser accesible desde Colab; VPN y túneles quedan fuera de
alcance.

## Provisionamiento

Con una identidad administrativa local o de CI:

```bash
export VAAET_DATABASE_ADMIN_URL='postgresql+psycopg2://admin:...@host:5432/vaaet'
alembic upgrade head
psql 'postgresql://admin:...@host:5432/vaaet' \
  -v ON_ERROR_STOP=1 -f migrations/provision-roles.sql
```

Creá cuatro usuarios LOGIN en el panel del proveedor o con SQL administrativo y
concedé exactamente un group role:

```sql
GRANT vaaet_collection_role TO vaaet_collection_user;
GRANT vaaet_inference_role TO vaaet_inference_user;
GRANT vaaet_training_role TO vaaet_training_user;
GRANT vaaet_reviewer_role TO vaaet_reviewer_user;
```

El usuario administrador no se configura en Colab. Algunos proveedores gestionan
usuarios desde su panel; respetá ese mecanismo sin ampliar grants.

## TLS y rotación

Descargá la CA desde el proveedor y configurá `VAAET_DB_SSLMODE=verify-full` más
`VAAET_DB_SSLROOTCERT` local o `VAAET_DB_SSLROOTCERT_PEM` en Secrets. Al rotar:

1. crear una credencial nueva para un solo perfil;
2. actualizar Secrets sin imprimir el valor;
3. ejecutar el health check del notebook;
4. revocar la credencial anterior;
5. revisar conexiones y errores del proveedor.

`require` cifra pero no autentica al servidor. `disable` sólo se admite en
localhost.

## Backup y restauración controlada

```bash
pg_dump --format=custom --no-owner --no-acl \
  --schema=vaaet_raw --schema=vaaet_ml --schema=vaaet_feedback --schema=vaaet_ops \
  --file=vaaet-db-v2.backup "$DATABASE_URL"

pg_restore -l vaaet-db-v2.backup
```

El notebook usa cliente 17, inspecciona el catálogo y extrae las entradas
`TABLE DATA` exactas mediante una lista TOC controlada y
`--data-only --no-owner --no-acl`. No pasa nombres `schema.table` a `--table`,
porque `pg_restore` no admite esa calificación. Nunca restaura DDL o roles del
archivo sobre la base viva. Probá restauraciones administrativas en una base
aislada antes de depender del backup.

## Paquete portable

Exportá las tablas autorizadas a CSV mediante herramientas administrativas y
construí el paquete:

```bash
python scripts/export-training-dataset.py \
  --features telemetry-features.csv \
  --predictions traffic-predictions.csv \
  --validations human-validations.csv \
  --output vaaet-training-dataset-v1.zip
```

El script verifica checksums, filas, columnas y rango temporal. El paquete puede
contener además `--raw raw-telemetry.csv`; permanece fuera de Git.

## Diagnóstico

Ejecutá la auditoría read-only con el perfil training:

```bash
python scripts/audit-postgres-database.py --output postgres-audit.json
```

El informe contiene revisión Alembic, TLS, owners, comentarios, constraints no
validadas, índices, tamaños, autovacuum, controles de integridad y `EXPLAIN`
sin `ANALYZE`. Nunca incluye DSN, contraseñas, certificados ni mensajes de error.

## Continuidad y mantenimiento

- habilitar cifrado en reposo y backups administrados en el proveedor;
- conservar al menos 30 días y un RPO máximo de 24 horas;
- probar trimestralmente la restauración en una base aislada y registrar el RTO;
- revisar parches de PostgreSQL y del proveedor periódicamente;
- auditar conexiones y DDL con logs del proveedor o `pgaudit` cuando esté disponible;
- ejecutar el auditor después de cada migración y antes de promover un modelo.

El backup lógico diario puede sustituirse por PITR administrado si cumple o
mejora el RPO. Los backups nunca se restauran directamente sobre producción.

## Diagnóstico rápido

- timeout/red: confirmar allowlist/firewall y endpoint público del proveedor;
- TLS: actualizar CA y hostname, sin degradar silenciosamente a `disable`;
- permisos: volver a aplicar `provision-roles.sql`, no usar administrador;
- schema ausente: ejecutar `alembic current` y `alembic upgrade head` fuera de Colab;
- backup incompatible: usar PostgreSQL client 17 o exportar CSV;
- escritura fallida: conservar y descargar CSV/video/paquete local antes de
  reiniciar el runtime.

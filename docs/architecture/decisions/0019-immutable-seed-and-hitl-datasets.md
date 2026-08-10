# ADR-0019 — Datasets semilla y HITL inmutables

- Estado: aceptado
- Fecha: 2026-08-10
- Versión: VAAET ML 4.5.0
- Complementa: [ADR-0017](0017-seed-bootstrap-and-hitl-retraining.md) y [ADR-0018](0018-versioned-frozen-human-holdouts.md)

## Contexto

Hasta VAAET 4.4.0 el notebook de entrenamiento escribía la semilla procesada en
una ruta fija y el notebook de inferencia exportaba feedback offline a otro ZIP
fijo. Repetir cualquiera de esas operaciones podía sobrescribir contenido sin
dejar una identidad estable. Además, el paquete HITL no se generaba cuando la
revisión también se persistía en PostgreSQL y el manifiesto del modelo no
enumeraba los datasets portables exactos usados durante el entrenamiento.

La versión `v1` de un nombre de paquete identifica el formato contractual; no
debe confundirse con la generación de datos ni con un número de ejecución.

## Decisión

Se separan tres responsabilidades:

1. La semilla procesada es un snapshot `vaaet-seed-bootstrap-v1` excepcionalmente
   versionado. `current.json` apunta a una generación inmutable. Los mismos datos
   reutilizan el fingerprint vigente; datos distintos exigen
   `CREATE_NEW_VERSION` y un motivo.
2. Cada sesión finalizada de revisión crea un
   `vaaet-training-dataset-v1.zip`, incluso cuando PostgreSQL está disponible.
   Features, predicciones, validaciones y filas pendientes permanecen ligadas a
   un `pipeline_run_id` y usan UUID globales.
3. `vaaet-dataset-catalog-v1` selecciona los paquetes HITL activos. El
   entrenamiento verifica checksums, resuelve cadenas append-only de correcciones
   entre paquetes y excluye predicciones no validadas.
4. Cada entrenamiento escribe un `vaaet-training-input-lock-v1` con el snapshot
   semilla, revisión del catálogo, IDs y fingerprints de paquetes, holdout humano
   y conteos finales. `model-manifest.json` conserva el UUID y fingerprint del
   lock sin añadir un quinto archivo al bundle de serving.

Google Drive montado actúa como filesystem canónico en Colab. Las escrituras de
pointer y catálogo usan archivo temporal, validación y reemplazo atómico. Un
fallo de sincronización conserva el paquete local como `pending-sync`; el catálogo
no se actualiza hasta que el ZIP canónico exista y su SHA-256 coincida.

Los paquetes legacy de semilla con contrato `vaaet-training-dataset-v1` y
procedencia `seed-bootstrap/weak-proxy` pueden importarse explícitamente durante
VAAET 4.x. Las rutas fijas dejan de ser fuentes implícitas.

## Integridad global del feedback

- Features, predicciones y validaciones portables usan UUID.
- Una corrección sólo puede sustituir una validación de la misma predicción.
- Cada validación tiene como máximo un sucesor y cada predicción una única raíz.
- Ciclos, ramas, referencias ausentes y etiquetas/features incompatibles detienen
  la ingestión.
- La hoja efectiva de cada cadena es ground truth; Accident continúa reservado al
  detector jerárquico y nunca ingresa al target del MLP.

## Consecuencias

- Repetir una finalización o una semilla idéntica es idempotente.
- Ninguna generación anterior se elimina o sobrescribe automáticamente.
- PostgreSQL continúa siendo la autoridad operacional y los ZIP representan
  fotografías portables auditables.
- El catálogo puede crecer; cuarentena y limpieza serán operaciones
  administrativas explícitas futuras.
- El contrato del bundle permanece en v2 y la base PostgreSQL no cambia.

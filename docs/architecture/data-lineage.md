# Linaje de datos — VAAET ML 4.5.3

## Flujo operacional

```mermaid
flowchart LR
    V["Video"] --> C["Collection"]
    C --> R[("vaaet_raw.traffic_data")]
    V --> I["Inference"]
    I --> F[("vaaet_ml.telemetry_features")]
    I --> P[("vaaet_ml.traffic_predictions")]
    P --> Q["Explicit HITL review"]
    Q --> H[("vaaet_feedback.human_validations")]
    Q --> HP["Immutable HITL session package"]
    HP --> HC["Active HITL catalog"]
    R --> S["Seed bootstrap"]
    S --> SP["Immutable seed snapshot"]
    SP --> T["HITL retraining"]
    HC --> T
    H --> T
    H --> HO["Frozen human holdout"]
    HO --> T
    T --> L["Training input lock"]
    L --> B["Bundle v2 candidate"]
```

Cada ejecución de adquisición o inferencia genera un `pipeline_run_id`. Los
timestamps se persisten como `TIMESTAMPTZ` UTC; valores históricos sin zona se
interpretan como `America/Argentina/Buenos_Aires` durante la migración.
El mismo contrato se aplica en memoria, CSV, backups, paquetes y sintéticos:
`record_time` siempre es timezone-aware en UTC. La hora argentina se recupera
únicamente para las features circadianas y la presentación, sin alterar el
instante persistido.

Desde 4.2.0 el UUID referencia `vaaet_ops.pipeline_runs`, que registra workflow,
estado, commit, contratos y conteos sin secretos. Cuando PostgreSQL es opcional,
el mismo contrato se conserva como JSON bajo `data/processed/pipeline-runs/`.

## Adquisición

`collect_traffic_telemetry.ipynb` produce video anotado, CSV raw y, al habilitarlo,
inserta exclusivamente en `vaaet_raw.traffic_data`. La unicidad
`(clip_id, record_time)` hace idempotente la repetición. No se persisten frames,
patentes ni identidades.

## Inferencia y revisión

`analyze_traffic_video.ipynb` persiste las 19 features y predicciones estables
0–2 con el perfil `inference`. Después del clip, una celda opcional abre una cola
`priority` o `all`; no interrumpe el procesamiento con pop-ups. El perfil
`review` agrega validaciones sin modificar predicciones. El modo `all` conserva la
validación previa como `supersedes_validation_id` al corregir. Accident requiere nota y
revisión explícita del contexto. Sin base disponible se exporta
un paquete inmutable `vaaet-training-dataset-v1.zip` por sesión. El paquete se
registra en `vaaet-dataset-catalog-v1`; los registros omitidos permanecen como no
supervisados y nunca son targets.

## Entrenamiento

`TrainingIngestionPlan` declara uno de dos modos. El tipo nunca se infiere por
columnas:

- `SEED_BOOTSTRAP`: raw desde servidor, backup o CSV; auditoría, ingeniería de
  las 19 features, etiqueta proxy y paquete semilla reutilizable;
- `HITL_RETRAINING`: paquete semilla más feedback con feature schema compatible
  y última validación humana, sin recalcular features;
- estados humanos 0–2: candidatos al MLP;
- estado humano 3: evaluación del detector de incidente, nunca target del MLP.

La etiqueta humana prevalece sobre el proxy para el mismo minuto. La memoria
proxy decrece por clase al crecer el soporte humano y desaparece en los umbrales
documentados por ADR-0017. Conflictos
entre validaciones efectivas detienen el entrenamiento. Sintéticos sólo aparecen
en train y siempre permanecen identificados.

Con `HUMAN_HOLDOUT_FROZEN=True`, validation y test se materializan como
`vaaet-human-holdout-v1` bajo Google Drive. El snapshot conserva las features y
etiquetas exactas, mientras `current.json` selecciona la generación activa. Sus
grupos se excluyen de train; actualizar el benchmark crea una nueva generación
sin sobrescribir la anterior. PostgreSQL continúa siendo la autoridad del
feedback y el ZIP sólo representa la fotografía reproducible de evaluación.

## Artefactos de dataset e input lock

La semilla procesada vive bajo `data/seed-bootstrap/snapshots/` en Drive y
`current.json` apunta a una generación inmutable. Las sesiones HITL viven bajo
`data/hitl-reviews/YYYY/MM/DD/` y `catalog.json` selecciona paquetes `active`.
Antes de exportar el bundle, entrenamiento escribe `vaaet-training-input-lock-v1`
con el snapshot semilla, revisión exacta del catálogo, fingerprints de cada ZIP y
holdout utilizado. El lock aporta linaje reproducible; no contiene pesos ni
reemplaza PostgreSQL.

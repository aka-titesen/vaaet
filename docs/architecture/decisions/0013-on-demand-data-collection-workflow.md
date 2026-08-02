# ADR-0013: Workflow de adquisición de datos bajo demanda

**Estado:** Aceptado  
**Fecha:** 2026-08-01

## Contexto

El notebook que produjo la telemetría inicial estaba tratado como material archivado y contenía un motor monolítico, instalaciones manuales y demos sintéticas. Aunque su ejecución habitual ya no sea necesaria, la adquisición es una capacidad válida para ampliar datos, reproducir procedencia y volver a calibrar el sistema.

## Decisión

- La adquisición es un tercer workflow de primer nivel: `notebooks/data-collection/collect_traffic_telemetry.ipynb`.
- Su ejecución es opcional y bajo demanda, no una fase obsoleta.
- El procesamiento visual compartido vive en `vaaet.vision.analysis.analyze_video` y sirve tanto a adquisición como a inferencia.
- Adquisición genera video anotado y `data/raw/traffic_data_raw.csv`; PostgreSQL es opcional, explícito e idempotente sobre `(clip_id, record_time)`.
- `record_time` proviene del nombre estándar del puente. Para nombres libres se usa una única hora de procesamiento y se declara su menor trazabilidad.
- Los notebooks tienen una sola celda de bootstrap editable y consumen extras definidos únicamente en `pyproject.toml`.

## Consecuencias

Se elimina duplicación visual y la adquisición queda reproducible sin convertirla en requisito para cada entrenamiento. Los pesos YOLO continúan descargándose en runtime. No cambian las 19 features, estados, umbrales, MLP, bundle ni esquema PostgreSQL.

ADR-0001 y ADR-0009 conservan su contexto histórico; esta decisión gobierna la clasificación actual del workflow.

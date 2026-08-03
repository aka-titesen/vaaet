# Linaje de datos — VAAET ML 4.0.0

## Adquisición bajo demanda

`notebooks/data-collection/collect_traffic_telemetry.ipynb` recibe video SISE, ejecuta YOLO, SORT, compensación de cámara y estimación de velocidad mediante `vaaet.vision.analysis`. Produce video anotado y registros por minuto.

```text
MP4 -> detecciones -> tracks -> velocidades y calidad -> agregados por minuto
    -> data/raw/traffic_data_raw.csv
    -> traffic_data (PostgreSQL, opcional)
```

El CSV se acumula y deduplica por `(clip_id, record_time)`. En nombres `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`, los timestamps derivan de la captura y soportan cruces de medianoche. Un nombre libre utiliza la hora de procesamiento y tiene menor trazabilidad.

Los videos pueden contener información sensible y no se versionan. VAAET no extrae patentes ni identidades y no persiste frames individuales.

## Entrenamiento

El notebook de entrenamiento carga `traffic_data`, CSV o backup, audita schema y procedencia, genera las 19 features v2 dentro de cada clip/segmento continuo y etiqueta provisionalmente tres estados estables. Reserva grupos temporalmente posteriores para test, crea validation por clips, ajusta el scaler sólo con train y usa class weights limitados. Los sintéticos sólo pueden aparecer en train; Accident sintético se reserva para estrés técnico del detector.

El resultado es un bundle de cuatro archivos: modelo, scaler, mapping y manifiesto. El manifiesto contiene commit, timestamp UTC, schema, dependencias, procedencia, presencia sintética, métricas y checksums. DVC versiona el bundle como unidad.

## Inferencia y feedback

El notebook de inferencia valida el bundle antes de deserializar, clasifica únicamente minutos completos y ejecuta la misma política jerárquica. Una sospecha permanece Congested con alerta separada; sólo una corrección humana validada produce Accident. Inferencia persiste feedback, pero el reentrenamiento ocurre exclusivamente en el notebook de entrenamiento.

```text
raw acquisition -> engineered dataset -> approved bundle
       ^                                      |
       |                                      v
       +-- human feedback <- classified telemetry
```

Este bucle es una capacidad de mejora continua, no un trigger automático de Continuous Training.

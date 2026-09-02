# ADR-0021: Core portable y laboratorio ML separados

- Estado: aceptada
- Fecha: 2026-08-27
- Decisores: Facundo Nicolás González
- Actualiza: ADR-0020

## Contexto

La percepción vehicular, la telemetría por minuto, las 19 features, la política
conservadora de estados y el bundle v2 serán reutilizados por los notebooks y
por workers de la futura API. Mantenerlos dentro del componente de laboratorio
forzaría a la aplicación a conocer DVC, Drive, PostgreSQL y configuración de
Colab.

## Decisión

La raíz sigue teniendo un único Git y DVC, pero incorpora dos distribuciones
Python internas:

```text
vaaet/
├─ vaaet-core/  # distribución vaaet-core==0.1.0, import vaaet
├─ vaaet-ml/    # distribución vaaet-ml==4.5.4, import vaaet_ml
├─ vaaet-app/   # reservado, sin código de API ni web todavía
├─ docs/
├─ .dvc/
└─ .github/
```

`vaaet-core` contiene contratos y errores de dominio, timestamps, telemetría,
ingeniería y etiquetado requeridos por serving, ciclo de vida/políticas del
bundle, validación manifest-first, carga de bundles y visión. Sus APIs públicas
son `vaaet.vision.analyze_video()`, `TrafficStatePrediction`,
`VideoAnalysisResult`, `vaaet.inference.load_traffic_bundle()` y
`TrafficStateEngine`. El motor clasifica minutos en memoria y nunca implementa
colas, workers, DVC, Drive ni persistencia.

`vaaet-ml` conserva los datasets, entrenamiento, evaluación, artefactos DVC,
notebooks Colab, migraciones, PostgreSQL y utilidades de laboratorio. Depende de
`vaaet-core==0.1.0`; en desarrollo y CI ambos componentes se instalan de forma
local y explícita. Los notebooks importan `vaaet` para operaciones y
`vaaet_ml` para laboratorios.

La futura API será el único adaptador entre la web y el core. Recibirá videos,
creará trabajos asíncronos consultables y almacenará referencias a resultados,
telemetría y video anotado. Sus workers invocarán el core de forma síncrona,
validarán el manifiesto v2 antes de deserializar y decidirán persistencia con
credenciales propias. La Web App sólo consumirá HTTP versionado; no importará
Python ni conocerá rutas, DVC, Drive, PostgreSQL, modelos o datos HITL.

## Invariantes

- DVC continúa en la raíz y gobierna `vaaet-ml/artifacts/traffic-state/`.
- El core recibe sólo un directorio local de bundle; desconoce su procedencia.
- Se preservan bundle v2, 19 features, tres salidas aprendidas y cuatro estados
  públicos. `Accident` nunca se publica automáticamente.
- `stationary_confirmed` sigue siendo una señal conservadora por vehículo y
  minuto; no se agregan zonas, eventos de parking ni permanencia.
- No se implementan todavía API, framework, frontend, cola distribuida ni
  persistencia de aplicación.

## Consecuencias

Las pruebas se separan por propiedad: visión, telemetría, contratos, bundle e
inferencia pertenecen al core; entrenamiento, evaluación, datos, PostgreSQL y
notebooks pertenecen a ML. CI instala y verifica los componentes por separado y
en conjunto. La validación manual de Colab GPU, Drive, DVC, YOLO y PostgreSQL
sigue siendo necesaria antes de promoción externa.

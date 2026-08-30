# ADR-0025 — Segmentos multi-vista con calibración explícita

- Estado: aceptada
- Fecha: 2026-08-30
- Decisores: Facundo Nicolás González
- Actualiza: alcance de cámara única de ADR-0013 y SRS
- Complementa: ADR-0021

## Contexto

El pipeline de visión procesa un clip ordenado con una única relación entre
píxeles y metros. Un cambio de cámara, pan, tilt o zoom invalida esa relación,
rompe IDs de tracking y puede mezclar conteos o velocidades incompatibles en el
mismo minuto.

## Decisión

`analyze_video()` acepta opcionalmente un `VideoViewPlan` portable. El plan
asigna rangos semiabiertos de frames a `CameraCalibration` versionadas. Cada
perfil declara resolución y al menos dos tramos viales de longitud conocida a
distintas profundidades; el core interpola una escala local con el contacto
inferior del bounding box. Los planes reales se cargan fuera de Git, DVC y
outputs públicos mediante un adaptador del consumidor.

Al cambiar de segmento el core reinicia flujo óptico, SORT, smoothing y estado
estacionario. No conserva identidad entre vistas. Si una transición atraviesa un
minuto, descarta ese minuto de telemetría y clasificación; también reinicia el
contexto de predicción y exige dos minutos válidos antes de volver a invocarla.
`VideoAnalysisResult.view_segments` expone sólo metadatos seguros del segmento;
la telemetría v2 y las 19 features no cambian.

## Consecuencias

- Sin `VideoViewPlan` se mantiene el comportamiento previo de vista única.
- El soporte es offline y secuencial; no agrega streaming, cámaras simultáneas,
  colas, reidentificación ni eventos de parking.
- Un cambio no declarado no recibe escala inventada: el video debe reprocesarse
  con un plan corregido.
- La interpolación por profundidad no es una homografía ni demuestra precisión
  de radar/GPS. Una homografía sólo podrá cambiar telemetría después de contar
  con calibración geométrica versionada y comparación contra ground truth.

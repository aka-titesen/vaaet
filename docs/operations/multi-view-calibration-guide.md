# Guía de calibración multi-vista

`vaaet-core` permite analizar un MP4 offline con cambios de cámara o encuadre
cuando cada tramo estable se declara previamente. Esta función evita mezclar
geometrías; no identifica cámaras automáticamente ni convierte el resultado en
una medición certificada de velocidad.

## Preparar un plan privado

Para cada cámara y zoom estable, medí como mínimo dos referencias conocidas del
plano vial a profundidades distintas. Priorizá segmentos longitudinales al
sentido de circulación. No uses dimensiones de vehículos ni una distancia
estimada visualmente. Guardá el archivo fuera de Git, DVC y outputs públicos;
por ejemplo, como `mi-video.vaaet-view-plan.private.json` en un directorio local
o privado.

El esquema es `vaaet-view-plan-v1`. Este ejemplo es sintético y no corresponde
al puente:

```json
{
  "schema_version": "vaaet-view-plan-v1",
  "profiles": [
    {
      "profile_id": "cam-norte-amplia",
      "revision": "v1",
      "frame_size": [1920, 1080],
      "references": [
        {
          "reference_id": "far-lane",
          "pixel_start": [210, 180],
          "pixel_end": [310, 180],
          "meters": 20.0
        },
        {
          "reference_id": "near-lane",
          "pixel_start": [280, 840],
          "pixel_end": [680, 840],
          "meters": 20.0
        }
      ]
    },
    {
      "profile_id": "cam-sur-zoom",
      "revision": "v1",
      "frame_size": [1920, 1080],
      "references": [
        {
          "reference_id": "far-lane",
          "pixel_start": [420, 260],
          "pixel_end": [500, 260],
          "meters": 20.0
        },
        {
          "reference_id": "near-lane",
          "pixel_start": [250, 800],
          "pixel_end": [640, 800],
          "meters": 20.0
        }
      ]
    }
  ],
  "segments": [
    {"start_frame": 1, "end_frame": 1801, "profile_id": "cam-norte-amplia"},
    {"start_frame": 1801, "end_frame": null, "profile_id": "cam-sur-zoom"}
  ]
}
```

Los rangos son 1-indexados y semiabiertos: el segundo segmento comienza en el
frame `1801`. Los segmentos deben ser contiguos, el último debe tener
`"end_frame": null`, y todos los perfiles deben coincidir con la resolución del
video. Versioná el identificador o `revision` cuando cambie la geometría.

## Usar en notebooks

En adquisición o inferencia, editá únicamente `VIEW_PLAN_PATH` en la celda de
configuración. `None` conserva el comportamiento histórico de una vista; una
ruta local privada carga y valida el plan antes de procesar. Los datos de
calibración no se montan, copian ni publican automáticamente.

Al cruzar un cambio de vista, el video anotado continúa, pero el minuto mixto se
omite de telemetría, clasificación, PostgreSQL e HITL. El resultado expone
`view_segments` para auditar perfil, revisión y minutos omitidos. No se asume
que un vehículo visto en dos cámaras sea el mismo.

## Validación manual obligatoria

1. Confirmá que cada perfil usa referencias reales y la resolución exacta.
2. Procesá un clip estable por perfil y compará desplazamiento observado contra
   las referencias medidas.
3. Procesá un clip con transición declarada y verificá el minuto omitido, el
   reinicio de IDs y la demora de dos minutos antes de una nueva clasificación.
4. Registrá cualquier comparación contra radar o GPS fuera del bundle. Hasta
   entonces no hay MAE físico publicado ni soporte de homografía.

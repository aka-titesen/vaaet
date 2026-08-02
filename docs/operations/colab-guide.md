# Guía de ejecución en Google Colab

## Runtime

Seleccioná **Runtime > Change runtime type > GPU**. VAAET soporta Python 3.10–3.12 y está preparado para el runtime Colab 2026.04 (Python 3.12, NumPy 2.0 y TensorFlow 2.19). La primera celda muestra versiones, GPU y commit para que cada ejecución sea auditable.

Referencias oficiales: [versiones del runtime de Colab](https://research.google.com/colaboratory/runtime-version-faq.html) e [instalación headless de Ultralytics](https://docs.ultralytics.com/quickstart/).

## Orden recomendado

1. Ejecutá adquisición sólo cuando necesites ampliar la telemetría inicial.
2. Ejecutá entrenamiento cuando cambie el dataset aprobado y conservá el bundle completo.
3. Ejecutá inferencia con un clip y un bundle validado.

Cada notebook clona o actualiza el repositorio en `/content/vaaet`, define un único `REPO_ROOT`, instala una sola vez en modo editable y ejecuta `pip check`. En el runtime administrado, este diagnóstico puede informar inconsistencias globales ajenas a VAAET —por ejemplo, `ipython` sin el paquete opcional `jedi`—; la advertencia se muestra, pero no bloquea los imports explícitos del workflow. En CI, donde el entorno es limpio, `pip check` continúa siendo estricto.

Si el runtime se reinicia, volvé a ejecutar desde la primera celda; los archivos de `/content` son efímeros.

## Secrets y PostgreSQL

Creá los secretos `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` y `DB_PASSWORD` en el panel de Secrets y habilitá acceso al notebook. Las variables de entorno son fallback. La persistencia permanece deshabilitada hasta cambiar explícitamente el interruptor del notebook.

## Artefactos y Drive

El entrenamiento genera cuatro archivos bajo `artifacts/traffic-state/` y puede copiarlos juntos a `MyDrive/vaaet-ml/artifacts/traffic-state`. Inferencia intenta, en orden: bundle local, Drive y upload manual. El manifiesto se valida antes de cargar Keras o joblib.

Después de aprobar un modelo, ejecutá localmente `dvc add artifacts/traffic-state` y `dvc push`. Los pesos YOLO se descargan desde Ultralytics en runtime y nunca se versionan.

## Entradas y salidas

- Adquisición: MP4 → MP4 anotado + `data/raw/traffic_data_raw.csv` + `traffic_data` opcional.
- Entrenamiento: CSV/BD → dataset procesado + bundle de cuatro archivos.
- Inferencia: MP4 + bundle → MP4 anotado + DataFrames + PostgreSQL opcional.

Descargá los outputs antes de cerrar la sesión. Un backup PostgreSQL binario puede requerir una versión de `pg_restore` igual o posterior a la que lo creó; si el runtime no la incluye, preferí el CSV exportado o instalá el cliente PostgreSQL correspondiente.

## Checklist manual

- Confirmar GPU y descarga automática de YOLO.
- Procesar un clip real en adquisición y descargar CSV/video.
- Activar una vez la persistencia y comprobar deduplicación en `traffic_data`.
- Entrenar y validar los cuatro archivos del bundle.
- Cargar el bundle desde local, Drive y upload en inferencia.
- Confirmar que un bundle incompleto o incompatible se rechaza antes de inferir.

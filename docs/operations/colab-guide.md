# Guía de ejecución en Google Colab

## Runtime

Seleccioná **Runtime > Change runtime type > GPU**. VAAET soporta Python 3.10–3.12 y está preparado para el runtime Colab 2026.04 (Python 3.12, NumPy 2.0 y TensorFlow 2.19). La primera celda muestra versiones, GPU y commit para que cada ejecución sea auditable.

Referencias oficiales: [versiones del runtime de Colab](https://research.google.com/colaboratory/runtime-version-faq.html) e [instalación headless de Ultralytics](https://docs.ultralytics.com/quickstart/).

## Orden recomendado

1. Ejecutá adquisición sólo cuando necesites ampliar la telemetría inicial.
2. Ejecutá entrenamiento cuando cambie el dataset aprobado y conservá el bundle completo.
3. Ejecutá inferencia con un clip y un bundle validado.

Cada notebook clona o actualiza el repositorio en `/content/vaaet`, define un único `REPO_ROOT` e instala una vez un wheel local con sus extras. La instalación no es editable en Colab porque el nombre `/content/vaaet` puede interpretarse como un paquete namespace y ocultar `src/vaaet`; después de instalar se limpia el caché de módulos y se valida el origen real del paquete. El desarrollo local sí conserva `pip install -e`.

El diagnóstico `pip check` puede informar inconsistencias globales ajenas a VAAET —por ejemplo, `ipython` sin el paquete opcional `jedi`—; la advertencia se muestra, pero no bloquea los imports explícitos del workflow. En CI, donde el entorno es limpio, `pip check` continúa siendo estricto.

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

Descargá los outputs antes de cerrar la sesión. Cuando se carga `traffic_data.backup`, el notebook de entrenamiento instala automáticamente `postgresql-client` si `pg_restore` no está disponible y muestra su versión antes de continuar. Esta dependencia pertenece al sistema operativo y por eso no forma parte de `pyproject.toml`.

Ubuntu puede incluir una versión de PostgreSQL fijada por la distribución. Si el backup fue creado con una versión posterior, el notebook conserva el diagnóstico de incompatibilidad: usá `traffic_data_raw.csv` o instalá manualmente un cliente igual o posterior. El workflow no agrega repositorios APT externos de forma silenciosa.

## Checklist manual

- Confirmar GPU y descarga automática de YOLO.
- Procesar un clip real en adquisición y descargar CSV/video.
- Activar una vez la persistencia y comprobar deduplicación en `traffic_data`.
- Entrenar y validar los cuatro archivos del bundle.
- Cargar el bundle desde local, Drive y upload en inferencia.
- Confirmar que un bundle incompleto o incompatible se rechaza antes de inferir.

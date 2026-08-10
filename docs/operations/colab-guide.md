# Guía de ejecución en Google Colab

## Runtime

Seleccioná **Runtime > Change runtime type > GPU**. VAAET soporta Python 3.10–3.12 y está preparado para el runtime Colab 2026.04 (Python 3.12, NumPy 2.0 y TensorFlow 2.19). La primera celda muestra versiones, GPU y commit para que cada ejecución sea auditable.

Referencias oficiales: [versiones del runtime de Colab](https://research.google.com/colaboratory/runtime-version-faq.html) e [instalación headless de Ultralytics](https://docs.ultralytics.com/quickstart/).

## Orden recomendado

1. Ejecutá adquisición sólo cuando necesites ampliar la telemetría inicial.
2. Ejecutá entrenamiento cuando cambie el dataset y revisá el informe de elegibilidad antes de conservar el bundle.
3. Ejecutá inferencia con un clip y un bundle v2 validado. Un piloto requiere
   `ALLOW_PILOT_BUNDLE=True`; un candidato no aprobado requiere una autorización
   experimental independiente.

Cada notebook clona o actualiza el repositorio en `/content/vaaet`, define un único `REPO_ROOT` e instala una vez un wheel local con sus extras. La instalación no es editable en Colab porque el nombre `/content/vaaet` puede interpretarse como un paquete namespace y ocultar `src/vaaet`; después de instalar se limpia el caché de módulos y se valida el origen real del paquete. El desarrollo local sí conserva `pip install -e`.

El diagnóstico `pip check` puede informar inconsistencias globales ajenas a VAAET —por ejemplo, `ipython` sin el paquete opcional `jedi`—; la advertencia se muestra, pero no bloquea los imports explícitos del workflow. En CI, donde el entorno es limpio, `pip check` continúa siendo estricto.

Si el runtime se reinicia, volvé a ejecutar desde la primera celda; los archivos de `/content` son efímeros.

## Secrets y PostgreSQL

Creá el endpoint común `VAAET_DB_HOST`, `VAAET_DB_PORT`, `VAAET_DB_NAME` y
`VAAET_DB_SSLMODE` en Colab Secrets. Para `verify-full`, agregá
`VAAET_DB_SSLROOTCERT_PEM` con el certificado CA del proveedor. Luego habilitá
sólo las credenciales necesarias:

| Workflow | Secrets |
|---|---|
| Adquisición | `VAAET_COLLECTION_DB_USER`, `VAAET_COLLECTION_DB_PASSWORD` |
| Inferencia | `VAAET_INFERENCE_DB_USER`, `VAAET_INFERENCE_DB_PASSWORD` |
| Entrenamiento | `VAAET_TRAINING_DB_USER`, `VAAET_TRAINING_DB_PASSWORD` |
| Revisión | `VAAET_REVIEW_DB_USER`, `VAAET_REVIEW_DB_PASSWORD`, `VAAET_REVIEWER_ID` |

Los notebooks consultan Secrets directamente: no copian contraseñas a variables,
celdas ni outputs. `sslmode=require` se admite con advertencia; `disable` sólo en
localhost. Los nombres `DB_*` funcionan de forma deprecada durante 4.x.

La migración se aplica una sola vez desde un entorno administrativo, nunca desde
Colab:

```bash
alembic upgrade head
psql 'postgresql://admin:...@host:5432/vaaet' -f migrations/provision-roles.sql
```

## Artefactos y Drive

El entrenamiento genera cuatro archivos bajo `artifacts/traffic-state/` y puede copiarlos juntos a `MyDrive/vaaet-ml/artifacts/traffic-state`. Inferencia intenta, en orden: bundle local, Drive y upload manual. El manifiesto se valida antes de cargar Keras o joblib. `ALLOW_PILOT_BUNDLE=True` permite ejecutar conscientemente el bootstrap; `ALLOW_EXPERIMENTAL_BUNDLE=True` queda reservado para candidatos HITL no aprobados. Ninguno promociona el artefacto.

Después de aprobar un modelo, ejecutá localmente `dvc add artifacts/traffic-state` y `dvc push`. Los pesos YOLO se descargan desde Ultralytics en runtime y nunca se versionan.

## Entradas y salidas

- Adquisición: MP4 → MP4 anotado + `data/raw/traffic_data_raw.csv` + `vaaet_raw.traffic_data` opcional.
- Entrenamiento semilla: raw explícito → 19 features → paquete semilla + bundle piloto.
- Reentrenamiento HITL: paquete semilla + features validadas → bundle candidato.
- Inferencia: MP4 + bundle → MP4 anotado + features/predicciones PostgreSQL opcionales.

El clasificador sólo consume minutos completos. Durante una ventana parcial se muestra el último estado estable. `Accident` nunca es automático: una evidencia persistente produce `Congested + accident_rule_triggered`; el código 3 exige feedback humano validado.

La cantidad de filas temporales es `floor(duración_del_clip / 60)`: 16 segundos
producen cero filas, 60 segundos producen una y 125 segundos producen dos; los
cinco segundos finales del último ejemplo se conservan en el video anotado pero
no se persisten como telemetría. Un clip menor a un minuto finaliza correctamente,
descarga su video anotado y omite CSV, PostgreSQL, clasificación y HITL. Usá clips
de varios minutos para comprobar transiciones, persistencia e histéresis.

La revisión HITL es una celda explícita posterior al clip, no un pop-up durante
inferencia. `priority` muestra incidentes candidatos, baja confianza, abstenciones
y transiciones; `all` permite revisar cada minuto. Sin conexión de review, la
sesión exporta `vaaet-training-dataset-v1.zip`, que luego puede declararse como
`DatasetPackageSource` en entrenamiento. Sólo validaciones humanas ingresan como
etiquetas; Accident se reserva para evaluar el detector.

En entrenamiento, seleccioná explícitamente `TrainingMode.SEED_BOOTSTRAP` o
`TrainingMode.HITL_RETRAINING`. El primer modo declara `RAW_SOURCES`, calcula las
features una vez y crea `vaaet-seed-bootstrap-v1.zip`. El segundo declara ese
paquete en `SEED_SOURCES` y las correcciones en `FEEDBACK_SOURCES`; no vuelve a
ejecutar `pg_restore` ni ingeniería. `ENABLE_DATA_UPLOAD=False` evita abrir el
selector cuando todas las fuentes son PostgreSQL. Cada tipo se habilita de forma
explícita: nunca se adivina por columnas.

`HUMAN_HOLDOUT_FROZEN` permanece `False` por defecto porque el Inicio Semilla no
tiene ground truth humano. En `TrainingMode.HITL_RETRAINING`, configurá:

```python
HUMAN_HOLDOUT_FROZEN = True
HUMAN_HOLDOUT_ACTION = HumanHoldoutAction.REUSE_OR_CREATE
HUMAN_HOLDOUT_UPDATE_REASON = None
```

La primera ejecución crea validation y test bajo
`MyDrive/vaaet-ml/data/holdouts/`; las posteriores cargan exactamente el ZIP
señalado por `current.json`. Los registros humanos nuevos quedan disponibles
para train y no alteran el benchmark automáticamente. Deben existir al menos
tres clips o grupos por cada estado estable para congelarlo.

Para incorporar nuevas revisiones al benchmark, usá una vez:

```python
HUMAN_HOLDOUT_ACTION = HumanHoldoutAction.CREATE_NEW_VERSION
HUMAN_HOLDOUT_UPDATE_REASON = "Incorporación de clips revisados de agosto de 2026"
```

Esto crea una nueva generación sin sobrescribir la anterior. La operación es
idempotente con la misma fotografía de datos. Después de resolverla, el runtime
vuelve a `REUSE_OR_CREATE`. Un fallo al montar Drive detiene el entrenamiento:
no se sustituye el benchmark por un split aleatorio.

Todo `record_time` se normaliza a UTC. Los backups y CSV legacy sin zona se
interpretan como `America/Argentina/Buenos_Aires`; por eso su rango UTC puede
desplazarse tres horas respecto del texto original. Las features horarias siguen
representando la hora argentina. Después de la aumentación, entrenamiento muestra
`Canonical timestamp timezone: UTC` antes de auditar el dataset.

Descargá los outputs antes de cerrar la sesión. El backup canónico usa el formato de archivo `1.16`, que requiere PostgreSQL 17 o posterior. Cuando debe procesarlo, el notebook de entrenamiento configura de forma visible el repositorio APT oficial PGDG, instala únicamente `postgresql-client-17` y ejecuta directamente `/usr/lib/postgresql/17/bin/pg_restore`. La clave, URL, codename y arquitectura del repositorio se declaran explícitamente en la celda; no se instala el servidor PostgreSQL. El importador selecciona entradas exactas del catálogo del archivo, por lo que admite tanto `public.traffic_data` legacy como `vaaet_raw.traffic_data` moderno sin restaurar DDL ni conectarse a una base viva.

Cada workflow registra un `pipeline_run_id`. Con PostgreSQL usa el schema
`vaaet_ops`; sin conexión genera un manifiesto JSON redactado en
`data/processed/pipeline-runs/`. Estos manifiestos no contienen rutas privadas,
credenciales, DSN, certificados ni mensajes de excepción.

El cliente es una dependencia del sistema operativo y por eso no forma parte de `pyproject.toml`. Si PGDG no está disponible o la instalación falla, cargá `traffic_data_raw.csv`; el notebook conserva el diagnóstico original y no continúa con datos vacíos.

## Checklist manual

- Confirmar GPU y descarga automática de YOLO.
- Procesar un clip real en adquisición y descargar CSV/video.
- Aplicar Alembic fuera de Colab y verificar health check/rol sin secretos.
- Activar una vez la persistencia y comprobar deduplicación en `vaaet_raw.traffic_data`.
- Entrenar y validar los cuatro archivos del bundle.
- Cargar el bundle desde local, Drive y upload en inferencia.
- Confirmar que un bundle incompleto o incompatible se rechaza antes de inferir.
- Confirmar que un bundle semilla queda `pilot`, no elegible para producción, y
  que inferencia muestra esa condición.
- Revisar candidatos de incidente sobre hard negatives y acumular horas negativas; no publicar recall de Accident sin casos reales.
- Completar una cola HITL, persistir con el perfil review y repetir una inferencia para confirmar que la validación sobrevive.
- Reentrenar combinando el paquete semilla y una fuente de feedback validado;
  comprobar el peso proxy decreciente por clase.

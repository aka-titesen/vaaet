# VAAET ML 4.5.3

PostgreSQL se organiza en `vaaet_raw`, `vaaet_ml`, `vaaet_feedback` y
`vaaet_ops`. Este último registra el ciclo de cada workflow sin almacenar
credenciales ni mensajes sensibles; Alembic es la única autoridad DDL.

VAAET ML es el repositorio de machine learning para analizar el tránsito del Puente General Manuel Belgrano. Implementa adquisición de telemetría bajo demanda, entrenamiento batch e inferencia batch con feedback. La ejecución y promoción del modelo siguen siendo manuales en Google Colab; por eso es una base de MLOps Nivel 1, no un sistema de Continuous Training autónomo.

## Workflows

```text
video -> collection -> vaaet_raw.traffic_data
raw -> seed bootstrap -> immutable seed snapshot -> pilot bundle
inference review -> immutable HITL session package -> active catalog
seed snapshot + active HITL catalog -> input lock -> HITL retraining -> candidate bundle
validated feedback -> frozen validation/test holdout -> repeatable evaluation
video + bundle -> inference -> vaaet_ml.telemetry_features + traffic_predictions
predictions -> explicit HITL review -> vaaet_feedback.human_validations
```

- `notebooks/data-collection/collect_traffic_telemetry.ipynb`: adquisición opcional y reutilizable.
- `notebooks/training/train_traffic_state_classifier.ipynb`: 19 features, entrenamiento, evaluación y bundle.
- `notebooks/inference/analyze_traffic_video.ipynb`: video anotado, estado del tráfico y persistencia opcional.
- `notebooks/evaluation/evaluate_models_and_eda.ipynb`: auditoría read-only Champion--Challenger y drift de las 19 features sobre cohortes explícitas.
- `../vaaet-core/src/vaaet/`: lógica portable de percepción, telemetría,
  clasificación y bundles; los notebooks la importan como `vaaet`.
- `src/vaaet_ml/`: datos, entrenamiento, evaluación y runtime de laboratorio.
- `data/sample/`: ejemplos pequeños, anónimos y aptos para Git.

## Instalación local

VAAET ML admite Python 3.10–3.13.

Desde la raíz del monorepo, creá y activá una `.venv` aislada. Instalá primero
el core y luego el laboratorio, siempre con `python -m pip`:

```bash
python -m venv .venv
# Activá .venv con tu shell.
python -m pip install --upgrade pip
python -m pip install -e "./vaaet-core[vision,inference,dev]"
python -m pip install -e "./vaaet-ml[training,visualization,database,dev]"
python -m pip check
ruff check vaaet-ml/src vaaet-ml/tests vaaet-ml/scripts
pytest vaaet-ml/tests
```

Cada notebook tiene una sola celda de preparación: clona o actualiza
`https://github.com/zgfnicolas/vaaet` en `/content/vaaet`, resuelve
`CORE_ROOT=/content/vaaet/vaaet-core` y `ML_ROOT=/content/vaaet/vaaet-ml`,
instala ambos componentes con los extras del workflow y ejecuta el preflight
tipado. Los imports operativos usan `vaaet`; los de laboratorio, `vaaet_ml`.
No se usan `requirements.txt`, `sys.path` ni instaladores ad hoc.
Para administrar el bundle DVC desde una máquina local, instalá el extra
`./vaaet-ml[dvc]`; los notebooks no instalan DVC por separado.

## Contrato del modelo

El bundle v2 contiene `traffic_classifier.keras`, `feature_scaler.joblib`, `label_mapping.joblib` y `model-manifest.json`. El MLP aprende `Normal`, `Reduced` y `Congested`; una sospecha de incidente conserva `Congested` y sólo una confirmación humana validada puede publicar `Accident`. El manifiesto fija las 19 features, calibración, política temporal, modo de entrenamiento, política de entrada, etapa de despliegue, elegibilidad, procedencia, checksums y descriptor del input lock. Consultá el [contrato de artefactos](../docs/ml/model-artifact-contract.md), [ADR-0014](../docs/architecture/decisions/0014-hierarchical-traffic-state-and-incident-policy.md), [ADR-0017](../docs/architecture/decisions/0017-seed-bootstrap-and-hitl-retraining.md) y [ADR-0019](../docs/architecture/decisions/0019-immutable-seed-and-hitl-datasets.md).

Los aproximadamente 2.068 registros históricos permiten crear un bundle
`pilot` mediante weak supervision. Ese resultado reproduce reglas preliminares,
pero no acredita calidad real de producción: carece de telemetría v2 completa,
holdout humano y accidentes reales. Los reentrenamientos HITL consumen features
ya calculadas y sustituyen progresivamente la memoria proxy por etiquetas
humanas; cada artefacto sigue siendo candidato hasta cumplir los gates.

En HITL, `HUMAN_HOLDOUT_FROZEN=True` crea o reutiliza validation y test humanos
versionados bajo Google Drive. Los nuevos registros no alteran ese benchmark;
una actualización explícita genera otra fotografía y conserva la anterior. El
fingerprint del holdout queda registrado en el manifiesto del modelo, conforme
a [ADR-0018](../docs/architecture/decisions/0018-versioned-frozen-human-holdouts.md).

Después de entrenar un candidato HITL, el notebook de evaluación carga los dos
bundles sólo tras validar sus manifiestos y exige el ZIP exacto del holdout cuyo
fingerprint ambos declaran. Compara exclusivamente los tres estados estables,
aplica la cadena de serving de cada bundle y presenta diferencias emparejadas;
no modifica DVC, artefactos, PostgreSQL ni la decisión humana de promoción. Su
EDA de drift requiere cohortes `traffic-features-v2` y puede leer telemetría v2
acotada con el perfil `training` de sólo lectura.

La futura Web App pertenece a `../vaaet-app/` y consumirá exclusivamente una API
versionada que valide bundles antes de deserializarlos. La [documentación](../docs/index.md)
y la [guía de Colab](../docs/operations/colab-guide.md) describen el flujo completo.

## PostgreSQL e HITL

Una única base PostgreSQL 14+ usa `vaaet_raw`, `vaaet_ml`, `vaaet_feedback` y
`vaaet_ops`. Cada notebook carga un perfil de mínimo privilegio desde Colab
Secrets o variables locales; las URLs se construyen con SQLAlchemy y TLS
`verify-full` es el valor recomendado. Alembic y el rol administrador quedan fuera
de Colab. Aplicación inicial:

```bash
export VAAET_DATABASE_ADMIN_URL='postgresql+psycopg2://...'
alembic upgrade head
psql 'postgresql://admin:...@host:5432/vaaet' -f migrations/provision-roles.sql
```

El entrenamiento declara `TrainingMode.SEED_BOOTSTRAP` o
`TrainingMode.HITL_RETRAINING` mediante `TrainingIngestionPlan`. La semilla se
guarda como snapshot inmutable y cada sesión de revisión produce un paquete HITL
catalogado. Sólo `human_validations` efectivas ingresan como etiquetas;
predicciones sin revisar nunca se convierten en ground truth. Cada entrenamiento
escribe un `vaaet-training-input-lock-v1` con los fingerprints exactos utilizados.
Consultá [ADR-0015](../docs/architecture/decisions/0015-postgresql-namespaces-security-and-hitl.md),
[ADR-0016](../docs/architecture/decisions/0016-postgresql-hardening-and-pipeline-runs.md) y
[ADR-0019](../docs/architecture/decisions/0019-immutable-seed-and-hitl-datasets.md).
El provisionamiento, backup, rotación y recuperación están en la
[guía PostgreSQL](../docs/operations/postgresql-guide.md).

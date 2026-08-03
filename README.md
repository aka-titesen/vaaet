# VAAET ML 4.0.0

VAAET ML es el repositorio de machine learning para analizar el tránsito del Puente General Manuel Belgrano. Implementa adquisición de telemetría bajo demanda, entrenamiento batch e inferencia batch con feedback. La ejecución y promoción del modelo siguen siendo manuales en Google Colab; por eso es una base de MLOps Nivel 1, no un sistema de Continuous Training autónomo.

## Workflows

```text
video -> notebooks/data-collection -> data/raw/traffic_data_raw.csv
                                      -> traffic_data (PostgreSQL opcional)
raw telemetry -> notebooks/training -> artifacts/traffic-state (bundle DVC)
video + bundle -> notebooks/inference -> video anotado + clasificación + feedback
```

- `notebooks/data-collection/collect_traffic_telemetry.ipynb`: adquisición opcional y reutilizable.
- `notebooks/training/train_traffic_state_classifier.ipynb`: 19 features, entrenamiento, evaluación y bundle.
- `notebooks/inference/analyze_traffic_video.ipynb`: video anotado, estado del tráfico y persistencia opcional.
- `src/vaaet/`: lógica compartida instalable; los notebooks sólo orquestan.
- `data/sample/`: ejemplos pequeños, anónimos y aptos para Git.

## Instalación local

```bash
python -m pip install -e ".[vision,training,visualization,database,dev]"
python -m pip check
ruff check src tests scripts
pytest
```

Cada notebook tiene una sola celda de preparación: clona o actualiza `https://github.com/zgfnicolas/vaaet` en `/content/vaaet`, resuelve `REPO_ROOT`, instala sus extras y muestra el diagnóstico de `pip check` sin bloquear por inconsistencias preexistentes de Colab. Los imports del workflow siguen siendo obligatorios y CI mantiene `pip check` estricto. No se usan `requirements.txt`, `sys.path` ni instaladores ad hoc.

## Contrato del modelo

El bundle v2 contiene `traffic_classifier.keras`, `feature_scaler.joblib`, `label_mapping.joblib` y `model-manifest.json`. El MLP aprende `Normal`, `Reduced` y `Congested`; una sospecha de incidente conserva `Congested` y sólo una confirmación humana validada puede publicar `Accident`. El manifiesto fija las 19 features, calibración, política temporal, elegibilidad, bloqueos, procedencia y checksums. Consultá el [contrato de artefactos](docs/ml/model-artifact-contract.md) y [ADR-0014](docs/architecture/decisions/0014-hierarchical-traffic-state-and-incident-policy.md).

Los aproximadamente 2.068 registros históricos permiten reproducir un baseline, pero no acreditan un modelo de producción: carecen de telemetría v2 completa, holdout humano y accidentes reales. El notebook lo refleja marcando el bundle como `experimental/shadow-only` hasta cumplir los gates.

La futura Web App pertenece a otro repositorio y consumirá únicamente bundles validados. La [documentación](docs/index.md) y la [guía de Colab](docs/operations/colab-guide.md) describen el flujo completo.

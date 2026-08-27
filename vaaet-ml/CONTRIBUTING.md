# Guía de contribución — VAAET ML 4.5.3

Antes de modificar el proyecto, leé [AGENTS.md](AGENTS.md) y el ADR aplicable en
[`../docs/architecture/decisions/`](../docs/architecture/decisions/).

## Reglas fundamentales

- La lógica portable vive en `../vaaet-core/src/vaaet/`; los notebooks sólo orquestan.
- Los imports de laboratorio usan `vaaet_ml.*`; las operaciones usan `vaaet.*`.
- `../vaaet-core/src/vaaet/settings.py` define contratos y umbrales; este
  componente define rutas de laboratorio y configuración DB en `src/vaaet_ml/settings.py`.
- Los notebooks instalan primero el core y luego ML.
- Las 19 features, cuatro estados, esquema PostgreSQL y arquitectura MLP son
  contratos; cualquier cambio requiere aprobación y un ADR nuevo.
- Los pesos YOLO y binarios `.keras`/`.joblib` no se versionan con Git. El bundle
  completo se registra como unidad mediante DVC.
- Código, APIs, nombres y comentarios técnicos se escriben en inglés; la
  documentación explicativa se mantiene en español.

## Entorno y calidad

Python soportado: 3.10–3.13.

```bash
python -m pip install -e "../vaaet-core[vision,inference,dev]"
python -m pip install -e ".[training,visualization,database,dev]"
ruff check src tests scripts
pytest tests/ -v --tb=short
python -m compileall -q src tests scripts
git diff --check
```

También deben compilar todas las celdas de los notebooks y resolver los enlaces
Markdown activos. La ejecución end-to-end con GPU, Drive, PostgreSQL y DVC
remoto se valida manualmente en Google Colab.

## Cambios por área

- Entrenamiento: `notebooks/training/train_traffic_state_classifier.ipynb` y
  [ADR-0008](../docs/architecture/decisions/0008-keras-traffic-state-classifier.md).
- Inferencia: `notebooks/inference/analyze_traffic_video.ipynb` y
  [ADR-0009](../docs/architecture/decisions/0009-modular-three-stage-architecture.md).
- Bundle y límite core/API: [ADR-0021](../docs/architecture/decisions/0021-portable-core-and-ml-laboratory-boundary.md).
- DVC: [guía DVC](../docs/ml/dvc-guide.md).

Actualizá [CHANGELOG.md](CHANGELOG.md) y la documentación cuando cambie un
comportamiento observable. Nunca incluyas secretos ni datos sensibles.

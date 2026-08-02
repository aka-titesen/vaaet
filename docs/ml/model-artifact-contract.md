# Contrato del bundle de modelo

El límite entre VAAET ML y el futuro repositorio web es un directorio portable
versionado como unidad con DVC: `artifacts/traffic-state/`.

## Contenido obligatorio

| Archivo | Función |
|---|---|
| `traffic_classifier.keras` | Clasificador MLP de estados del tráfico |
| `feature_scaler.joblib` | `StandardScaler` ajustado con el training set |
| `label_mapping.joblib` | Mapping de los cuatro códigos canónicos |
| `model-manifest.json` | Compatibilidad, integridad y procedencia |

El manifiesto contiene versión del contrato, versión del modelo, versión del
schema, orden exacto de las 19 features, clases, timestamp UTC, commit Git,
dependencias, métricas, origen del dataset, presencia de datos sintéticos y
checksums SHA-256 de los tres binarios.

Todo consumidor debe ejecutar `vaaet.artifacts.validate_manifest()` después de
obtener el bundle —sea local, Drive, upload o registry— y antes de cargar Keras
o joblib. Un campo ausente, JSON inválido, versión no soportada, schema/mapping
incompatible, archivo faltante o checksum alterado rechaza el bundle con un
error de dominio explícito.

## Versionado con DVC

Después del primer entrenamiento se elimina `.gitkeep` y se registra el
directorio completo:

```bash
dvc add artifacts/traffic-state
git add artifacts/traffic-state.dvc .gitignore
dvc push
```

La infraestructura de publicación y descarga para la Web App queda fuera de
este repositorio. Véase [ADR-0012](../architecture/decisions/0012-ml-web-boundary-and-artifact-contract.md).

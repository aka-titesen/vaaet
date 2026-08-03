# Contrato del bundle de modelo

El límite entre VAAET ML y el futuro repositorio web es un directorio portable
versionado como unidad con DVC: `artifacts/traffic-state/`.

## Contenido obligatorio

| Archivo | Función |
|---|---|
| `traffic_classifier.keras` | MLP de tres estados estables |
| `feature_scaler.joblib` | `StandardScaler` ajustado con el training set |
| `label_mapping.joblib` | Mapping de los cuatro códigos canónicos |
| `model-manifest.json` | Compatibilidad, integridad y procedencia |

El contrato v2 distingue `model_output_mapping` —Normal, Reduced y Congested—
de los cuatro estados públicos. También contiene la política jerárquica,
temperatura de calibración, umbrales por clase, margen, histéresis, prohibición de Accident automático,
confirmación humana obligatoria, elegibilidad, bloqueos de promoción y
checksums SHA-256.

Todo consumidor debe ejecutar `vaaet.artifacts.validate_manifest()` después de
obtener el bundle —sea local, Drive, upload o registry— y antes de cargar Keras
o joblib. Un campo ausente, JSON inválido, versión no soportada, schema/mapping
incompatible, archivo faltante o checksum alterado rechaza el bundle con un
error de dominio explícito.

`production_eligible=false` no invalida la integridad del bundle, pero obliga a
tratarlo como artefacto experimental/shadow-only. El notebook de inferencia lo
rechaza por defecto y sólo permite habilitarlo explícitamente para evaluación
offline.

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
La semántica jerárquica está gobernada por [ADR-0014](../architecture/decisions/0014-hierarchical-traffic-state-and-incident-policy.md).

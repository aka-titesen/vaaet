# Contrato del bundle de modelo

El límite entre VAAET ML y el futuro repositorio web es un directorio portable
versionado como unidad con DVC: `vaaet-ml/artifacts/traffic-state/`.

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

`training_lifecycle` vuelve explícitos cuatro datos de serving:

- `training_mode`: `seed-bootstrap` o `hitl-retraining`.
- `supervision`: `weak-proxy` o `human-validated`.
- `input_policy`: `legacy-v1-bootstrap` o `canonical-v2`.
- `deployment_stage`: `pilot`, `candidate` o `production`.

Desde VAAET 4.4.0, un modelo que declara `data_provenance.human_holdout=true`
debe incluir también `human_holdout`: contrato `vaaet-human-holdout-v1`, UUID del
snapshot, generación, fingerprint SHA-256 y filas de validation/test. Un modelo
sin benchmark congelado utiliza `null`. El descriptor no incorpora el dataset;
sólo permite auditar con qué fotografía humana se midió el candidato y rechazar
comparaciones automáticas entre fingerprints distintos.

Desde VAAET 4.5.0, `training_input_lock` puede incorporar el descriptor
`vaaet-training-input-lock-v1`: UUID del lock y fingerprint SHA-256 de la selección
exacta de semilla, catálogo HITL y holdout. El JSON completo permanece en
`MyDrive/vaaet-ml/training-runs/<run-id>/`; el bundle conserva sólo su descriptor
y continúa teniendo cuatro archivos.

Un bundle semilla siempre es `pilot` y `production_eligible=false`. La política
legacy neutraliza las tres evidencias de calidad desconocidas tanto durante el
entrenamiento como durante la inferencia, evitando una divergencia train/serve.

Todo consumidor debe ejecutar `vaaet.artifacts.validate_manifest()` después de
obtener el bundle —sea local, Drive, upload o registry— y antes de cargar Keras
o joblib. Un campo ausente, JSON inválido, versión no soportada, schema/mapping
incompatible, archivo faltante o checksum alterado rechaza el bundle con un
error de dominio explícito.

`production_eligible=false` no invalida la integridad del bundle. Inferencia
permite un `pilot` sólo mediante `ALLOW_PILOT_BUNDLE=True`, lo identifica en el
output y conserva sus decisiones conservadoras. Los candidatos HITL no aprobados
requieren la autorización experimental separada; ningún flag cambia la metadata
ni promociona el artefacto.

## Versionado con DVC

Después del primer entrenamiento se elimina `.gitkeep` y se registra el
directorio completo:

```bash
dvc add vaaet-ml/artifacts/traffic-state
git add vaaet-ml/artifacts/traffic-state.dvc .gitignore
dvc push
```

La infraestructura de publicación y descarga para la Web App queda fuera de
este repositorio. Véase [ADR-0012](../architecture/decisions/0012-ml-web-boundary-and-artifact-contract.md).
La semántica jerárquica está gobernada por [ADR-0014](../architecture/decisions/0014-hierarchical-traffic-state-and-incident-policy.md).
El ciclo semilla/HITL está gobernado por [ADR-0017](../architecture/decisions/0017-seed-bootstrap-and-hitl-retraining.md).
El benchmark humano congelado está gobernado por [ADR-0018](../architecture/decisions/0018-versioned-frozen-human-holdouts.md).
La gestión inmutable de datasets está gobernada por [ADR-0019](../architecture/decisions/0019-immutable-seed-and-hitl-datasets.md).

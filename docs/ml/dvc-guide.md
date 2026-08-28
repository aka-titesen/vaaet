# Guía DVC — VAAET ML 4.5.3

DVC versiona el bundle promocionable sin guardar binarios pesados en Git. El
bundle se trata como una unidad indivisible porque modelo, scaler, mapping y
manifiesto deben pertenecer al mismo entrenamiento.

El baseline v1 permanece recuperable en el historial/DVC, pero no es compatible
con el runtime v2. Un bundle con `production_eligible=false` puede conservarse
como experimento, aunque no debe promoverse al canal consumido por la Web App.

## Bundle requerido

```text
vaaet-ml/artifacts/traffic-state/
├── traffic_classifier.keras
├── feature_scaler.joblib
├── label_mapping.joblib
└── model-manifest.json
```

Mientras todavía no exista un entrenamiento materializado, `.gitkeep` conserva
el directorio. Después del primer entrenamiento debe eliminarse y ser reemplazado
por `vaaet-ml/artifacts/traffic-state.dvc`.

## Registrar y publicar

```bash
rm vaaet-ml/artifacts/traffic-state/.gitkeep
dvc add vaaet-ml/artifacts/traffic-state
git add vaaet-ml/artifacts/traffic-state.dvc .gitignore
git commit -m "feat(models): registrar bundle de tráfico"
dvc push
```

No ejecutes `dvc add` por archivo: permitiría combinar artefactos incompatibles.
El manifiesto debe validarse antes de registrar el directorio.

## Descargar y verificar

```bash
dvc pull vaaet-ml/artifacts/traffic-state.dvc
dvc status
dvc doctor
```

```python
from vaaet.artifacts import validate_manifest

manifest = validate_manifest("vaaet-ml/artifacts/traffic-state")
print(manifest["model_version"])
```

## Google Colab

Los notebooks copian siempre los cuatro archivos y no instalan DVC por separado.
La administración de DVC se realiza localmente desde el workspace:

```bash
python -m pip install -e "./vaaet-core"
python -m pip install -e "./vaaet-ml[dvc]"
python -m pip check
dvc pull vaaet-ml/artifacts/traffic-state.dvc
```

Google Drive es el remote predeterminado; S3 y local son alternativas declaradas
en `.dvc/config`. Credenciales y opciones privadas pertenecen a
`.dvc/config.local`, nunca a Git. Para inicializar una máquina ejecutá
`bash vaaet-ml/scripts/setup-dvc.sh` desde la raíz.

## Diagnóstico

| Problema | Acción |
|---|---|
| Falta `model-manifest.json` | Reejecutar la exportación de entrenamiento; no publicar |
| Checksum incompatible | Rechazar el bundle y recuperar una versión coherente |
| `dvc push` falla | Revisar autenticación y `dvc remote list` |
| Cache grande | Usar `dvc gc --workspace` con precaución |

Referencias: [contrato del bundle](model-artifact-contract.md), [ADR-0011](../architecture/decisions/0011-dvc-model-registry.md), [ADR-0012](../architecture/decisions/0012-ml-web-boundary-and-artifact-contract.md) como antecedente histórico y [ADR-0021](../architecture/decisions/0021-portable-core-and-ml-laboratory-boundary.md) para la frontera vigente.

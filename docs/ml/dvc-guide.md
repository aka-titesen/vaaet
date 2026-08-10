# Guía DVC — VAAET ML 4.5.0

DVC versiona el bundle promocionable sin guardar binarios pesados en Git. El
bundle se trata como una unidad indivisible porque modelo, scaler, mapping y
manifiesto deben pertenecer al mismo entrenamiento.

El baseline v1 permanece recuperable en el historial/DVC, pero no es compatible
con el runtime v2. Un bundle con `production_eligible=false` puede conservarse
como experimento, aunque no debe promoverse al canal consumido por la Web App.

## Bundle requerido

```text
artifacts/traffic-state/
├── traffic_classifier.keras
├── feature_scaler.joblib
├── label_mapping.joblib
└── model-manifest.json
```

Mientras todavía no exista un entrenamiento materializado, `.gitkeep` conserva
el directorio. Después del primer entrenamiento debe eliminarse y ser reemplazado
por `artifacts/traffic-state.dvc`.

## Registrar y publicar

```bash
rm artifacts/traffic-state/.gitkeep
dvc add artifacts/traffic-state
git add artifacts/traffic-state.dvc .gitignore
git commit -m "feat(models): registrar bundle de tráfico"
dvc push
```

No ejecutes `dvc add` por archivo: permitiría combinar artefactos incompatibles.
El manifiesto debe validarse antes de registrar el directorio.

## Descargar y verificar

```bash
dvc pull artifacts/traffic-state.dvc
dvc status
dvc doctor
```

```python
from vaaet.artifacts import validate_manifest

manifest = validate_manifest("artifacts/traffic-state")
print(manifest["model_version"])
```

## Google Colab

Los notebooks copian siempre los cuatro archivos. Si se usa DVC directamente:

```python
!pip install -q "dvc[gdrive]>=3.50.0"
!dvc pull artifacts/traffic-state.dvc
```

Google Drive es el remote predeterminado; S3 y local son alternativas declaradas
en `.dvc/config`. Credenciales y opciones privadas pertenecen a
`.dvc/config.local`, nunca a Git. Para inicializar una máquina ejecutá
`bash scripts/setup-dvc.sh` desde la raíz.

## Diagnóstico

| Problema | Acción |
|---|---|
| Falta `model-manifest.json` | Reejecutar la exportación de entrenamiento; no publicar |
| Checksum incompatible | Rechazar el bundle y recuperar una versión coherente |
| `dvc push` falla | Revisar autenticación y `dvc remote list` |
| Cache grande | Usar `dvc gc --workspace` con precaución |

Referencias: [contrato del bundle](model-artifact-contract.md), [ADR-0011](../architecture/decisions/0011-dvc-model-registry.md), [ADR-0012](../architecture/decisions/0012-ml-web-boundary-and-artifact-contract.md).

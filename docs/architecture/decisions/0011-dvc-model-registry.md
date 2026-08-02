<!-- context: VAAET/docs/architecture/decisions/0011-dvc-model-registry.md
Referenciado por AGENTS.md, docs/ml/dvc-guide.md y ADR-0012. -->

# ADR-011: DVC como Model Registry

| Campo | Valor |
|---|---|
| **Estado** | Aceptado |
| **Fecha** | 2026-07-30 |
| **Decisores** | Facundo Nicolás González |
| **Supersede** | — (nueva decisión) |
| **Referenciado por** | AGENTS.md, DVC_GUIDE.md, PROJECT_PLAN.md |

## Contexto

El pipeline VAAET genera artefactos binarios pesados que Git no puede versionar eficientemente:

| Artefacto | Formato | Tamaño Estimado |
|---|---|---|
| Clasificador MLP | `.keras` | ~500 KB – 2 MB |
| Feature scaler | `.joblib` | ~10 KB |
| Label mapping | `.joblib` | ~5 KB |
| Manifiesto del bundle | `.json` | ~5 KB |
| Modelos YOLO | `.pt` | ~50 MB (descargados, no versionados) |
| Backups de BD | `.csv` | ~500 KB – 5 MB |

Estos artefactos estaban en `.gitignore` sin ningún sistema de versionado. Al re-entrenar el modelo, la versión anterior se perdía.

## Opciones Evaluadas

### Opción A: Google Drive + Naming Convention

- **Ventaja**: Cero setup, nativo con Colab
- **Desventaja**: Sin historial automático, sin integración con Git, sin reproducibilidad
- **Veredicto**: Insuficiente para MLOps Nivel 1

### Opción B: DVC (Data Version Control) ✅

- **Ventaja**: Git-like para datos, integración nativa con Git, múltiples backends, gratuito
- **Desventaja**: Dependencia adicional, curva de aprendizaje mínima
- **Veredicto**: Estándar de la industria en 2026 para proyectos de ML

### Opción C: MLflow

- **Ventaja**: UI web, experiment tracking, model registry completo
- **Desventaja**: Requiere servidor corriendo, excesivo para 1 desarrollador + Colab
- **Veredicto**: Sobreingeniería para el alcance actual

### Opción D: W&B (Weights & Biases)

- **Ventaja**: UI excelente, integración con Keras
- **Desventaja**: Plataforma propietaria, datos en servidores de terceros
- **Veredicto**: Conflicto con principios de privacidad (datos viales gubernamentales)

## Decisión

**Adoptar DVC con Google Drive como storage remoto por defecto**, con remotes alternativos pre-configurados (S3, local) para permitir migración sin fricción.

El directorio `artifacts/traffic-state/` se registra como una única unidad DVC;
no se crean pointers separados por binario. El bundle incluye obligatoriamente
los tres binarios y `model-manifest.json`, conforme a ADR-0012.

### Configuración Adoptada

```
[core]
    remote = gdrive
['remote "gdrive"']
    url = gdrive://VAAET-DVC-Storage
['remote "s3"']
    url = s3://vaaet-model-registry
['remote "local"']
    url = /tmp/vaaet-dvc-local
```

## Consecuencias

### Positivas

1. Cada commit de Git tiene asociado un bundle completo y específico (reproducibilidad)
2. `dvc pull` en Colab descarga exactamente los artefactos correctos
3. Cambiar de Google Drive a S3 requiere una sola línea (`dvc remote default s3`)
4. Compatible con el pipeline CI/CD existente (GitHub Actions)
5. Sin costo adicional (DVC es open source, Google Drive es gratuito)

### Negativas

1. Dependencia adicional (`dvc[gdrive]`) en el entorno de desarrollo
2. Paso extra en el flujo de trabajo: `dvc add` + `dvc push` después de entrenar
3. Primera autenticación OAuth con Google Drive requiere interacción manual

### Neutrales

1. `artifacts/traffic-state.dvc` se commitea en Git como metadata liviana del directorio
2. El cache local de DVC puede crecer; se limpia con `dvc gc --workspace`

## Referencias

- [DVC Documentation](https://dvc.org/doc)
- [DVC con Google Drive](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive)
- [ADR-0009: Arquitectura modular de tres módulos](0009-modular-three-stage-architecture.md)
- [ADR-0010: Pipeline MLOps 19 features](0010-mlops-pipeline-19-features.md)
- [ADR-0012: Límite ML/Web y bundle portable](0012-ml-web-boundary-and-artifact-contract.md)

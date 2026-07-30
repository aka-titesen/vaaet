# ADR-010: Pipeline MLOps con 19 Features y Señales de Calidad

**Estado:** Aceptado
**Fecha:** 2026-07-23
**Complementa:** ADR-008 (TF/Keras), ADR-009 (Arquitectura modular)
**Decisores:** Facundo Nicolás González

## Contexto

El clasificador MLP fue diseñado originalmente con 14 features (ADR-008, columna Input(14,)). Durante la evolución del pipeline de percepción en el Módulo 2, se identificó que las señales de calidad de medición de velocidad, el movimiento cercano a cero, y los estacionarios confirmados aportan información valiosa para la clasificación del tráfico.

Simultáneamente, el pipeline se formalizó como un sistema CT/CI (Continuous Training / Continuous Inference) de MLOps Nivel 1, requiriendo documentación explícita de:
- El contrato canónico de features
- La estrategia de proveniencia de datos (real vs sintético)
- El versionado de artefactos del modelo
- Los contratos de datos tipados

## Decisión

### 1. Ampliar las features canónicas de 14 a 19

Las 5 nuevas features son señales de calidad de la percepción:

| Feature | Fuente | Justificación |
|---|---|---|
| `cumulative_delta_speed` | Ventana rolling sobre `delta_speed` | Captura tendencias de desaceleración sostenida |
| `low_speed_persistence` | Ventana rolling sobre `avg_speed < umbral` | Distingue congestión transitoria de sostenida |
| `speed_measurement_quality` | Pipeline de percepción | Ratio de muestras aceptadas/intentadas |
| `near_zero_motion_ratio` | Pipeline de percepción | Porcentaje de tracks con movimiento casi nulo |
| `stationary_confirmed_ratio` | Pipeline de percepción | Porcentaje de tracks confirmados como estacionarios |

### 2. Implementar contratos de datos tipados

`src/contracts.py` define dataclasses frozen con validación en `__post_init__`:
- `TelemetryRecord`: Registro crudo validado
- `EngineeredTelemetryRecord`: Registro con las 19 features validadas
- `ClassificationRecord`: Resultado de clasificación validado
- `TrackSpeedState`: Estado de velocidad por track

### 3. Implementar proveniencia de datos

Columnas `data_origin` ("real"/"synthetic") y `synthetic_scenario` ("observed"/"accident"/"congestion") permiten:
- Rastrear el origen de cada registro en la BD
- Excluir datos sintéticos de evaluaciones de ground truth
- Mantener trazabilidad end-to-end

### 4. Versionado de modelo

`MODEL_VERSION = "mlp-v1.1"` en `src/config.py` identifica la versión del modelo que generó cada clasificación. La tabla `traffic_classifications` tiene constraint UNIQUE(telemetry_id, model_version) para soportar múltiples versiones.

## Justificación

1. Las señales de calidad de percepción reducen falsos positivos de accidente (donde `speed_measurement_quality` baja puede indicar problemas de cámara, no accidente real)
2. Los contratos de datos previenen drift silencioso entre módulos
3. La proveniencia permite separar datos reales de sintéticos en evaluaciones futuras
4. El versionado de modelo es requisito mínimo para un pipeline MLOps operativo

## Consecuencias

### Positivas
- El clasificador tiene acceso a señales de calidad que antes se descartaban
- Los contratos de datos detectan errores en tiempo de construcción, no en runtime
- La proveniencia habilita evaluaciones de rendimiento segregadas por tipo de datos
- El versionado permite comparar modelos side-by-side

### Negativas
- Los artefactos de modelo de la versión anterior (14 features) son incompatibles y deben regenerarse
- La documentación existente que referenciaba "14 features" requiere actualización (SAD.md, ADR-008)
- La complejidad del esquema de BD aumenta (27 columnas en `telemetry_raw`)

### Deuda técnica
- No hay Model Registry formal (los artefactos se almacenan en Google Drive o localmente)
- No hay experiment tracking (MLflow/W&B)
- No hay monitoreo de data drift en producción

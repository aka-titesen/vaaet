<!-- context: VAAET/Docs/adr/ADR-008 — Decisión de usar TensorFlow/Keras para clasificación de estado de tráfico.
Referenciado por AGENTS.md, DDS.md §8, README.md, notebook 02_traffic_state_classifier.ipynb.
Introduce la Etapa 2 (Inteligencia) del pipeline VAAET. -->

# ADR-008: TensorFlow/Keras para Clasificación de Estado de Tráfico

**Estado:** Aceptado  
**Fecha:** 2026-03-07  
**Decisores:** Equipo VAAET

## Contexto

La Etapa 1 (Percepción) de VAAET produce telemetría cruda cada minuto — 9 campos (velocidad promedio, conteos por tipo de vehículo, total) que se persisten en la tabla `traffic_data` de PostgreSQL. Sin embargo, estos datos crudos no responden la pregunta operacional clave: **¿cuál es el estado actual del tráfico en el puente?**

Se necesita una capa de inteligencia que transforme la telemetría en una clasificación de 4 estados operacionales:

| Estado | Código | Semántica (calibrada al puente Belgrano) |
|---|---|---|
| Normal | 0 | Flujo libre, >25 km/h o volumen <5 veh/min |
| Reducido | 1 | Flujo degradado, 7–25 km/h, 5–12 veh/min |
| Atascado | 2 | Congestión, <7 km/h, >8 veh/min, persistencia ≥2 min |
| Accidente | 3 | Evento disruptivo, <2 km/h, delta_speed < −15, persistencia ≥2 min |

> **Nota (2026-03-11)**: Los umbrales fueron recalibrados desde valores genéricos de ingeniería de tránsito a percentiles del dataset real del puente (P25 speed ≈ 7.78 km/h, mediana vehicles ≈ 3, P75 ≈ 6). El dataset real solo contenía estados Normal y Reducido; secuencias sintéticas de Congestión y Accidente se generan en `src/synthetic.py` para completar las 4 clases.

Se evaluaron las siguientes alternativas:

1. **scikit-learn `MLPClassifier`**: Simple, liviano, ya presente en el proyecto (Etapa 1 usa `MLPRegressor`)
2. **TensorFlow/Keras `Sequential`**: Framework de deep learning completo, preinstalado en Colab
3. **XGBoost**: Gradient boosting, excelente para datos tabulares
4. **Reglas If/Else sin ML**: Determinístico, sin entrenamiento

## Decisión

Se adopta **TensorFlow/Keras** con un modelo `Sequential` MLP como clasificador de Fase 1, diseñado para evolucionar a LSTM en Fase 2.

### Model Architecture (Phase 1 — Tabular MLP)

```
Input(shape=(14,))
  → Dense(64, relu) → BatchNormalization → Dropout(0.3)
  → Dense(32, relu) → BatchNormalization → Dropout(0.2)
  → Dense(n_classes, softmax)
```

### Pipeline de datos

1. **Fuente**: Telemetría real de `traffic_data` (~2000 registros del backup `pg_dump`)
2. **Feature engineering**: 9 campos crudos → 14 features (ratios, deltas, varianza, temporales)
3. **Auto-labeling**: Reglas de ingeniería de tránsito asignan estados como proxy de ground truth
4. **Balanceo**: SMOTE sobre el training set para compensar desbalance (~80% Normal)
5. **Entrenamiento**: EarlyStopping + ReduceLROnPlateau, seed=42

### Persistencia

Dos tablas nuevas con FK, separadas de `traffic_data` (legacy intocable):

- **`telemetry_raw`**: 14 features engineered + FK a `traffic_data(id)`
- **`traffic_classifications`**: Predicción del modelo + campos HITL para validación futura

### HITL (Human-in-the-Loop)

Diseñado en el esquema (`is_human_validated`, `human_override_state`, `validated_at`) pero **implementado en Fase 2 futura**. En Fase 1, las clasificaciones son automáticas con auto-labeling como base.

## Razonamiento

1. **Preinstalado en Colab**: TensorFlow viene disponible en Google Colab sin instalación adicional, lo que elimina fricción de setup y mantiene compatibilidad con el runtime principal del proyecto (ADR-007)

2. **Evolución a LSTM sin reescritura**: La API `Sequential` de Keras permite agregar capas `LSTM` o `GRU` en el futuro cambiando solo la definición del modelo. scikit-learn no tiene capas recurrentes nativas

3. **Persistencia temporal**: Los estados Reducido, Atascado y Accidente tienen requisitos de persistencia (>60s, >120s, >180s respectivamente). Un MLP no modela secuencias, pero un LSTM sí — y la migración es trivial con Keras. Con sklearn habría que reescribir todo al agregar temporalidad

4. **Callbacks sofisticados**: `EarlyStopping(restore_best_weights=True)` y `ReduceLROnPlateau` permiten control fino del entrenamiento que sklearn no ofrece nativamente

5. **Exportación autónoma**: El formato `.keras` es autodescriptivo y portable. No depende de `pickle` (que tiene vulnerabilidades de seguridad) como sklearn

6. **XGBoost descartado**: Excelente para datos tabulares estáticos, pero no evolucionable a modelos secuenciales. Agrega una dependencia externa que no viene preinstalada en Colab

## Consecuencias

### Positivas

- Modelo MLP funcional desde Fase 1, evolucionable a LSTM en Fase 2
- Callbacks `EarlyStopping` + `ReduceLROnPlateau` previenen overfitting automáticamente
- TensorFlow preinstalado en Colab: cero fricción de setup
- Esquema de BD diseñado para HITL desde el inicio (campos presentes, lógica futura)
- Dos tablas con FK mantienen trazabilidad completa: telemetría cruda → clasificación → validación humana
- Auto-labeling con reglas de ingeniería permite entrenamiento sin anotación manual costosa

### Negativas

- **TensorFlow es heavy** (~450MB): import más lento que sklearn, uso de memoria mayor
- **Overkill para tabular puro**: Un MLP de 2 capas no necesita la maquinaria de TF; sklearn sería suficiente para Fase 1 aislada
- **Auto-labeling no es ground truth**: Los umbrales de ingeniería son aproximaciones; pueden haber clasificaciones erróneas que solo HITL futuro corregirá
- **SMOTE genera muestras sintéticas**: Especialmente para la clase Accidente (extremadamente rara), las muestras sintéticas pueden no ser realistas

### Deuda técnica aceptada

- MLP Fase 1 sin temporalidad — el modelo no aprende secuencias, solo snapshots por minuto
- Auto-labeling como proxy de ground truth — aceptable hasta que HITL esté operativo
- HITL diseñado pero no implementado — campos en la tabla, sin interfaz de operador
- `weather_condition` simulado por hora del día — no es dato meteorológico real
- Sin connection pooling para las nuevas tablas (hereda patrón de Etapa 1, ADR-005)

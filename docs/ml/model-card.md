<!-- context: VAAET/docs/MODEL_CARD.md — Model Card del clasificador MLP.
Formato HuggingFace completo. Complementa SAD.md y BIAS_AND_LIMITATIONS.md. -->

# Model Card — Clasificador MLP de Estados del Tráfico

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 3.0.0 |
| **Estado** | Aprobado |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

## Información del Modelo

| Campo | Valor |
|---|---|
| **Nombre** | VAAET Traffic State Classifier |
| **Versión** | mlp-v1.1 |
| **Tipo** | Multi-Layer Perceptron (MLP) — Red neuronal tabular |
| **Framework** | TensorFlow 2.15+ / Keras |
| **Formato de artefacto** | `.keras` (modelo) + `.joblib` (scaler, label mapping) |
| **Licencia** | MIT |
| **Desarrollador** | Facundo Nicolás González |
| **Fecha de entrenamiento** | Pendiente primera ejecución del Módulo 1 |
| **Repositorio** | [github.com/zgfnicolas/vaaet](https://github.com/zgfnicolas/vaaet) |

---

## Uso Previsto

### Usuarios Previstos

- **Operadores SISE**: Monitoreo del estado del tráfico en el Puente Gral. Manuel Belgrano
- **Investigadores de tráfico**: Análisis académico de patrones vehiculares
- **Ingenieros de tráfico**: Planificación urbana basada en datos

### Casos de Uso Previstos

- Clasificación del estado del tráfico minuto a minuto a partir de telemetría de video procesado
- Alerta temprana de condiciones anómalas (congestión, posible accidente)
- Análisis retrospectivo de patrones de tráfico

### Casos de Uso Fuera de Alcance

- **No usar para**: Decisiones de emergencia en tiempo real sin validación humana
- **No usar para**: Otros puentes, autopistas o contextos urbanos sin re-entrenamiento
- **No usar para**: Clasificación de vehículos individuales (el modelo opera sobre agregados por minuto)

---

## Arquitectura del Modelo

```
Input(19 features)
  → Dense(64, ReLU) → BatchNormalization → Dropout(0.3)
  → Dense(32, ReLU) → BatchNormalization → Dropout(0.2)
  → Dense(4, Softmax)
```

| Hiperparámetro | Valor |
|---|---|
| **Optimizador** | Adam |
| **Loss** | Sparse Categorical Crossentropy |
| **Callbacks** | EarlyStopping(patience=15), ReduceLROnPlateau(patience=5, factor=0.5) |
| **Semilla** | 42 (reproducibilidad) |
| **Partición** | 80/20 estratificada por clase |
| **Balanceo** | SMOTE en conjunto de entrenamiento (k_neighbors adaptativo) |
| **Escalado** | StandardScaler (fit en train, transform en ambos) |

---

## Features de Entrada (19)

| # | Feature | Tipo | Fuente | Descripción |
|---|---|---|---|---|
| 1 | `avg_speed` | NUMERIC | Directa | Velocidad promedio de vehículos en movimiento (km/h) |
| 2 | `total_vehicles` | INTEGER | Directa | Conteo total de vehículos en el minuto |
| 3 | `count_car` | INTEGER | Directa | Autos detectados |
| 4 | `count_truck` | INTEGER | Directa | Camiones detectados |
| 5 | `count_bus` | INTEGER | Directa | Colectivos detectados |
| 6 | `count_motorcycle` | INTEGER | Directa | Motocicletas detectadas |
| 7 | `count_bicycle` | INTEGER | Directa | Bicicletas detectadas |
| 8 | `heavy_vehicle_ratio` | NUMERIC | Derivada | `(truck+bus) / total.clip(1)` |
| 9 | `delta_speed` | NUMERIC | Derivada | `avg_speed.diff()` — aceleración/desaceleración |
| 10 | `delta_count` | INTEGER | Derivada | `total_vehicles.diff()` — tasa de cambio de volumen |
| 11 | `transition_flag` | BINARY | Derivada | Cambio abrupto simultáneo de velocidad y conteo |
| 12 | `speed_variance` | NUMERIC | Derivada | `avg_speed.rolling(5).std()` — estabilidad del flujo |
| 13 | `cumulative_delta_speed` | NUMERIC | Derivada | Delta de velocidad acumulado en ventana rolling |
| 14 | `low_speed_persistence` | NUMERIC | Derivada | Persistencia de baja velocidad |
| 15 | `speed_measurement_quality` | NUMERIC | Calidad | Ratio muestras aceptadas / intentadas |
| 16 | `near_zero_motion_ratio` | NUMERIC | Calidad | Ratio de tracks con movimiento cercano a cero |
| 17 | `stationary_confirmed_ratio` | NUMERIC | Calidad | Ratio de tracks confirmados como estacionarios |
| 18 | `hour_of_day` | INTEGER | Temporal | Hora del día (0-23) |
| 19 | `weather_condition` | BINARY | Proxy | Proxy por hora: 0=diurno (6-18h), 1=nocturno |

---

## Datos de Entrenamiento

### Fuente

| Atributo | Valor |
|---|---|
| **Origen** | Telemetría real del Puente Gral. Manuel Belgrano + secuencias sintéticas |
| **Período real** | Abril–Julio 2025 |
| **Registros reales** | ~2.000 (1 registro/minuto) |
| **Registros sintéticos** | ~200 (accidente + congestión) |
| **Etiquetado** | Auto-etiquetado con reglas de ingeniería (NO ground truth humano) |
| **Balanceo** | SMOTE post-inyección sintética |

### Distribución de Clases

| Clase | Datos Reales | Sintéticos | Frecuencia Combinada |
|---|---|---|---|
| Normal (0) | ~2.004 | 0 | ~75% |
| Reducido (1) | ~63 | 0 | ~3% |
| Congestionado (2) | 0 | ~100 | ~4% |
| Accidente (3) | 0 | ~100 | ~4% |

**Nota crítica:** Las clases Congestionado y Accidente son **100% sintéticas**. El modelo no ha sido validado con eventos reales de estas clases.

### Pre-procesamiento

1. Inyección de secuencias sintéticas (`src/synthetic.py`, IDs ≥ 50.001)
2. Feature engineering de 9 campos crudos → 19 features (`src/features.py`)
3. Auto-etiquetado con reglas calibradas al puente (`src/labeling.py`)
4. StandardScaler fit en train, transform en train+test
5. SMOTE en conjunto de entrenamiento solamente

---

## Métricas de Evaluación

| Métrica | Objetivo | Estado |
|---|---|---|
| **F1-macro** | ≥ 0.85 | Pendiente primera ejecución |
| **Recall Normal** | > 0.90 | Pendiente |
| **Recall Reducido** | > 0.50 | Pendiente |
| **Recall Congestionado** | > 0.50 | Pendiente |
| **Recall Accidente** | > 0 | Pendiente (clase crítica) |

### Gate Conservador de Accidentes

El modelo incluye un **gate post-predicción** que puede anular la clasificación del MLP cuando la evidencia heurística de accidente es fuerte:

- Score de evidencia ≥ 0.75
- Velocidad persistentemente baja + frenado reciente o acumulado + evidencia de movimiento
- Solo se aplica si el modelo predijo Congestionado/Accidente O tiene baja confianza (< 0.70)

---

## Sesgos y Limitaciones

### Sesgos Conocidos

1. **Sesgo de auto-etiquetado circular**: El modelo aprende los umbrales que lo etiquetaron, no necesariamente el estado real del tráfico
2. **Sesgo temporal**: Datos solo de abril-julio 2025 (otoño-invierno argentino). No hay datos de verano ni temporada alta
3. **Sesgo geográfico**: Entrenado exclusivamente con datos del Puente Belgrano. No es transferible a otros contextos
4. **Clases sintéticas**: Congestionado y Accidente son aproximaciones de ingeniería, no eventos observados
5. **Proxy de clima**: `weather_condition` es un proxy binario por hora, no datos meteorológicos reales

### Limitaciones Técnicas

1. **Sin temporalidad**: El MLP procesa cada registro independientemente. No tiene memoria de registros anteriores
2. **Sin ground truth**: Las etiquetas no son validadas por humanos
3. **Desequilibrio residual**: Incluso con SMOTE, Normal domina (~75%)

### Recomendaciones de Mitigación

1. Implementar validación HITL con operadores SISE para generar ground truth
2. Evolucionar a LSTM para capturar patrones temporales
3. Reemplazar proxy de clima con API meteorológica real
4. Re-entrenar con datos de todas las estaciones del año

---

## Consideraciones Éticas

### Privacidad

- El modelo opera sobre **datos agregados por minuto**, no sobre vehículos individuales
- No procesa ni almacena patentes, imágenes de personas, ni datos de identidad
- Los videos fuente son propiedad del sistema SISE y no se distribuyen

### Fairness

- El modelo no discrimina por tipo de vehículo en la clasificación de estado (todos contribuyen igualmente)
- El sesgo de detección de YOLO 11 hacia bicicletas (sub-representadas en COCO) puede afectar indirectamente la calidad de la telemetría
- No hay evaluación formal de fairness por hora del día o condición ambiental

### Impacto Ambiental

| Aspecto | Estimación |
|---|---|
| **Hardware de entrenamiento** | Google Colab Free (GPU T4/V100 compartida) |
| **Tiempo de entrenamiento estimado** | < 5 minutos (dataset pequeño, MLP liviano) |
| **Huella de carbono estimada** | Despreciable (< 0.001 kg CO₂eq por entrenamiento) |
| **Optimización de eficiencia** | EarlyStopping previene entrenamiento innecesario |

---

## Cómo Citar

```bibtex
@software{vaaet_traffic_classifier,
  author = {González, Facundo Nicolás},
  title = {VAAET Traffic State Classifier},
  year = {2025},
  url = {https://github.com/zgfnicolas/vaaet},
  license = {MIT}
}
```

---

## Historial de Versiones del Modelo

| Versión | Fecha | Cambios |
|---|---|---|
| mlp-v1.0 | 2025-03-07 | Versión inicial con 14 features |
| mlp-v1.1 | 2025-07-14 | Ampliación a 19 features (señales de calidad, cumulative delta, low-speed persistence) |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23

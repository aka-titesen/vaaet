<!-- context: VAAET/docs/ml/bias-and-limitations.md — Sesgos y limitaciones.
Complementa KPIs/KPIs.md (métricas) y DATA_LINEAGE.md (linaje). -->

# Sesgos y Limitaciones — VAAET

Este documento describe los sesgos conocidos, las limitaciones técnicas, y las restricciones operativas del sistema VAAET. Su propósito es proveer transparencia sobre las condiciones bajo las cuales el sistema puede no operar correctamente.

---

## 1. Sesgos de Detección

### 1.1 Sesgo del Dataset COCO

YOLO 11 está pre-entrenado en MS COCO 2017, cuya distribución de clases vehiculares no es representativa del tráfico argentino:

| Clase | Representación en COCO | Impacto Esperado |
|---|---|---|
| auto | Alta (sobre-representada) | Detección confiable |
| camión | Media | Detección aceptable |
| colectivo | Baja | Posible confusión con camiones grandes |
| motocicleta | Media | Aceptable, pero las motos argentinas pueden diferir visualmente |
| bicicleta | Baja (sub-representada) | **Riesgo de sub-detección**, especialmente a distancia |

**Mitigación actual**: Ninguna. Se confía en la capacidad de generalización de YOLO 11.

**Mitigación futura recomendada**: Fine-tuning con dataset de tráfico argentino (especialmente bicicletas y colectivos).

### 1.2 Sesgo de Condiciones Ambientales

El sistema NO ha sido evaluado sistemáticamente bajo:

| Condición | Riesgo | Evaluación |
|---|---|---|
| Lluvia intensa | Reflejos en pavimento, gotas en lente | No evaluado |
| Niebla | Contraste y visibilidad reducidos | No evaluado |
| Noche (sin iluminación) | Señal baja del sensor de cámara | No evaluado |
| Contraluz (amanecer/atardecer) | Siluetas, pérdida de detalle | No evaluado |
| Sombras pronunciadas | Posibles falsos positivos | No evaluado |
| Nieve/escarcha | N/A para Corrientes (clima subtropical) | No aplica |

### 1.3 Sesgo de Geometría de Cámara

- Las cámaras SISE son dinámicas (pan/tilt/zoom), introduciendo variabilidad de perspectiva
- La corrección de perspectiva usa un modelo simplificado (factor por zona Y), no una homografía completa
- Los vehículos en los extremos laterales del frame tienen corrección de perspectiva menos precisa
- El zoom variable puede alterar la relación `pixels_per_meter` durante el video

---

## 2. Limitaciones Técnicas

### 2.1 Velocidad sin Ground Truth

- La conversión píxel-a-metro depende de la calibración manual de `pixels_per_meter`
- NO existe un dataset de velocidades reales para el Puente Belgrano
- El MAE real es desconocido — el objetivo de "< 5 km/h" es una meta sin benchmark
- La calibración se estimó a partir de dimensiones conocidas del puente (8,3m de ancho, cámaras a 60m)

### 2.2 El MLP Opcional de Fusión de Velocidad No es Parte del Flujo Activo

- El pipeline activo de telemetría es physics-first y **no** conecta la fusión MLP opcional por defecto
- La nomenclatura legada `cnn_validator` es histórica y engañosa; el path de código opcional no se trata como señal validada
- La mitigación actual: el runtime activo depende de compensación de flujo óptico, filtros de plausibilidad, gates de confiabilidad, y agregación robusta
- Ver [ADR-0004](../architecture/decisions/0004-mlp-speed-smoother.md) como contexto histórico, no como diseño del runtime actual

### 2.3 Tracking sin Re-identificación Visual

- SORT usa matching por distancia euclidiana al centroide más cercano
- Si un vehículo es ocluido por >1 segundo, pierde su ID y recibe uno nuevo
- En tráfico denso con vehículos del mismo tipo cercanos, pueden ocurrir asignaciones incorrectas
- Ver [ADR-0003](../architecture/decisions/0003-sort-over-deepsort.md)

### 2.4 Detección de Estacionarios

- Requiere 200 frames (~6,5s) mínimo de observación — no hay detección temprana
- Vehículos con micro-movimientos (vibración del motor, viento) pueden no detectarse como estacionarios
- Umbrales en píxeles fijos — no se adaptan a resolución ni zoom
- Ver [ADR-0006](../architecture/decisions/0006-conservative-stationary-detection.md)

---

## 3. Limitaciones de Infraestructura

### 3.1 Google Colab

| Limitación | Impacto |
|---|---|
| Sesiones máximas ~12h (Free) | Videos muy largos pueden no completarse |
| GPU no garantizada en horas pico | Puede recurrir a CPU (10x más lento) |
| Desconexiones aleatorias | Progreso perdido — no hay checkpointing |
| Almacenamiento efímero | Videos de salida se pierden al cerrar la sesión |
| Sin ejecución programática | No se puede automatizar vía API o cron |

### 3.2 AWS RDS

| Limitación | Impacto |
|---|---|
| Requiere instancia provisionada externamente | Costo recurrente de AWS |
| Sin connection pooling | Cada escritura abre/cierra conexión |
| Sin reintento automático | Si falla la conexión, se pierde el registro del minuto |
| Sin SSL por defecto | Conexión en texto plano (riesgo en redes no confiables) |
| Sin migraciones de esquema | `CREATE TABLE IF NOT EXISTS` como único mecanismo |

---

## 4. Limitaciones del Código

El workflow activo de adquisición evita el antiguo motor monolítico. Persisten
estas limitaciones técnicas del enfoque visual:

| Problema | Ubicación | Impacto |
|---|---|---|
| Calibración manual de escala | Estimación de velocidad | La precisión depende de geometría y perspectiva de cámara |
| SORT sin reidentificación visual | Tracking | Oclusiones largas pueden crear un ID nuevo |
| Etiquetas mediante reglas | Entrenamiento | Son proxies de ingeniería, no ground truth humano |
| Runtime efímero | Google Colab | Videos y outputs deben descargarse o copiarse antes del reinicio |
| Persistencia opt-in | PostgreSQL | Sin credenciales configuradas, los resultados quedan sólo en memoria/CSV |

El código compartido permite probar y mejorar estos componentes sin duplicarlos
en los notebooks.

---

## 5. Recomendaciones de Mejora Priorizadas

1. **Evaluar con video real** — medir KPIs con ground truth
2. **Fine-tune de YOLO** con datos de tráfico argentino — mejorar detección de bicicletas
3. **Agregar SSL** a la conexión PostgreSQL — seguridad
4. **Implementar checkpointing** — resiliencia ante desconexiones de Colab
5. **Reemplazar proxy de clima** con API meteorológica real
6. **Evolucionar MLP a LSTM** — capturar patrones temporales en clasificación

---

## 6. Sesgos y Limitaciones del Clasificador (Módulos 1 y 2)

### 6.1 Sesgo de Auto-etiquetado

Las etiquetas de entrenamiento NO son ground truth humano. Se generan con reglas de ingeniería (umbrales de velocidad, volumen, persistencia). Esto introduce sesgo circular: el modelo aprende los umbrales que lo etiquetaron, no necesariamente el estado real del tráfico.

**Mitigación actual**: Los umbrales están calibrados a percentiles de la distribución de datos del Puente Belgrano (recalibrados el 2026-03-11, reemplazando valores genéricos de libro). Ver `src/vaaet/settings.py` `LABELING_THRESHOLDS`.

**Mitigación futura**: HITL (campos `is_human_validated`, `human_override_state` en `traffic_classifications`) permitirá que operadores SISE refinen las etiquetas.

### 6.2 Desequilibrio de Clases y Datos Sintéticos

El dataset real del Puente Belgrano (abril-julio 2025, ~2.000 registros) solo exhibe tráfico Normal y Reducido. Los eventos de Congestión y Accidente nunca ocurrieron durante el período de observación (velocidad mínima registrada: 5,4 km/h, conteo máximo: 28 vehículos/min).

| Clase | Datos Reales | Sintéticos | Frecuencia Combinada | Riesgo |
|---|---|---|---|---|
| Normal | ~2.004 | 0 | ~75% | Sobre-representada — el modelo tiende a predecir Normal |
| Reducido | ~63 | 0 | ~3% | Baja representación — SMOTE compensa |
| Congestionado | 0 | ~50 | ~2% | Totalmente sintético — puede no capturar todos los patrones reales |
| Accidente | 0 | ~50 | ~2% | Totalmente sintético — plausible pero no validado |

**Mitigaciones**:
1. **Secuencias sintéticas** (`src/vaaet/features/synthetic.py`): Telemetría plausible de casos extremos inyectada antes del feature engineering. IDs ≥ 50.001 para trazabilidad.
2. **SMOTE**: Aplicado en el conjunto de entrenamiento después de la inyección sintética.
3. **Recalibración de umbrales**: Los umbrales de Reducido y Congestionado se ajustaron a la distribución específica del puente.

**Limitación**: Las muestras sintéticas de Accidente y Congestión son aproximaciones de ingeniería, no eventos observados. El rendimiento del modelo en incidentes reales es desconocido hasta que la validación HITL provea correcciones de ground truth.

### 6.3 Supuestos Temporales

Los umbrales de persistencia en el auto-etiquetado asumen que cada registro representa aproximadamente 1 minuto. Si cambia la frecuencia de adquisición, los estados temporales deben recalibrarse.

### 6.4 Clima Simulado

La feature `weather_condition` es un proxy basado en hora del día (0=diurno 6-18h, 1=nocturno). NO son datos meteorológicos reales. Esto limita la capacidad del modelo para aprender patrones relacionados con el clima.

### 6.5 Sin Temporalidad en el MLP

El MLP procesa cada registro independientemente — no tiene memoria de registros anteriores. No puede aprender patrones secuenciales como "velocidad decreciente durante 5 minutos consecutivos". La evolución a LSTM en una fase futura abordará esta limitación.

### 6.6 Generalización

El modelo está entrenado exclusivamente con datos del Puente General Manuel Belgrano. NO es transferible a otros puentes, autopistas o contextos urbanos sin re-entrenamiento con datos locales.

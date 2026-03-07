<!-- context: VAAET/Docs/BIAS_AND_LIMITATIONS.md — Sesgos y limitaciones del sistema.
Complementa KPIs.md (métricas) y DATA_LINEAGE.md (linaje). -->

# Sesgos y Limitaciones — VAAET

Este documento describe los sesgos conocidos, limitaciones técnicas y restricciones operativas del sistema VAAET. Su propósito es proveer transparencia sobre las condiciones en las cuales el sistema puede no operar correctamente.

---

## 1. Sesgos de Detección

### 1.1 Sesgo del Dataset COCO

YOLO 11 está pre-entrenado en MS COCO 2017, cuya distribución de clases vehiculares no es representativa del tráfico argentino:

| Clase | Representación en COCO | Impacto Esperado |
|---|---|---|
| car | Alta (sobre-representado) | Detección confiable |
| truck | Media | Detección aceptable |
| bus | Baja | Posible confusión con trucks grandes |
| motorcycle | Media | Aceptable, pero motos argentinas pueden diferir visualmente |
| bicycle | Baja (sub-representado) | **Riesgo de sub-detección**, especialmente a distancia |

**Mitigación actual**: Ninguna. Se confía en la capacidad de generalización de YOLO 11.

**Mitigación futura recomendada**: Fine-tuning con dataset de tráfico argentino (especialmente bicicletas y colectivos).

### 1.2 Sesgo por Condiciones Ambientales

El sistema NO ha sido evaluado sistemáticamente en las siguientes condiciones:

| Condición | Riesgo | Evaluación |
|---|---|---|
| Lluvia intensa | Reflexiones en pavimento, gotas en lente | No evaluado |
| Niebla | Reducción de contraste y visibilidad | No evaluado |
| Noche (sin iluminación) | Baja señal en sensor de cámara | No evaluado |
| Contraluz (amanecer/atardecer) | Siluetas, pérdida de detalle | No evaluado |
| Sombras pronunciadas | Posibles falsos positivos | No evaluado |
| Nieve/escarcha | N/A para Corrientes (clima subtropical) | No aplica |

### 1.3 Sesgo por Geometría de Cámara

- Las cámaras SISE son dinámicas (pan/tilt/zoom), lo que introduce variabilidad en perspectiva
- La corrección de perspectiva usa un modelo simplificado (factor por zona Y), no una homografía completa
- Vehículos en los extremos laterales del frame tienen corrección de perspectiva menos precisa
- El zoom variable puede alterar la relación `pixels_per_meter` durante el video

---

## 2. Limitaciones Técnicas

### 2.1 Velocidad sin Ground Truth

- La conversión pixel → metro depende de `pixels_per_meter` calibrado manualmente
- NO existe dataset de velocidades reales para el Puente Belgrano
- El MAE real es desconocido — el target de "< 5 km/h" es un objetivo sin benchmark
- La calibración fue estimada a partir de las dimensiones conocidas del puente (8.3m ancho, cámaras a 60m)

### 2.2 MLP Entrenado con Datos Random

- El componente `cnn_validator` (nombre incorrecto — es un `MLPRegressor`) se entrena con `np.random.rand()`
- NO aporta aprendizaje real sobre patrones de velocidad
- Actúa como regularizador estocástico hacia la media (~60 km/h)
- Su contribución está acotada al 30% de la fusión, limitando el daño
- Consultar [ADR-004](adr/ADR-004-mlp-como-suavizador.md) para más detalle

### 2.3 Tracking sin Re-Identificación Visual

- SORT usa matching por distancia euclidiana al centroide más cercano
- Si un vehículo es ocluido por más de ~1 segundo, pierde su ID y recibe uno nuevo
- En tráfico denso con vehículos del mismo tipo muy cercanos, puede haber asignaciones incorrectas
- Consultar [ADR-003](adr/ADR-003-sort-sobre-deepsort.md)

### 2.4 Detección de Estacionarios

- Requiere 200 frames (~6.5s) de observación mínima — no hay detección temprana
- Vehículos con micro-movimientos (vibración del motor, viento) pueden no ser detectados como estacionarios
- Umbrales fijos en píxeles — no se adaptan a resolución ni zoom
- Consultar [ADR-006](adr/ADR-006-deteccion-estacionarios-conservadora.md)

---

## 3. Limitaciones de Infraestructura

### 3.1 Google Colab

| Limitación | Impacto |
|---|---|
| Sesiones máximo ~12h (Free) | Videos muy largos pueden no completarse |
| GPU no garantizada en horas pico | Puede caer a CPU (10x más lento) |
| Desconexiones aleatorias | Se pierde el progreso — sin checkpointing |
| Almacenamiento efímero | Videos de salida se pierden al cerrar sesión |
| Sin ejecución programática | No se puede automatizar via API o cron |

### 3.2 AWS RDS

| Limitación | Impacto |
|---|---|
| Requiere instancia aprovisionada externamente | Costo AWS recurrente |
| Sin connection pooling | Cada escritura abre/cierra conexión |
| Sin retry automático | Si falla la conexión, se pierde el registro del minuto |
| Sin SSL por defecto | Conexión en texto plano (riesgo en redes no confiables) |
| Sin migraciones de esquema | CREATE TABLE IF NOT EXISTS como único mecanismo |

---

## 4. Limitaciones del Código

| Issue | Ubicación | Impacto |
|---|---|---|
| Código muerto en `is_stationary()` | Cell 3 | Segundo bloque de criterios inalcanzable después del primer `return` |
| Método duplicado `get_smoothed_average()` | Cell 3 | Dos definiciones — la segunda sobreescribe la primera |
| Inconsistencia `pixels_per_meter`: 12 vs 15 | Cell 3 vs Cell 5 | Posible error en conversión de velocidad |
| Inconsistencia `speed_limits` | Cell 3 vs Cell 5 | Rangos diferentes para filtrado |
| Inconsistencia `min_track_frames`: 10 vs 20 | Cell 5 vs Cell 3 | Umbral inconsistente |
| `threading` importado pero no usado | Cell 2 | Dependencia innecesaria |
| `scipy` en README pero no importado | README | Dependencia falsa |
| `detection_zone` definido pero no usado | Cell 5 | Código muerto |
| Nombre "CNN" para un MLP | Cell 3, docs | Terminología incorrecta |

---

## 5. Recomendaciones de Mejora Priorizadas

1. **Corregir inconsistencias de config** (Cells 3 y 5) — impacto directo en precisión
2. **Eliminar código muerto y duplicados** — mejorar mantenibilidad
3. **Agregar seed fijo al MLP** — reproducibilidad entre ejecuciones
4. **Renombrar `cnn_validator` → `mlp_speed_smoother`** — claridad semántica
5. **Evaluar con video real** — medir KPIs con ground truth
6. **Fine-tuning de YOLO** con datos de tráfico argentino — mejorar detección de bicicletas
7. **Agregar SSL** a la conexión PostgreSQL — seguridad
8. **Implementar checkpointing** — resiliencia ante desconexiones de Colab

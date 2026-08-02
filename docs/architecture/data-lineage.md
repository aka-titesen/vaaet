<!-- context: VAAET/docs/DATA_LINEAGE.md — Documentación de linaje de datos.
Complementa SAD.md (arquitectura) y BIAS_AND_LIMITATIONS.md (sesgos). -->

# Linaje de Datos — VAAET

Este documento describe el origen, transformación y destino de todos los datos que fluyen a través del sistema VAAET.

---

## 1. Fuentes de Datos

### Video de Entrada

| Atributo | Valor |
|---|---|
| **Origen** | Cámaras de vigilancia SISE en el Puente Gral. Manuel Belgrano |
| **Formato** | MP4 (H.264) |
| **Resolución típica** | 1920×1080 (Full HD) |
| **FPS** | 30 fps (configurable por cámara) |
| **Convención de nombre** | `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4` |
| **Propietario** | Sistema SISE (Dirección Nacional de Vialidad) |
| **Acceso** | Restringido — los videos no se distribuyen con el proyecto |

### Modelos Pre-entrenados

| Modelo | Fuente | Dataset de Entrenamiento | Clases Relevantes |
|---|---|---|---|
| yolo11n/s/m/l/x.pt | Ultralytics Hub | MS COCO 2017 | auto, camión, colectivo, motocicleta, bicicleta |

**Nota**: Los modelos se descargan automáticamente en runtime y NO se versionan en el repositorio.

---

## 2. Módulo 0 — Pipeline Bootstrap (ARCHIVADO)

Ubicado en `archive/00_bootstrap/01_legacy_collection.ipynb`. Este pipeline generó la tabla histórica `traffic_data` y **no se ejecuta nunca más**.

```
Video MP4
  │
  ├── [1] Validación de nombre ──── Rechaza si el formato no coincide
  │
  ├── [2] Extracción de frames ──── OpenCV VideoCapture @ 30fps
  │
  ├── [3] Detección YOLO 11 ─────── Por frame:
  │     │                             Entrada: frame BGR (numpy array)
  │     │                             Salida: lista de (bbox, clase, confianza)
  │     │                             Filtros: conf > 0.5, NMS IoU < 0.4
  │     └── Solo clases vehiculares
  │
  ├── [4] Tracking SORT ─────────── Matching por distancia euclidiana
  │     │                             Umbral: 100px, mismo tipo de vehículo
  │     └── Almacena historial de centroides en deque(30)
  │
  ├── [5] Velocidad híbrida ─────── Por track activo:
  │     │   [5a] Flujo Óptico (Lucas-Kanade) → vector de movimiento global
  │     │   [5b] Compensación de cámara → restar movimiento global
  │     │   [5c] Desplazamiento euclidiano → distancia en píxeles
  │     │   [5d] Corrección de perspectiva → factor por zona Y
  │     │   [5e] Conversión → píxeles/metro × factor → km/h
  │     │   [5f] Predicción MLP → 10 features → velocidad suavizada
  │     │   [5g] Fusión → 0.7 × física + 0.3 × MLP
  │     │   [5h] Filtro de plausibilidad → [2, 120] km/h
  │     └── Velocidades fuera de rango descartadas silenciosamente
  │
  ├── [6] Clasificación estacionarios ── AND de 6 criterios estadísticos
  │     │                                  Si estacionario → velocidad = 0
  │     └── Requiere mínimo 200 frames (~6.5s) de observación
  │
  ├── [7] Anotación visual ─────── Bounding boxes, tipo + ID, velocidad, HUD
  │
  └── [8] Persistencia (c/60s) ─── INSERT en PostgreSQL:
        │   clip_id, record_time, avg_speed, count_* por tipo, total
        └── Opcional — fallo silencioso si no hay BD
```

---

## 3. Datos de Entrenamiento del MLP Suavizador de Velocidad

| Atributo | Valor |
|---|---|
| **Tipo** | Datos sintéticos aleatorios |
| **Generación** | `np.random.rand(100, 10)` para features, `np.random.rand(100) * 80 + 20` para targets |
| **Semilla** | No fija — cada ejecución genera datos diferentes |
| **Propósito** | Scaffold para inicializar MLPRegressor. NO es entrenamiento real |
| **Impacto** | El MLP actúa como regularizador hacia la media (~60 km/h); contribución limitada al 30% |

Ver [ADR-004](adr/ADR-004-mlp-como-suavizador.md) para la justificación.

---

## 4. Datos de Salida del Módulo 0

### Video Anotado

| Atributo | Valor |
|---|---|
| **Formato** | MP4 (codec mp4v) |
| **Contenido** | Frames originales + bounding boxes + tipo/ID + velocidad + HUD |
| **Destino** | Descarga automática al dispositivo del usuario (Colab) |
| **Retención** | Efímera — se pierde al cerrar la sesión de Colab |

### Registros de Base de Datos (`traffic_data`)

| Campo | Tipo | Descripción |
|---|---|---|
| `clip_id` | TEXT | Identificador derivado del nombre del video |
| `record_time` | TIMESTAMP | Marca temporal del minuto registrado |
| `avg_speed` | NUMERIC(5,2) | Velocidad promedio de vehículos en movimiento (km/h) |
| `count_car` | INTEGER | Autos detectados en el minuto |
| `count_truck` | INTEGER | Camiones detectados |
| `count_bus` | INTEGER | Colectivos detectados |
| `count_motorcycle` | INTEGER | Motocicletas detectadas |
| `count_bicycle` | INTEGER | Bicicletas detectadas |
| `total_vehicles` | INTEGER | Total de vehículos en el minuto |

**Frecuencia**: Un registro cada 60 segundos de video procesado.
**Destino**: Tabla `traffic_data` en PostgreSQL (AWS RDS).

---

## 5. Módulo 1 — Pipeline de Preparación de Datos

`notebooks/01_data_prep/data_preparation.ipynb` — Se ejecuta una vez para entrenar el clasificador.

```
traffic_data (PostgreSQL)
  │
  ├── [1] Consulta SQL ─────── SELECT 9 campos + id, ordenados por record_time
  │
  ├── [2] DataFrame ─────────── ~2.000 registros × 10 columnas (pandas)
  │
  ├── [3] Inyección sintética ── Accidentes + congestión vía src/synthetic.py
  │     └── IDs ≥ 50.001 para trazabilidad
  │
  ├── [4] Feature Engineering ── 9 campos crudos → 19 features vía src/features.py:
  │     │   heavy_vehicle_ratio, delta_speed, delta_count,
  │     │   transition_flag, speed_variance, cumulative_delta_speed,
  │     │   low_speed_persistence, speed_measurement_quality,
  │     │   near_zero_motion_ratio, stationary_confirmed_ratio,
  │     │   hour_of_day, weather_condition
  │     └── Elimina filas iniciales con NaN de diff()
  │
  ├── [5] Auto-etiquetado ───── Reglas de ingeniería → 4 estados vía src/labeling.py:
  │     │   Accidente(3) → Congestionado(2) → Reducido(1) → Normal(0)
  │     └── NO es ground truth humano. Es un proxy de ingeniería.
  │
  ├── [6] SMOTE ──────────────── Balanceo del conjunto de entrenamiento (solo train)
  │     │   StandardScaler fit en train, transform en ambos
  │     └── El conjunto de test mantiene la distribución original
  │
  ├── [7] Entrenamiento MLP ─── Dense(64) → Dense(32) → Softmax(4)
  │     │   EarlyStopping + ReduceLROnPlateau, seed=42
  │     └── Exporta: traffic_classifier.keras, feature_scaler.joblib
  │
  ├── [8] Evaluación ────────── F1-macro, matriz de confusión, recall por clase
  │
  └── [9] Persistencia ──────── 2 tablas:
        │   telemetry_raw: 19 features + FK a traffic_data(id)
        │   traffic_classifications: predicción + confianza + HITL
        └── Opcional — fallo silencioso si no hay BD
```

### Artefactos del Módulo 1

| Artefacto | Ruta | Formato | Gitignored |
|---|---|---|---|
| Modelo entrenado | `models/intelligence/traffic_classifier.keras` | Keras nativo | Sí |
| Scaler | `models/intelligence/feature_scaler.joblib` | joblib | Sí |
| Mapeo de etiquetas | `models/intelligence/label_mapping.joblib` | joblib | Sí |

---

## 6. Módulo 2 — Pipeline de Producción

`notebooks/02_production/traffic_analyzer.ipynb` — Se ejecuta continuamente para análisis de tráfico.

```
Video MP4 (clip nuevo)
  │
  ├── [1] Pipeline de percepción ── YOLODetector + SORTTracker + estimación de velocidad
  │     │   vía módulos src/perception/
  │     └── Produce DataFrame de telemetría (9 campos crudos por minuto)
  │
  ├── [2] Feature engineering ────── 9 → 19 features vía src/features.py
  │
  ├── [3] Clasificación ──────────── Cargar modelo + scaler de models/intelligence/
  │     │   Escalar features, predecir estado + confianza
  │     │   Aplicar gate conservador de accidentes
  │     └── 4 estados: Normal(0), Reducido(1), Congestionado(2), Accidente(3)
  │
  ├── [4] Persistencia ──────────── INSERT en telemetry_raw + traffic_classifications
  │     └── Opcional — fallo silencioso si no hay BD
  │
  └── [5] Scaffold HITL ────────── Correcciones de operadores SISE
        │   Actualiza is_human_validated, human_override_state
        └── Experimental — no es un bucle productivo validado
```

---

## 7. Privacidad y Seguridad de Datos

### Datos Potencialmente Sensibles

- **Patentes**: Los videos SISE pueden capturar patentes. VAAET NO extrae, almacena ni procesa patentes individuales
- **Frames individuales**: NO se almacenan en BD. Solo datos agregados por minuto
- **Ubicación temporal**: Los timestamps en las tablas permiten inferir patrones de tráfico por hora

### Credenciales

- Credenciales de AWS RDS obtenidas vía variables de entorno o `getpass`
- NUNCA impresas en outputs de celdas
- NUNCA hardcodeadas en código
- NUNCA versionadas en Git (`.gitignore` excluye `.env`)

### Datos NO Recolectados

- Identidad de conductores
- Patentes individuales
- Imágenes de personas
- Datos de tracking individual fuera del video procesado

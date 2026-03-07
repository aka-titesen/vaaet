<!-- context: VAAET/Docs/DATA_LINEAGE.md — Documentación del linaje de datos del sistema.
Complementa DDS.md (diseño técnico) y BIAS_AND_LIMITATIONS.md (sesgos). -->

# Linaje de Datos — VAAET

Este documento describe el origen, transformación y destino de todos los datos que fluyen a través del sistema VAAET.

---

## 1. Fuente de Datos

### Video de Entrada

| Atributo | Valor |
|---|---|
| **Origen** | Cámaras de vigilancia SISE del Puente Gral. Manuel Belgrano |
| **Formato** | MP4 (H.264) |
| **Resolución típica** | 1920×1080 (Full HD) |
| **FPS** | 30 fps (configurable por la cámara) |
| **Convención de nombre** | `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4` |
| **Propietario** | Sistema SISE (Dirección Nacional de Vialidad) |
| **Acceso** | Restringido — los videos no se distribuyen con el proyecto |

### Modelos Pre-entrenados

| Modelo | Fuente | Dataset de Entrenamiento | Clases Relevantes |
|---|---|---|---|
| yolo11n/s/m/l/x.pt | Ultralytics Hub | MS COCO 2017 | car, truck, bus, motorcycle, bicycle |

**Nota**: Los modelos se descargan automáticamente en runtime y NO se versionan en el repositorio.

---

## 2. Pipeline de Transformación

```
Video MP4
  │
  ├── [1] Validación de nombre ──── Rechaza si no cumple formato
  │
  ├── [2] Extracción de frames ──── OpenCV VideoCapture @ 30fps
  │
  ├── [3] Detección YOLO 11 ──────── Por cada frame:
  │     │                              Input: frame BGR (numpy array)
  │     │                              Output: lista de (bbox, clase, confianza)
  │     │                              Filtros: conf > 0.5, NMS IoU < 0.4
  │     │
  │     └── Solo clases vehiculares: car, truck, bus, motorcycle, bicycle
  │
  ├── [4] Tracking SORT ─────────── Matching por distancia euclidiana
  │     │                              Input: detecciones del frame actual
  │     │                              Output: track_id asignado a cada detección
  │     │                              Umbral: 100px, mismo tipo de vehículo
  │     │
  │     └── Almacena historial de centroides en deque(30)
  │
  ├── [5] Velocidad híbrida ─────── Por cada track activo:
  │     │   [5a] Optical Flow (Lucas-Kanade) → vector de movimiento global
  │     │   [5b] Compensación de cámara → restar movimiento global
  │     │   [5c] Desplazamiento euclidiano → distancia en píxeles
  │     │   [5d] Corrección perspectiva → factor por zona Y
  │     │   [5e] Conversión → píxeles/meter × factor → km/h
  │     │   [5f] MLP prediction → 10 features → velocidad suavizada
  │     │   [5g] Fusión → 0.7 × física + 0.3 × MLP
  │     │   [5h] Filtro plausibilidad → [2, 120] km/h
  │     │
  │     └── Velocidades fuera de rango se descartan silenciosamente
  │
  ├── [6] Clasificación estacionario ── AND de 6 criterios estadísticos
  │     │                                  Si es estacionario → velocidad = 0
  │     │
  │     └── Requiere mínimo 200 frames de observación
  │
  ├── [7] Anotación visual ──────── Dibujar en frame:
  │     │   Bounding boxes, tipo + ID, velocidad, HUD informativo
  │     │
  │     └── Escribir frame al video de salida (OpenCV VideoWriter)
  │
  └── [8] Persistencia (cada 60s) ── INSERT en PostgreSQL:
        │   clip_id, record_time, avg_speed, count_* por tipo, total
        │
        └── Opcional — falla silenciosa si no hay BD configurada
```

---

## 3. Datos de Entrenamiento del MLP

| Atributo | Valor |
|---|---|
| **Tipo** | Datos sintéticos aleatorios |
| **Generación** | `np.random.rand(100, 10)` para features, `np.random.rand(100) * 80 + 20` para targets |
| **Seed** | No fijado — cada ejecución genera datos diferentes |
| **Propósito** | Scaffold para inicializar el MLPRegressor. NO es entrenamiento real |
| **Impacto** | El MLP actúa como regularizador hacia la media (~60 km/h), su contribución está acotada al 30% |

Consultar [ADR-004](adr/ADR-004-mlp-como-suavizador.md) para el razonamiento detrás de esta decisión.

---

## 4. Datos de Salida

### Video Anotado

| Atributo | Valor |
|---|---|
| **Formato** | MP4 (codec mp4v) |
| **Contenido** | Frames originales + bounding boxes + tipo/ID + velocidad + HUD |
| **Destino** | Descarga automática al dispositivo del usuario (Colab) |
| **Retención** | Efímero — se pierde al cerrar la sesión de Colab |

### Registros de Base de Datos

| Campo | Tipo | Descripción |
|---|---|---|
| `clip_id` | TEXT | Identificador derivado del nombre del video |
| `record_time` | TIMESTAMP | Marca temporal del minuto registrado |
| `avg_speed` | NUMERIC(5,2) | Velocidad promedio de vehículos en movimiento (km/h) |
| `count_car` | INTEGER | Autos detectados en el minuto |
| `count_truck` | INTEGER | Camiones detectados en el minuto |
| `count_bus` | INTEGER | Buses detectados en el minuto |
| `count_motorcycle` | INTEGER | Motos detectadas en el minuto |
| `count_bicycle` | INTEGER | Bicicletas detectadas en el minuto |
| `total_vehicles` | INTEGER | Total de vehículos en el minuto |

**Frecuencia**: Un registro cada 60 segundos de video procesado.
**Destino**: Tabla `traffic_data` en PostgreSQL (AWS RDS).
**Retención**: Indefinida (depende de la política de la instancia RDS).

---

## 5. Privacidad y Seguridad de Datos

### Datos sensibles potenciales

- **Patentes vehiculares**: Los videos de cámaras SISE pueden capturar patentes. VAAET NO extrae, almacena ni procesa patentes individualmente
- **Frames individuales**: NO se almacenan frames en la BD. Solo datos agregados por minuto
- **Ubicación temporal**: Los timestamps en `traffic_data` permiten inferir patrones de tráfico por hora

### Credenciales

- Las credenciales de AWS RDS se obtienen via variables de entorno o `getpass`
- NUNCA se imprimen en outputs de celdas
- NUNCA se hardcodean en el código
- NUNCA se versionan en Git (`.gitignore` excluye `.env`)

### Datos que NO se recopilan

- Identidad de conductores
- Patentes individuales
- Imágenes de personas
- Datos de tracking individual fuera del video procesado

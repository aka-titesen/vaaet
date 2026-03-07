# AGENTS.md — Contexto Agéntico para VAAET

> Este archivo es la memoria a largo plazo para agentes de IA que operen sobre este repositorio.
> Léelo completo antes de realizar cualquier modificación al proyecto.

---

## Identidad del Proyecto

**VAAET** (Video Análisis Avanzado de Tráfico) es un sistema de visión por computadora para analizar tráfico vehicular en el **Puente General Manuel Belgrano** (Corrientes, Argentina). Detecta, clasifica, rastrea y estima la velocidad de vehículos usando video de cámaras de vigilancia SISE.

### Arquitectura Fundamental

- **Pipeline de dos etapas en notebooks separados**:
  - **Etapa 1 — Percepción**: `01_legacy_collection.ipynb` — YOLO 11 + OpenCV + SORT → telemetría cruda cada minuto
  - **Etapa 2 — Inteligencia**: `02_traffic_state_classifier.ipynb` — TF/Keras MLP → clasificación de estado de tráfico
- **Entorno de ejecución**: Google Colab (acceso a GPU gratuita). NO hay servidor, API REST, microservicios ni contenedores
- **Persistencia**: PostgreSQL en AWS RDS (opcional). 3 tablas: `traffic_data` (Etapa 1), `telemetry_raw` + `traffic_classifications` (Etapa 2)
- **No hay CI/CD clásico**: No hay pipeline de build, no hay deploy. El "deploy" es abrir los notebooks en Colab y ejecutar las celdas

### Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Detección de objetos | YOLO 11 (Ultralytics) — 5 variantes por duración |
| Visión por computadora | OpenCV — video I/O, optical flow, anotaciones |
| Cómputo numérico | NumPy — operaciones vectoriales, estadísticas |
| ML (suavizado) | scikit-learn `MLPRegressor` — NO es una CNN real |
| Clasificación de tráfico | TensorFlow/Keras — MLP Sequential (evolucionable a LSTM) |
| Análisis de datos | Pandas + SQLAlchemy — feature engineering y persistencia |
| Balanceo de clases | imbalanced-learn SMOTE — oversampling sintético |
| Base de datos | PostgreSQL via `psycopg2-binary` / SQLAlchemy (AWS RDS) |
| Runtime | Google Colab (primario) o Python 3.8+ local |

---

## Estructura del Proyecto

```
vaaet/
├── notebooks/
│   ├── phase_1_perception/
│   │   └── 01_legacy_collection.ipynb   # Etapa 1: YOLO 11 + tracking + velocidad
│   └── phase_2_intelligence/
│       └── 02_traffic_state_classifier.ipynb  # Etapa 2: clasificación de estado
├── models/
│   ├── perception/                      # Modelos YOLO (descargados en runtime)
│   └── intelligence/                    # Artefactos Etapa 2 (.keras, .joblib)
├── data/
│   ├── raw/                             # Backups de BD (gitignored)
│   ├── processed/                       # CSVs de features (gitignored)
│   └── samples/                         # Datos de ejemplo
├── src/utils/                           # Utilidades compartidas (futuro)
├── docs/
│   ├── PRD.md                           # Requisitos del producto
│   ├── DDS.md                           # Diseño de software
│   ├── GUIA_USUARIO.md                  # Guía de usuario
│   ├── DATA_LINEAGE.md                  # Linaje de datos
│   ├── BIAS_AND_LIMITATIONS.md          # Sesgos y limitaciones
│   ├── KPIs/KPIs.md                     # Métricas y validación
│   ├── adr/                             # Decisiones arquitectónicas (8 ADRs)
│   │   ├── ADR-001 a ADR-007            # Etapa 1
│   │   └── ADR-008-tensorflow-keras-traffic-classifier.md  # Etapa 2
│   └── diagrams/                        # Diagramas Mermaid (8 diagramas)
│       ├── pipeline-flow.md, speed-calculation.md, erd.md
│       ├── colab-aws-architecture.md, model-selection.md, multi-camera-layout.md
│       ├── intelligence-pipeline.md     # Pipeline Etapa 2
│       └── erd-phase2.md                # ERD con 3 tablas
├── README.md
├── AGENTS.md                            # Este archivo
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── llms.txt
├── llms-full.txt
└── .gitignore
```

---

## Orden de Ejecución de Celdas

### Etapa 1 — Percepción (`01_legacy_collection.ipynb`)

| Celda | Contenido | Dependencias |
|---|---|---|
| 1 | Configuración de BD PostgreSQL | Ninguna |
| 2 | Instalación de dependencias + imports | Ninguna |
| 3 | Clase `VAAETHybrid` (motor principal) | Celda 2 |
| 4 | Utilidades: validación, selección de modelo, carga de video | Celdas 2, 3 |
| 5 | Parámetros de calibración (`BRIDGE_CONFIG`, colores) | Celda 3 |
| 6 | Visualización + función principal `process_bridge_video()` | Celdas 3, 4, 5 |
| 7 | Interfaz de carga (Colab upload / local file picker) | Celdas 2-6 |
| 8 | Generador de videos sintéticos para demos | Celdas 2, 3 |
| 9 | Ejecutor de demos | Celda 8 |

### Etapa 2 — Inteligencia (`02_traffic_state_classifier.ipynb`)

| Celda | Contenido | Dependencias |
|---|---|---|
| 0 | Setup de entorno (Colab clone/cd, local no-op) | Ninguna |
| 1 | Dependencias + imports (TF/Keras, pandas, etc.) | Celda 0 |
| 2 | Conexión BD + extracción de telemetría | Celda 1 |
| 3 | Ingeniería de features (9 → 14 columnas) | Celda 2 |
| 4 | Auto-labeling (reglas de ingeniería → 4 estados) | Celda 3 |
| 5 | SMOTE + Train/Test split | Celdas 3, 4 |
| 6 | Definición del modelo MLP + entrenamiento | Celda 5 |
| 7 | Evaluación + exportación del modelo | Celda 6 |
| 8 | Crear tablas BD + persistir resultados | Celdas 2, 7 |

---

## Contrato de Validación

No hay build ni CI/CD. El equivalente funcional es:

### Etapa 1
1. **Smoke test**: Ejecutar Cell 2 sin errores de importación
2. **Test funcional**: Ejecutar `test_sistema()` en Cell 7 — verifica que todos los componentes se inicializan correctamente
3. **Test end-to-end**: Generar un video sintético (Cells 8-9) y verificar que produce un video de salida anotado
4. **Test de BD**: Si hay acceso a AWS RDS, verificar que `save_to_database()` persiste un registro y no expone credenciales

### Etapa 2
5. **Smoke test**: Ejecutar Cell 1 sin errores de importación (TF/Keras)
6. **Data**: Cell 2 carga DataFrame con >0 filas desde `traffic_data`
7. **Features**: Cell 3 produce DataFrame con 14 columnas, sin NaN
8. **Clasificación**: Cell 7 muestra F1-macro ≥ 0.85
9. **Artefactos**: Existen `traffic_classifier.keras`, `feature_scaler.joblib`, `label_mapping.joblib` en `models/intelligence/`

---

## Límites Arquitectónicos (NO MODIFICAR sin ADR)

1. **Todo el código de Etapa 1 vive en `01_legacy_collection.ipynb`** — NO crear módulos `.py` separados
2. **`VAAETHybrid` es la única clase de Etapa 1** — NO subdividir en múltiples clases
3. **Cell 5 (`BRIDGE_CONFIG`) es el ÚNICO punto de configuración** para parámetros del puente
4. **Cell 1 es el ÚNICO punto de configuración** de base de datos en Etapa 1
5. **La fusión de velocidad es 70% física + 30% MLP** — no alterar sin evidencia experimental
6. **Los criterios de `is_stationary()` usan AND-conjunction** — no relajar a OR sin ADR
7. **El formato de nombre de archivo es estricto**: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`
8. **`01_legacy_collection.ipynb` es INTOCABLE** — no modificar el notebook de Etapa 1
9. **Las tablas `telemetry_raw` y `traffic_classifications` requieren ADR-008** — no modificar esquema sin nuevo ADR

---

## Patrones de Manejo de Errores

- **Degradación silenciosa**: Si la BD falla, el sistema continúa sin persistencia. Si optical flow falla, usa solo el cálculo físico
- **Try/except con emoji**: Todas las excepciones se capturan y se imprimen con prefijo emoji (🔴 error, ✅ éxito, ⚠️ warning). NO se propagan excepciones
- **Sin logging formal**: Se usa `print()` con emojis, no el módulo `logging` de Python
- **Filtros de plausibilidad**: Velocidades fuera de rango [2, 120] km/h se descartan silenciosamente

---

## Sistema de Gobernanza: Always / Ask / Never

### ✅ Always (hacer sin supervisión)

- Corregir errores de sintaxis Python
- Actualizar docstrings y comentarios
- Agregar type hints a funciones existentes
- Agregar markdown cells narrativas al notebook
- Formatear código (PEP 8)
- Actualizar documentación que esté desactualizada respecto al código
- Corregir inconsistencias entre `BRIDGE_CONFIG` y `VAAETHybrid.__init__`

### 🟡 Ask (requiere aprobación humana)

- Cambiar parámetros de calibración (`BRIDGE_CONFIG`, `bridge_calibration`, `perspective_zones`)
- Modificar la lógica de `calculate_enhanced_speed()` o `is_stationary()`
- Agregar nuevas dependencias al proyecto
- Cambiar el esquema de la tabla `traffic_data`
- Modificar la lógica de selección de modelo YOLO (`select_optimal_model()`)
- Refactorizaciones que afecten más del 20% de una celda
- Cambiar umbrales de confianza o NMS
- Cambiar umbrales de auto-labeling de estados de tráfico (Etapa 2)
- Modificar arquitectura del MLP/LSTM clasificador (Etapa 2)

### 🔴 Never (restricciones absolutas)

- Hardcodear credenciales de AWS RDS (host, password, etc.)
- Crear archivos `.py` fuera del notebook
- Eliminar `test_sistema()` o el generador de demos sintéticas
- Modificar la tabla `traffic_data` sin redactar un ADR previo
- Romper compatibilidad con Google Colab Free Tier
- Eliminar la validación estricta de nombre de archivo
- Hacer commit de archivos `.pt` (modelos YOLO) al repositorio
- Imprimir credenciales en outputs de celdas
- Modificar `telemetry_raw` o `traffic_classifications` sin ADR
- Eliminar campos HITL de `traffic_classifications`
- Hacer commit de archivos `.keras`, `.joblib` o CSVs de `data/processed/`

---

## Criterios de Parada y Handoff

El agente DEBE detenerse y solicitar intervención humana cuando:

1. **Quiere cambiar el modelo de detección** (ej: reemplazar YOLO 11 por RT-DETR)
2. **Quiere modificar la fórmula de fusión 70/30** de velocidad
3. **Quiere alterar los criterios ultra-conservadores** de `is_stationary()`
4. **Quiere cambiar la estructura de la tabla** de PostgreSQL
5. **Detecta inconsistencias entre la documentación y el código** que no puede resolver con certeza
6. **Un cambio requiere acceso a AWS RDS** o credenciales reales
7. **Un cambio afecta el rendimiento en Colab Free** (ej: aumentar resolución de inferencia)

---

## Decisiones Arquitectónicas Clave

Consultar los ADRs en `Docs/adr/` antes de proponer cambios que contradigan estas decisiones:

| ADR | Decisión |
|---|---|
| ADR-001 | Notebook monolítico sobre módulos Python |
| ADR-002 | YOLO 11 con selección adaptativa por duración |
| ADR-003 | SORT sobre DeepSORT/ByteTrack |
| ADR-004 | MLP como suavizador (no estimador primario) |
| ADR-005 | PostgreSQL (AWS RDS) sobre SQLite/local |
| ADR-006 | Detección de estacionarios ultra-conservadora |
| ADR-007 | Google Colab como entorno de ejecución principal |
| ADR-008 | TF/Keras para clasificación de tráfico + dos tablas + auto-labeling |

---

## Contexto del Dominio

- **Puente**: General Manuel Belgrano, 1700m longitud, 8.3m calzada
- **Cámaras**: SISE dinámicas a 60m altura, con zoom, paneo, visión nocturna
- **Tipos de vehículo**: car, truck, bus, motorcycle, bicycle
- **Velocidades típicas**: 40-80 km/h flujo normal, 0-20 km/h congestión
- **Persistencia**: Un registro por minuto con velocidad promedio y conteos por tipo

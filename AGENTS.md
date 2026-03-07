# AGENTS.md — Contexto Agéntico para VAAET

> Este archivo es la memoria a largo plazo para agentes de IA que operen sobre este repositorio.
> Léelo completo antes de realizar cualquier modificación al proyecto.

---

## Identidad del Proyecto

**VAAET** (Video Análisis Avanzado de Tráfico) es un sistema de visión por computadora para analizar tráfico vehicular en el **Puente General Manuel Belgrano** (Corrientes, Argentina). Detecta, clasifica, rastrea y estima la velocidad de vehículos usando video de cámaras de vigilancia SISE.

### Arquitectura Fundamental

- **Notebook monolítico**: Todo el código vive en `vaaet.ipynb` — un único Jupyter Notebook con 9 celdas de código
- **Entorno de ejecución**: Google Colab (acceso a GPU gratuita). NO hay servidor, API REST, microservicios ni contenedores
- **Persistencia**: PostgreSQL en AWS RDS (opcional). No hay SQLite, CSV ni JSON de salida
- **No hay CI/CD clásico**: No hay pipeline de build, no hay deploy. El "deploy" es abrir el notebook en Colab y ejecutar las celdas

### Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Detección de objetos | YOLO 11 (Ultralytics) — 5 variantes por duración |
| Visión por computadora | OpenCV — video I/O, optical flow, anotaciones |
| Cómputo numérico | NumPy — operaciones vectoriales, estadísticas |
| ML (suavizado) | scikit-learn `MLPRegressor` — NO es una CNN real |
| Base de datos | PostgreSQL via `psycopg2-binary` (AWS RDS) |
| Runtime | Google Colab (primario) o Python 3.8+ local |

---

## Estructura del Proyecto

```
vaaet/
├── vaaet.ipynb          # Todo el código (9 celdas)
├── README.md            # Visión general y uso
├── AGENTS.md            # Este archivo
├── CONTRIBUTING.md      # Guía de contribución
├── CHANGELOG.md         # Historial de cambios
├── LICENSE              # MIT
├── requirements.txt     # Dependencias pinneadas
├── llms.txt             # Índice para agentes RAG
├── llms-full.txt        # Documentación completa para LLMs
├── .gitignore
└── Docs/
    ├── PRD.md           # Requisitos del producto
    ├── DDS.md           # Diseño de software
    ├── GUIA_USUARIO.md  # Guía de usuario
    ├── DATA_LINEAGE.md  # Linaje de datos
    ├── BIAS_AND_LIMITATIONS.md  # Sesgos y limitaciones
    ├── KPIs/
    │   └── KPIs.md      # Métricas y validación
    ├── adr/             # Registros de Decisiones Arquitectónicas
    │   ├── ADR-001-notebook-monolitico.md
    │   ├── ADR-002-yolo11-seleccion-adaptativa.md
    │   ├── ADR-003-sort-sobre-deepsort.md
    │   ├── ADR-004-mlp-como-suavizador.md
    │   ├── ADR-005-postgresql-aws-rds.md
    │   ├── ADR-006-deteccion-estacionarios-conservadora.md
    │   └── ADR-007-google-colab-como-runtime.md
    └── diagrams/
        ├── pipeline-flow.md
        ├── speed-calculation.md
        ├── erd.md
        ├── colab-aws-architecture.md
        ├── model-selection.md
        └── multi-camera-layout.md
```

---

## Orden de Ejecución de Celdas

Las celdas del notebook DEBEN ejecutarse en este orden secuencial:

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

---

## Contrato de Validación

No hay build ni CI/CD. El equivalente funcional es:

1. **Smoke test**: Ejecutar Cell 2 sin errores de importación
2. **Test funcional**: Ejecutar `test_sistema()` en Cell 7 — verifica que todos los componentes se inicializan correctamente
3. **Test end-to-end**: Generar un video sintético (Cells 8-9) y verificar que produce un video de salida anotado
4. **Test de BD**: Si hay acceso a AWS RDS, verificar que `save_to_database()` persiste un registro y no expone credenciales

---

## Límites Arquitectónicos (NO MODIFICAR sin ADR)

1. **Todo el código vive en `vaaet.ipynb`** — NO crear módulos `.py` separados
2. **`VAAETHybrid` es la única clase del sistema** — NO subdividir en múltiples clases
3. **Cell 5 (`BRIDGE_CONFIG`) es el ÚNICO punto de configuración** para parámetros del puente
4. **Cell 1 es el ÚNICO punto de configuración** de base de datos
5. **La fusión de velocidad es 70% física + 30% MLP** — no alterar sin evidencia experimental
6. **Los criterios de `is_stationary()` usan AND-conjunction** — no relajar a OR sin ADR
7. **El formato de nombre de archivo es estricto**: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`

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

### 🔴 Never (restricciones absolutas)

- Hardcodear credenciales de AWS RDS (host, password, etc.)
- Crear archivos `.py` fuera del notebook
- Eliminar `test_sistema()` o el generador de demos sintéticas
- Modificar la tabla `traffic_data` sin redactar un ADR previo
- Romper compatibilidad con Google Colab Free Tier
- Eliminar la validación estricta de nombre de archivo
- Hacer commit de archivos `.pt` (modelos YOLO) al repositorio
- Imprimir credenciales en outputs de celdas

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

---

## Contexto del Dominio

- **Puente**: General Manuel Belgrano, 1700m longitud, 8.3m calzada
- **Cámaras**: SISE dinámicas a 60m altura, con zoom, paneo, visión nocturna
- **Tipos de vehículo**: car, truck, bus, motorcycle, bicycle
- **Velocidades típicas**: 40-80 km/h flujo normal, 0-20 km/h congestión
- **Persistencia**: Un registro por minuto con velocidad promedio y conteos por tipo

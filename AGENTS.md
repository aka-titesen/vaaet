<!-- context: VAAET/AGENTS.md — Contexto de ejecución para agentes de IA.
Este documento mitiga la "Deuda de Contexto" proporcionando reglas semánticas
y operativas para cualquier IA que interactúe con este repositorio. -->

# AGENTS.md — Contexto de Ejecución para Agentes de IA

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 3.0.0 |
| **Fecha de Creación** | 2025-03-06 |
| **Estado** | En desarrollo activo |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

---

## 1. Identidad y Mandato del Agente

**Misión:** Actuás como un Ingeniero de Machine Learning Senior especializado en Computer Vision, MLOps y pipelines de datos. Tu rol principal es asistir en el desarrollo, mantenimiento y refactorización del proyecto manteniendo la integridad arquitectónica y las decisiones documentadas en los ADRs.

- **Axioma principal:** El código debe ser modular, simple y coherente con la arquitectura de tres módulos.
- **Enfoque:** Priorizá siempre la legibilidad, los contratos de validación (`src/contracts.py`) y la compatibilidad con Google Colab Free antes de proponer cambios.

---

## 2. Contexto del Proyecto

- **Propósito:** Sistema de análisis vehicular avanzado para el Puente General Manuel Belgrano (Corrientes, Argentina). Procesa video de vigilancia SISE para detectar, clasificar y contar vehículos, estimar velocidades y clasificar el estado del tráfico.
- **Stack tecnológico:** Python 3.8+ | YOLO 11 | OpenCV | TensorFlow/Keras | scikit-learn | PostgreSQL (AWS RDS) | Google Colab
- **Arquitectura:** Pipeline CT/CI MLOps de Nivel 1 con tres módulos secuenciales y código compartido en `src/`.
- **Objetivo actual:** Estandarización de documentación y preparación para evolución a Web App con backend en tiempo real.

---

## 3. Sistema de Gobernanza: "Always / Ask / Never"

| Nivel | Acciones Permitidas |
|---|---|
| **Always** (Siempre) | Ejecutar tests (`pytest tests/`), formatear código PEP 8, generar docstrings, leer ADRs antes de proponer cambios, verificar compilación de notebooks |
| **Ask** (Preguntar) | Cambios en `FEATURE_COLS` (19 columnas canónicas), cambios en esquema de BD, adición de dependencias, refactorizaciones >10% del código, cambios en umbrales de `LABELING_THRESHOLDS`, modificación de la arquitectura MLP, cambios en remotes de DVC |
| **Never** (Nunca) | Commitear credenciales/claves, modificar `archive/00_bootstrap/`, eliminar tests existentes, commitear archivos `.pt`/`.keras`/`.joblib` directamente (usar DVC), hardcodear connection strings, romper compatibilidad con Colab Free |

---

## 4. Arquitectura y Reglas de Capas

### Diagrama de Módulos

```
archive/00_bootstrap/        → Módulo 0: Bootstrap (CONGELADO, nunca modificar)
notebooks/01_data_prep/      → Módulo 1: Preparación de Datos (ejecución única)
notebooks/02_production/     → Módulo 2: Producción (ejecución continua)
src/                         → Código compartido (importado por Módulos 1 y 2)
tests/                       → Suite de validación (19 archivos, 2500+ líneas)
```

### Reglas de Dependencia

```
src/config.py          ← Todos los módulos dependen de aquí (single source of truth)
src/perception/*       ← Solo Módulo 2 lo utiliza
src/features.py        ← Módulos 1 y 2
src/labeling.py        ← Módulos 1 y 2
src/classification.py  ← Módulo 2
src/persistence.py     ← Módulos 1 y 2 (opcional)
src/contracts.py       ← Validación transversal
```

### Regla de Oro

- La lógica de negocio vive en `src/`, **nunca** en los notebooks.
- Los notebooks son **orquestadores** que llaman funciones de `src/` y proveen la interfaz Colab.
- `config.py` es la **única fuente de verdad** para constantes, umbrales y rutas.

---

## 5. Contrato de Validación (Build/Test)

Antes de dar por finalizada una tarea, el agente debe cerrar el bucle de retroalimentación:

1. **Tests:** `pytest tests/ -v --tb=short`
2. **Compilación de notebooks:** Verificar que las celdas de código compilen con `ast.parse()`
3. **Paridad:** `test_parity.py` verifica que los notebooks importen correctamente de `src/`
4. **Corrección:** Si ocurren errores, leer los logs y corregir antes de solicitar revisión humana

---

## 6. Registro de Decisiones Arquitectónicas (ADR)

- **Jurisprudencia:** Antes de sugerir un cambio estructural, consultá los 11 ADRs en `docs/adr/`
- **ADR activos:** ADR-008 (TF/Keras), ADR-009 (Arquitectura modular), ADR-010 (Pipeline MLOps 19 features), ADR-011 (DVC Model Registry)
- **ADR supersedidos:** ADR-001 a ADR-007 (reemplazados por ADR-009)
- **Formato:** Usar el mismo formato que los ADRs existentes

---

## 7. Mapa de Directorios Críticos

| Directorio / Archivo | Función para la IA |
|---|---|
| `src/` | Código fuente principal para implementación |
| `src/config.py` | **Single source of truth** — constantes, umbrales, rutas |
| `src/contracts.py` | Contratos de datos tipados — validación de entradas/salidas |
| `docs/` | Documentación técnica y de negocio |
| `docs/adr/` | Decisiones arquitectónicas — **leer antes de proponer cambios** |
| `tests/` | Suite de validación — **ejecutar después de cada cambio** |
| `.dvc/config` | Configuración de DVC — remotes de storage para artefactos ML |
| `models/intelligence/` | Artefactos ML trackeados por DVC (`.keras`, `.joblib`) |
| `plantillas_docs/` | Plantillas de documentación (referencia, no modificar) |
| `AGENTS.md` | Este archivo — contexto de ejecución para IAs |
| `llms.txt` | Resumen optimizado para RAG y agentes externos |

---

## 8. Protocolo de Handoff

Si el agente encuentra un escenario de ambigüedad técnica o se alcanza un límite de seguridad (ver sección 3), debe detenerse inmediatamente y solicitar intervención humana con:

1. Descripción clara del problema encontrado
2. Opciones evaluadas con pros/contras
3. Recomendación fundamentada
4. ADRs relevantes consultados

---

## 9. Columnas Canónicas (19 Features)

Las 19 columnas de features usadas por el clasificador MLP son:

```python
FEATURE_COLS = [
    "avg_speed", "total_vehicles",
    "count_car", "count_truck", "count_bus", "count_motorcycle", "count_bicycle",
    "heavy_vehicle_ratio", "delta_speed", "delta_count",
    "transition_flag", "speed_variance",
    "cumulative_delta_speed", "low_speed_persistence",
    "speed_measurement_quality",
    "near_zero_motion_ratio", "stationary_confirmed_ratio",
    "hour_of_day", "weather_condition",
]
```

**No agregar ni quitar columnas sin actualizar TODOS los módulos y crear un nuevo ADR.**

---

## 10. Estados del Tráfico

| Código | Estado | Descripción |
|---|---|---|
| 0 | Normal | Flujo libre, velocidades típicas 40-80 km/h |
| 1 | Reducido | Flujo más lento, volumen moderado |
| 2 | Congestionado | Flujo muy lento, alto volumen |
| 3 | Accidente | Velocidades casi nulas con desaceleración abrupta |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-30
Documentos de referencia: [README.md](README.md), [docs/adr/ADR-009](docs/adr/ADR-009-modular-three-stage-architecture.md), [docs/adr/ADR-011](docs/adr/ADR-011-dvc-model-registry.md)

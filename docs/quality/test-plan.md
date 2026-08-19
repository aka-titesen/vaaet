<!-- context: VAAET/docs/quality/test-plan.md — Plan de Pruebas.
Complementa SRS.md (requisitos) y SAD.md (arquitectura). -->

# Plan de Pruebas (Test Plan) — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 4.5.1 |
| **Estado** | Aprobado |
| **Responsable Técnico** | Facundo Nicolás González |
| **Responsable de QA** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

---

## 1. Objetivos Principales

- Verificar la corrección de los módulos compartidos en `src/` mediante tests unitarios y de integración
- Garantizar la paridad entre los notebooks y el código compartido
- Validar la compilación de todas las celdas de código de los notebooks activos
- Establecer una red de seguridad automática integrada en el pipeline CI de GitHub Actions

---

## 2. Niveles y Estrategia de Pruebas

Se adopta el modelo del "Testing Trophy", con énfasis en tests de integración para validar la interacción entre los módulos `src/`.

| Nivel de Prueba | Esfuerzo Estimado | Enfoque Principal |
|---|---|---|
| **Unitarias** | 40% | Lógica pura de cada módulo en `src/` |
| **Integración** | 40% | Interacción entre módulos (features → labeling → classification) |
| **Paridad** | 10% | Sincronía notebooks ↔ `src/` |
| **Compilación** | 10% | Validación sintáctica de celdas de notebooks |

### 2.1 Pruebas Unitarias

Verifican el funcionamiento aislado de cada módulo en `src/`.

- **Herramienta:** pytest 7.4+
- **Ubicación:** `tests/test_*.py` (junto al módulo que testean)
- **Cobertura actual:** 20 archivos Python de soporte y pruebas

| Archivo de Test | Módulo Testeado | Cobertura |
|---|---|---|
| `tests/unit/test_settings.py` | `src/vaaet/settings.py` | Constantes, umbrales, rutas |
| `tests/contracts/test_contracts.py` | `src/vaaet/contracts.py` | Validación de contratos de datos |
| `tests/unit/test_engineering.py` | `src/vaaet/features/engineering.py` | Feature engineering de 19 columnas |
| `tests/unit/test_labeling.py` | `src/vaaet/features/labeling.py` | Auto-etiquetado de 4 estados |
| `tests/unit/test_traffic_state.py` | `src/vaaet/inference/traffic_state.py` | Tres salidas, histéresis, candidato de incidente y override humano |
| `tests/unit/test_database.py` | `src/vaaet/data/database.py` | Factory de engine, credenciales |
| `tests/unit/test_persistence.py` | `src/vaaet/data/persistence.py` | Persistencia en BD |
| `tests/unit/test_vision.py` | `src/vaaet/vision/` | Pipeline de percepción completo |
| `tests/unit/test_optical_flow.py` | `src/vaaet/vision/optical_flow.py` | Estimador de flujo óptico |
| `tests/unit/test_telemetry.py` | `src/vaaet/vision/telemetry.py` | Extracción de telemetría |
| `tests/unit/test_analysis.py` | `src/vaaet/vision/analysis.py` | Video anotado con/sin clasificación |
| `tests/unit/test_calibration.py` | `src/vaaet/evaluation/calibration.py` | Calibración de velocidad |
| `tests/unit/test_datasets.py` | `src/vaaet/data/datasets.py` | Carga y validación de datos |
| `tests/unit/test_synthetic.py` | `src/vaaet/features/synthetic.py` | Generación de datos sintéticos |
| `tests/unit/test_video.py` | `src/vaaet/vision/video.py` | Utilidades de video I/O |
| `tests/unit/test_reporting.py` | `src/vaaet/evaluation/reporting.py` | Reportes y visualizaciones |
| `tests/unit/test_human_holdout.py` | `src/vaaet/training/holdout.py` | Snapshot, checksums, versiones, idempotencia y leakage |
| `tests/repository/test_notebook_parity.py` | Notebooks ↔ paquete | Paridad de código |
| `tests/repository/test_repository_structure.py` | Repositorio | Higiene, rutas y enlaces |

### 2.2 Pruebas de Integración

Validan la interacción entre módulos sin mocks.

- **Herramienta:** pytest con fixtures compartidos en `conftest.py`
- **Entorno:** DataFrames sintéticos generados por `src/vaaet/features/synthetic.py`
- **Mocks:** para unidades puras; schemas, migración, vistas y grants se prueban con PostgreSQL 17 real en CI

### 2.3 Pruebas de Paridad

`test_parity.py` verifica que las funciones importadas en los notebooks coincidan con las implementaciones en `src/`.

### 2.4 Validación Manual (Smoke Test)

- **Entorno:** Google Colab Free con GPU T4
- **Frecuencia:** Antes de cada release mayor
- **Alcance:** Ejecución end-to-end de ambos notebooks activos

---

## 3. Stack Tecnológico y Herramientas

| Ámbito | Herramienta | Propósito |
|---|---|---|
| **Runner** | pytest 7.4+ | Ejecutor de tests |
| **Cobertura** | pytest-cov 4.1+ | Reportes de cobertura |
| **Fixtures** | conftest.py | Datos de prueba compartidos |
| **CI** | GitHub Actions | Ejecución automática en PRs |
| **Notebooks** | ast.parse() | Validación sintáctica de celdas |

---

## 4. Criterios de Éxito y Calidad

- **Tests unitarios:** Todos deben pasar (`pytest tests/ -v`)
- **Notebooks:** Todas las celdas de código deben compilar sin errores de sintaxis
- **CI/CD:** Ningún merge a `main` si los tests fallan en GitHub Actions
- **Métrica del modelo:** F1-macro ≥ 0.85 en el clasificador MLP

---

## 5. Ejecución de Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v --tb=short

# Con cobertura
pytest tests/ --cov=src --cov-report=html

# Solo un módulo específico
pytest tests/test_features.py -v
```

---

## 6. Gestión de Riesgos en Pruebas

| Riesgo | Mitigación |
|---|---|
| Tests que dependen de GPU | Todos los tests de `src/` pueden ejecutarse en CPU |
| Tests que dependen de BD PostgreSQL | Mocks en `test_db.py` y `test_persistence.py` |
| Videos reales no disponibles | DataFrames sintéticos y generador en `src/vaaet/features/synthetic.py` |
| Notebooks no ejecutables en CI | Validación sintáctica con `ast.parse()` como proxy |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23

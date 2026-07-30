<!-- context: VAAET/docs/TEST_PLAN.md — Plan de Pruebas.
Complementa SRS.md (requisitos) y SAD.md (arquitectura). -->

# Plan de Pruebas (Test Plan) — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 3.0.0 |
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
- **Cobertura actual:** 19 archivos de test, ~2.556 líneas

| Archivo de Test | Módulo Testeado | Cobertura |
|---|---|---|
| `test_config.py` | `src/config.py` | Constantes, umbrales, rutas |
| `test_contracts.py` | `src/contracts.py` | Validación de contratos de datos |
| `test_features.py` | `src/features.py` | Feature engineering de 19 columnas |
| `test_labeling.py` | `src/labeling.py` | Auto-etiquetado de 4 estados |
| `test_classification.py` | `src/classification.py` | Clasificación y gate de accidentes |
| `test_db.py` | `src/db.py` | Factory de engine, credenciales |
| `test_persistence.py` | `src/persistence.py` | Persistencia en BD |
| `test_perception.py` | `src/perception/` | Pipeline de percepción completo |
| `test_optical_flow.py` | `src/perception/optical_flow.py` | Estimador de flujo óptico |
| `test_pipeline.py` | `src/perception/pipeline.py` | Extracción de telemetría |
| `test_calibration.py` | `src/calibration.py` | Calibración de velocidad |
| `test_dataset.py` | `src/dataset.py` | Carga y validación de datos |
| `test_synthetic.py` | `src/synthetic.py` | Generación de datos sintéticos |
| `test_video.py` | `src/video.py` | Utilidades de video I/O |
| `test_reporting.py` | `src/reporting.py` | Reportes y visualizaciones |
| `test_parity.py` | Notebooks ↔ src/ | Paridad de código |
| `test_repo_hygiene.py` | Repositorio | Higiene general del repo |

### 2.2 Pruebas de Integración

Validan la interacción entre módulos sin mocks.

- **Herramienta:** pytest con fixtures compartidos en `conftest.py`
- **Entorno:** DataFrames sintéticos generados por `src/synthetic.py`
- **Mocks:** Solo para servicios externos (AWS RDS, Ultralytics Hub)

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
| Videos reales no disponibles | DataFrames sintéticos y generador de datos en `src/synthetic.py` |
| Notebooks no ejecutables en CI | Validación sintáctica con `ast.parse()` como proxy |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23

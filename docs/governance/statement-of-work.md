<!-- context: VAAET/docs/governance/statement-of-work.md — Declaración de Trabajo.
Preparado para uso futuro en contexto corporativo/contractual. -->

# Declaración de Trabajo (SOW) — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 1.0.0 |
| **Estado** | Borrador (para uso futuro) |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

---

## 1. Descripción del Trabajo

### 1.1 Resumen Ejecutivo

VAAET es un sistema de análisis vehicular avanzado que procesa video de vigilancia para el Puente General Manuel Belgrano (Corrientes, Argentina). El sistema detecta vehículos, estima velocidades y clasifica el estado del tráfico en tiempo real mediante técnicas de visión artificial y aprendizaje automático.

### 1.2 Alcance

| Dentro del Alcance | Fuera del Alcance |
|---|---|
| Pipeline de detección con YOLO 11 | Instalación física de cámaras |
| Estimación de velocidad por visión artificial | Análisis de patentes individuales |
| Clasificación de estados del tráfico (4 estados) | Predicción de tráfico futuro |
| Persistencia en PostgreSQL | Infraestructura de red del puente |
| Documentación técnica completa | Soporte 24/7 de operaciones |
| Pipeline de re-entrenamiento | Aplicación móvil |

---

## 2. Entregables

| # | Entregable | Descripción | Formato | Criterio de Aceptación |
|---|---|---|---|---|
| E-01 | Pipeline de percepción | YOLO 11 + SORT + estimación de velocidad | Notebooks + `src/` | Procesa video de 1h en < 30min (GPU T4) |
| E-02 | Clasificador de tráfico | MLP entrenado con 19 features | `.keras` + `.joblib` | F1-macro ≥ 0.85 |
| E-03 | Persistencia de datos | Telemetría en PostgreSQL | DDL + `vaaet-ml/src/vaaet_ml/data/persistence.py` | Upsert idempotente funcional |
| E-04 | Documentación completa | 25+ documentos técnicos y de negocio | Markdown | 100% de plantillas cubiertas |
| E-05 | Suite de tests | 20 archivos Python de soporte y tests | pytest | 100% pass rate en CI |
| E-06 | CI/CD | GitHub Actions pipeline | YAML | Tests automáticos en PRs |

---

## 3. Cronograma y Hitos

| Hito | Entregable | Fecha | Estado |
|---|---|---|---|
| M1: Percepción | E-01 (Pipeline YOLO + velocidad) | 2025-03-15 | ✅ Completado |
| M2: Inteligencia | E-02 (Clasificador MLP) | 2025-07-14 | ✅ Completado |
| M3: Persistencia | E-03 (PostgreSQL + upsert) | 2025-07-14 | ✅ Completado |
| M4: Testing | E-05 (Suite de 19 tests) | 2026-06-30 | ✅ Completado |
| M5: Documentación | E-04 (25+ documentos) | 2026-07-23 | ✅ Completado |
| M6: CI/CD | E-06 (GitHub Actions) | 2026-07-23 | ✅ Completado |
| M7: Web App MVP | Dashboard web (futuro) | TBD | 📋 Planificado |

---

## 4. Recursos y Responsabilidades

### 4.1 Equipo

| Rol | Persona | Dedicación |
|---|---|---|
| Desarrollador principal | Facundo Nicolás González | 100% |
| Proveedor de datos | SISE / Vialidad Nacional | Según disponibilidad |

### 4.2 Infraestructura Requerida

| Recurso | Proveedor | Costo |
|---|---|---|
| GPU T4/V100 | Google Colab Free | $0 |
| PostgreSQL | Proveedor compatible, tamaño inicial pequeño | Variable |
| Repositorio | GitHub (público) | $0 |
| CI/CD | GitHub Actions (gratuito) | $0 |

---

## 5. Condiciones y Supuestos

1. Los videos de vigilancia SISE están disponibles bajo autorización del ente correspondiente
2. Google Colab Free proporciona acceso a GPU de forma razonable para el procesamiento
3. El endpoint PostgreSQL elegido es accesible y estable durante las sesiones de Colab
4. Los umbrales de auto-etiquetado son calibrados para el Puente Belgrano específicamente
5. Congested sintético sólo aumenta train; Accident permanece fuera del MLP y exige confirmación humana

---

## 6. Firma de Aceptación

| Parte | Nombre | Fecha | Firma |
|---|---|---|---|
| **Proveedor** | Facundo Nicolás González | _____________ | _____________ |
| **Cliente** | __________________ | _____________ | _____________ |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23

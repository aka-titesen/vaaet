<!-- context: VAAET/docs/INDEX.md — Índice maestro de documentación.
Punto de entrada para navegar toda la documentación del proyecto. -->

# Índice Maestro de Documentación — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 3.0.0 |
| **Total de Documentos** | 30+ |
| **Última Revisión** | 2026-07-23 |

---

## Documentos en Raíz del Proyecto

| Documento | Descripción | Público |
|---|---|---|
| [README.md](../README.md) | Visión general, arquitectura, inicio rápido | Todos |
| [AGENTS.md](../AGENTS.md) | Contexto de ejecución para agentes de IA | IA |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Guía de contribución y convenciones | Técnico |
| [CHANGELOG.md](../CHANGELOG.md) | Historial de cambios (SemVer) | Todos |
| [SECURITY.md](../SECURITY.md) | Política de seguridad y reporte de vulnerabilidades | Todos |
| [SUPPORT.md](../SUPPORT.md) | Canales de soporte | Todos |
| [LICENSE](../LICENSE) | Licencia MIT | Todos |
| [llms.txt](../llms.txt) | Resumen optimizado para RAG y LLMs | IA |

---

## Documentación Técnica (`docs/`)

### Requisitos y Producto

| Documento | Descripción | Plantilla Base |
|---|---|---|
| [PRD.md](PRD.md) | Requisitos del producto con historias de usuario | ✅ plantilla_PRD |
| [SRS.md](SRS.md) | Especificación de requisitos de software (IEEE 830) | ✅ plantilla_SRS |
| [USE_CASES.md](USE_CASES.md) | Casos de uso del negocio (7 CUN) | ✅ plantilla_CUN |
| [USER_PERSONAS.md](USER_PERSONAS.md) | Perfiles de usuario (4 personas) | ✅ plantilla_personas |

### Arquitectura y Diseño

| Documento | Descripción | Plantilla Base |
|---|---|---|
| [SAD.md](SAD.md) | Arquitectura de software (ex DDS.md) | ✅ plantilla_SAD |
| [DATA_MODEL.md](DATA_MODEL.md) | Modelo de datos y diccionario (3 tablas) | ✅ plantilla_modelo_datos |
| [DATA_LINEAGE.md](DATA_LINEAGE.md) | Linaje de datos end-to-end | Personalizado |

### Machine Learning

| Documento | Descripción | Plantilla Base |
|---|---|---|
| [MODEL_CARD.md](MODEL_CARD.md) | Model Card estilo HuggingFace (MLP clasificador) | HuggingFace |
| [BIAS_AND_LIMITATIONS.md](BIAS_AND_LIMITATIONS.md) | Sesgos y limitaciones del sistema | Personalizado |

### Calidad y Validación

| Documento | Descripción | Plantilla Base |
|---|---|---|
| [TEST_PLAN.md](TEST_PLAN.md) | Plan de pruebas (19 archivos, 2.556 líneas) | ✅ plantilla_plan_pruebas |
| [KPIs/KPIs.md](KPIs/KPIs.md) | Métricas de rendimiento y guía de validación | Personalizado |

### Operaciones y Despliegue

| Documento | Descripción | Plantilla Base |
|---|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | Guía de usuario para operadores e investigadores | Personalizado |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Manual de despliegue (DevOps Playbook) | ✅ plantilla_manual_despliegue |
| [DVC_GUIDE.md](DVC_GUIDE.md) | Guía de DVC para versionado de modelos | Personalizado |

### Gestión y Estrategia

| Documento | Descripción | Plantilla Base |
|---|---|---|
| [FEASIBILITY.md](FEASIBILITY.md) | Estudio de factibilidad (técnica, económica, legal) | ✅ plantilla_factibilidad |
| [RISK_MATRIX.md](RISK_MATRIX.md) | Matriz de riesgos (10 riesgos identificados) | ✅ plantilla_riesgos |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | Plan de gestión del proyecto | ✅ plantilla_plan_gestion |
| [BUSINESS_CANVAS.md](BUSINESS_CANVAS.md) | Business Model Canvas (visión comercial) | ✅ plantilla_canvas |

### Seguridad y Legal

| Documento | Descripción | Plantilla Base |
|---|---|---|
| [SECURITY_POLICY.md](SECURITY_POLICY.md) | Política de seguridad y privacidad detallada | ✅ plantilla_seguridad |
| [NDA.md](NDA.md) | Acuerdo de confidencialidad (borrador futuro) | ✅ plantilla_NDA |
| [SOW.md](SOW.md) | Declaración de trabajo (borrador futuro) | ✅ plantilla_SOW |

---

## Decisiones Arquitectónicas (`docs/adr/`)

| ADR | Título | Estado |
|---|---|---|
| [ADR-001](adr/ADR-001-notebook-monolitico.md) | Notebook monolítico | Supersedido (ADR-009) |
| [ADR-002](adr/ADR-002-yolo11-seleccion-adaptativa.md) | Selección adaptativa de YOLO 11 | Aceptado |
| [ADR-003](adr/ADR-003-sort-sobre-deepsort.md) | SORT sobre DeepSORT | Aceptado |
| [ADR-004](adr/ADR-004-mlp-como-suavizador.md) | MLP como suavizador de velocidad | Aceptado (histórico) |
| [ADR-005](adr/ADR-005-postgresql-aws-rds.md) | PostgreSQL en AWS RDS | Aceptado |
| [ADR-006](adr/ADR-006-deteccion-estacionarios-conservadora.md) | Detección conservadora de estacionarios | Aceptado |
| [ADR-007](adr/ADR-007-google-colab-como-runtime.md) | Google Colab como runtime | Aceptado |
| [ADR-008](adr/ADR-008-tensorflow-keras-traffic-classifier.md) | TensorFlow/Keras para clasificación | Aceptado |
| [ADR-009](adr/ADR-009-modular-three-stage-architecture.md) | Arquitectura modular de tres módulos | Aceptado |
| [ADR-010](adr/ADR-010-mlops-pipeline-19-features.md) | Pipeline MLOps con 19 features | Aceptado |
| [ADR-011](adr/ADR-011-dvc-model-registry.md) | DVC como Model Registry | Aceptado |

---

## Diagramas (`docs/diagrams/`)

| Diagrama | Descripción |
|---|---|
| [pipeline-flow.md](diagrams/pipeline-flow.md) | Flujo del pipeline completo |
| [speed-calculation.md](diagrams/speed-calculation.md) | Cálculo de velocidad |
| [model-selection.md](diagrams/model-selection.md) | Selección de modelo YOLO |
| [colab-aws-architecture.md](diagrams/colab-aws-architecture.md) | Arquitectura Colab ↔ AWS |
| [intelligence-pipeline.md](diagrams/intelligence-pipeline.md) | Pipeline de inteligencia |
| [erd.md](diagrams/erd.md) | Diagrama entidad-relación (original) |
| [erd-phase2.md](diagrams/erd-phase2.md) | ERD extendido (3 tablas) |
| [multi-camera-layout.md](diagrams/multi-camera-layout.md) | Layout multi-cámara |

---

## Infraestructura del Proyecto

| Archivo | Descripción |
|---|---|
| [pyproject.toml](../pyproject.toml) | Configuración de paquete Python |
| [requirements.txt](../requirements.txt) | Dependencias (compatibilidad) |
| [.env.example](../.env.example) | Template de variables de entorno |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | Pipeline CI con GitHub Actions (tests + DVC check) |
| [.dvc/config](../.dvc/config) | Configuración de DVC — remotes de storage |
| [.gitignore](../.gitignore) | Archivos excluidos de Git |

---

## Mapa de Navegación por Público

### 🧑‍💻 Técnico (Desarrollador / Contribuidor)

```
README.md → CONTRIBUTING.md → docs/SAD.md → docs/SRS.md → docs/adr/
              ↓
         AGENTS.md (si es IA)
```

### 📊 No Técnico (Operador SISE / Ingeniero Municipal)

```
README.md → docs/USER_GUIDE.md → docs/PRD.md → docs/USER_PERSONAS.md
```

### 🤖 IA (Agentes / RAG)

```
AGENTS.md → llms.txt → src/config.py → docs/adr/ → docs/SAD.md
```

### 📋 Gestión (Administrativo / Académico)

```
README.md → docs/PRD.md → docs/FEASIBILITY.md → docs/PROJECT_PLAN.md → docs/SOW.md
```

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-30

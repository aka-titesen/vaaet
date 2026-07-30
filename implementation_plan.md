# Análisis Integral y Plan de Documentación — VAAET

## Resumen del Análisis

Tras examinar exhaustivamente las **3.707 líneas de código fuente** (`src/`), **2.556 líneas de tests** (19 archivos), **3 notebooks**, **9 ADRs**, **8 diagramas**, **5 documentos técnicos**, y las **24 plantillas** en `plantillas_docs/`, presento el análisis completo y el plan de acción.

---

## 1. Clasificación Formal de la Arquitectura

### Nombre formal: **CT/CI Pipeline — Continuous Training / Continuous Inference**

Tu arquitectura se clasifica formalmente como un **MLOps Pipeline de Nivel 1** según la taxonomía de Google (MLOps Maturity Model), específicamente un **pipeline de Entrenamiento Continuo con Inferencia Continua** (CT/CI). Las tres fases que describes mapean directamente a este patrón:

| Tu Fase | Nombre Formal MLOps | Patrón |
|---|---|---|
| **Fase A** (archive) | Data Collection / Feature Store Bootstrap | One-shot ingestion |
| **Fase B** (01_data_prep) | Training Pipeline | Feature Engineering → Training → Model Registry |
| **Fase C** (02_production) | Inference Pipeline + Feedback Loop | Online inference → Monitoring → CT trigger |

El **bucle de mejora continua** que describes (producción → datos → re-entrenamiento → producción) es el componente central que eleva esto de un simple "train-once-deploy" a un sistema CT/CI. En la literatura MLOps se conoce como:

- **Google**: "MLOps Level 1: ML Pipeline Automation"
- **Microsoft**: "Continuous Training Pipeline"
- **AWS**: "MLOps Foundation — Automated Retraining"

### ¿Se ajusta el repositorio actual a este alcance?

**Parcialmente sí, con brechas importantes:**

| Criterio | Estado | Detalle |
|---|---|---|
| Separación de fases | ✅ Bien | 3 módulos claros (archive, data_prep, production) |
| Código compartido | ✅ Bien | `src/` con alta cohesión y bajo acoplamiento |
| Feature Engineering reutilizable | ✅ Bien | `src/features.py` es single source of truth |
| Contratos de datos | ✅ Bien | `src/contracts.py` con dataclasses validados |
| Persistencia del modelo | ✅ Bien | `.keras` + `.joblib` exportables |
| Feedback loop | ⚠️ Parcial | HITL scaffold existe pero es experimental |
| Model Registry | ❌ Ausente | No hay versionado formal de modelos |
| Data Versioning | ❌ Ausente | No hay DVC/LakeFS/equivalente |
| Experiment Tracking | ❌ Ausente | No hay MLflow/W&B/equivalente |
| CI/CD Pipeline | ❌ Ausente | No hay GitHub Actions/workflows |
| Monitoring / Drift Detection | ❌ Ausente | No hay monitoreo de model drift |

---

## 2. Estrategia de Repositorio: Monorepo vs Multi-repo

### Recomendación: **Monorepo con separación lógica por capas** (fase actual) → **Multi-repo cuando el backend web sea productivo**

#### Justificación para Monorepo AHORA:

1. **Desarrollador único**: No hay fricción de ownership entre equipos. Un monorepo simplifica el workflow de un solo desarrollador.
2. **Acoplamiento real**: `src/` es compartido entre notebooks y el futuro backend. Mantenerlo junto evita drift de versiones.
3. **Atomicidad de cambios**: Si cambias `FEATURE_COLS` en `config.py`, necesitas actualizar notebooks y tests en el mismo commit.
4. **Overhead mínimo**: Un multi-repo con 1 desarrollador añade complejidad de sincronización sin beneficio.

#### Justificación para Multi-repo DESPUÉS (cuando el backend web sea real):

1. **Ciclos de release distintos**: El entorno de experimentación (notebooks) tiene releases "cuando termina el entrenamiento". El backend web tiene releases "cuando hay un PR merged". Mezclarlos en un mono-repo genera ruido.
2. **Dependencias divergentes**: Los notebooks necesitan `tensorflow`, `ultralytics`, `imbalanced-learn`. El backend web necesita `FastAPI`, `uvicorn`, `redis`. Mezclarlas en un solo `requirements.txt` es insostenible.
3. **CI/CD diferenciado**: Los tests del pipeline ML (pesados, con GPU) no deberían bloquear un deploy del frontend web.

#### Estructura propuesta para la transición:

```
Fase actual (Monorepo):
vaaet/
├── src/            ← Lógica compartida (importable como paquete)
├── notebooks/      ← Experimentación/Training
├── archive/        ← Histórico
├── tests/          ← Tests unitarios de src/
└── Docs/           ← Documentación

Fase futura (Multi-repo):
vaaet-ml/           ← Experimentación + Training (este repo, renombrado)
├── src/
├── notebooks/
├── tests/
└── models/         ← Model registry local

vaaet-api/          ← Backend web (nuevo repo)
├── app/
├── src/inference/  ← Solo lo necesario de src/ para inferencia
├── models/         ← Artefactos descargados del registry
└── tests/
```

> [!IMPORTANT]
> Para la transición, `src/` se publicaría como un paquete Python interno (`vaaet-core`) que ambos repos importan. Esto elimina el copy-paste.

---

## 3. Análisis Crítico — Puntos Débiles y Mejoras

### 🔴 Puntos Críticos

| # | Problema | Impacto | Mejora Propuesta |
|---|---|---|---|
| 1 | **AGENTS.md, llms.txt, llms-full.txt referenciados en README pero NO existen** | Documentación rota, confusa para IA y humanos | Crear estos archivos o eliminar referencias |
| 2 | **Directorio `Docs/` con D mayúscula vs `docs/` referenciado en README y ADRs** | Inconsistencia de rutas; enlaces rotos en sistemas case-sensitive | Estandarizar a `docs/` (minúscula, convención universal) |
| 3 | **No hay `pyproject.toml` ni `setup.py`** | `src/` no es instalable como paquete; los notebooks hackean `sys.path` | Crear `pyproject.toml` con `[project]` y `[tool.pytest]` |
| 4 | **Sin CI/CD (GitHub Actions)** | No hay validación automática en PRs; los tests podrían romperse sin que nadie lo note | Crear `.github/workflows/ci.yml` |
| 5 | **FEATURE_COLS tiene 19 columnas pero DDS.md dice "9 → 14"** | Desincronización documentación ↔ código | Auditar y alinear; la documentación existente tiene datos desactualizados |

### 🟡 Puntos de Mejora

| # | Problema | Mejora |
|---|---|---|
| 6 | Sin `SECURITY.md` | Crear con política de reporte de vulnerabilidades |
| 7 | Sin `SUPPORT.md` | Crear con canales de soporte |
| 8 | Sin Model Card (estándar ML 2026) | Crear `Docs/MODEL_CARD.md` documentando el MLP |
| 9 | `requirements.txt` sin pins exactos | Crear `requirements-lock.txt` con versiones exactas |
| 10 | Sin `.env.example` | Crear template de variables de entorno |
| 11 | Directorio `models/` y `data/` referenciados pero no verificables | Crear `.gitkeep` + READMEs explicativos |
| 12 | `scripts/` sin documentación de uso | El README.md en scripts es escueto |
| 13 | Sin Makefile / task runner | Crear para estandarizar `make test`, `make lint`, etc. |
| 14 | DDS.md dice "14 features", config.py tiene 19 | Evolución no documentada en changelog |

### 🟢 Fortalezas Destacables

| Fortaleza | Detalle |
|---|---|
| Contratos tipados (`contracts.py`) | Excepcional para un proyecto académico. Previene drift silencioso. |
| 19 archivos de test | Cobertura amplia incluyendo `test_parity.py` para sincronía notebook↔src |
| 9 ADRs documentados | Excelente trazabilidad de decisiones arquitectónicas |
| Degradación silenciosa | El sistema funciona sin DB, sin GPU, con videos corruptos |
| Módulos `src/` desacoplados | Cada módulo tiene responsabilidad única (SOLID) |
| Generador de datos sintéticos | Solución elegante al problema de clases ausentes |

---

## 4. Plan de Documentación — Paso a Paso

### Inventario de Plantillas Disponibles (24)

A continuación mapeo cada plantilla a su estado actual y prioridad:

| # | Plantilla | ¿Existe Equivalente? | Prioridad | Acción |
|---|---|---|---|---|
| 1 | **README** | ✅ `README.md` | P0 | Auditar y alinear a plantilla |
| 2 | **AGENTS** | ❌ No existe | P0 | **Crear** — referenciado pero ausente |
| 3 | **Changelog** | ✅ `CHANGELOG.md` | P1 | Auditar formato y agregar sección Context Health |
| 4 | **PRD** | ✅ `Docs/PRD.md` | P1 | Auditar y completar secciones faltantes |
| 5 | **SAD (Arquitectura)** | ✅ `Docs/DDS.md` | P1 | Auditar, renombrar, y alinear a plantilla |
| 6 | **TDD/ADR** | ✅ `Docs/adr/` (9 ADRs) | P1 | Auditar y agregar ADR-010 para pipeline MLOps |
| 7 | **Modelo de Datos** | ⚠️ Parcial en DDS.md | P1 | **Crear** `Docs/DATA_MODEL.md` dedicado |
| 8 | **Plan de Pruebas** | ❌ No existe | P1 | **Crear** `Docs/TEST_PLAN.md` |
| 9 | **SRS (Especificación)** | ⚠️ Parcial en PRD.md | P1 | **Crear** `Docs/SRS.md` o integrar en PRD |
| 10 | **Manual de Despliegue** | ⚠️ Parcial en USER_GUIDE | P2 | **Crear** `Docs/DEPLOYMENT.md` |
| 11 | **Especificación completa** | ⚠️ Distribuida | P2 | Evaluar si se consolida o se mantiene distribuida |
| 12 | **SECURITY** | ❌ No existe | P1 | **Crear** `SECURITY.md` en raíz |
| 13 | **SUPPORT** | ❌ No existe | P2 | **Crear** `SUPPORT.md` en raíz |
| 14 | **LICENSE** | ✅ `LICENSE` | P3 | OK — MIT correcta |
| 15 | **Matriz de Riesgos** | ❌ No existe | P2 | **Crear** `Docs/RISK_MATRIX.md` |
| 16 | **Perfiles de Usuario** | ❌ No existe | P2 | **Crear** `Docs/USER_PERSONAS.md` |
| 17 | **Business Model Canvas** | ❌ No existe | P3 | **Crear** si aplica al contexto académico |
| 18 | **Casos de Uso** | ❌ No existe | P2 | **Crear** `Docs/USE_CASES.md` |
| 19 | **NDA** | ❌ No existe | P3 | Evaluar si aplica (proyecto MIT) |
| 20 | **SOW** | ❌ No existe | P3 | Evaluar si aplica (desarrollador único) |
| 21 | **Plan de Gestión** | ❌ No existe | P3 | Evaluar si aplica |
| 22 | **Factibilidad** | ❌ No existe | P2 | **Crear** `Docs/FEASIBILITY.md` |
| 23 | **Política de Seguridad** | ❌ No existe | P2 | **Crear** `Docs/SECURITY_POLICY.md` |
| 24 | **Protocolo de Soporte** | ❌ No existe | P3 | **Crear** `Docs/SUPPORT_PROTOCOL.md` |

### Plan de Ejecución por Fases

#### Fase 1 — Correcciones Críticas (P0) 🔴
> Documentos que están rotos o ausentes y causan confusión inmediata.

1. **Normalizar directorio `Docs/` → `docs/`** (convención universal)
2. **Crear `AGENTS.md`** usando la plantilla Agents — definir runtime context para IAs
3. **Crear `llms.txt`** — resumen optimizado para RAG/LLMs externos
4. **Auditar `README.md`** — alinear a plantilla README, corregir referencias rotas, actualizar feature count (14→19)
5. **Crear `.env.example`** con variables de entorno documentadas
6. **Crear `pyproject.toml`** para que `src/` sea importable como paquete

#### Fase 2 — Documentos Estratégicos (P1) 🟡
> Documentos que definen el "qué" y el "por qué" del proyecto.

7. **Auditar y completar `docs/PRD.md`** — alinear a plantilla PRD, agregar historias de usuario y métricas
8. **Auditar y evolucionar `docs/DDS.md`** — alinear a plantilla SAD, corregir "14 features" → "19 features"
9. **Crear `docs/SRS.md`** — Especificación de Requisitos formal (IEEE 830)
10. **Crear `docs/DATA_MODEL.md`** — Diccionario de datos y modelo relacional dedicado
11. **Crear `docs/TEST_PLAN.md`** — Plan de pruebas con la pirámide existente (19 test files)
12. **Crear `SECURITY.md`** en raíz — Política de reporte de vulnerabilidades
13. **Auditar `CHANGELOG.md`** — Agregar secciones de Context Health y Security
14. **Crear `docs/MODEL_CARD.md`** — Model Card estándar ML 2026 para el MLP
15. **Crear ADR-010** — Documentar la evolución a 19 features y pipeline MLOps

#### Fase 3 — Documentos de Contexto (P2) 🟢
> Documentos para públicos no técnicos y completitud.

16. **Crear `docs/USER_PERSONAS.md`** — Perfiles: Operador SISE, Investigador, Ingeniero de tráfico
17. **Crear `docs/USE_CASES.md`** — Casos de uso del negocio formales
18. **Crear `docs/RISK_MATRIX.md`** — Matriz de riesgos y mitigación
19. **Crear `docs/DEPLOYMENT.md`** — Manual de despliegue (Colab + futuro Docker)
20. **Crear `docs/FEASIBILITY.md`** — Estudio de factibilidad técnica
21. **Crear `docs/SECURITY_POLICY.md`** — Política de seguridad y privacidad detallada
22. **Crear `SUPPORT.md`** en raíz — Canales y protocolo de soporte

#### Fase 4 — Documentos Opcionales (P3) 🔵
> Evaluar relevancia según contexto académico.

23. **Evaluar `docs/BUSINESS_CANVAS.md`** — ¿Aplica a un proyecto académico?
24. **Evaluar `docs/NDA.md`** — Proyecto MIT, probablemente no aplica
25. **Evaluar `docs/SOW.md`** — Desarrollador único, evaluar relevancia
26. **Evaluar `docs/PROJECT_PLAN.md`** — Plan de gestión del proyecto

#### Fase 5 — Estandarización Final
> Verificación y alineamiento general.

27. **Crear índice maestro `docs/INDEX.md`** — Mapa de navegación para toda la documentación
28. **Actualizar todos los cross-references** entre documentos
29. **Crear `.github/workflows/docs-lint.yml`** — Verificación automática de enlaces
30. **Revisión final de coherencia** — Asegurar que todos los documentos usen la misma terminología

---

## 5. Preguntas Antes de Ejecutar

> [!IMPORTANT]
> Necesito tu confirmación en estos puntos antes de proceder:

### Preguntas de Alcance

1. **Idioma de la documentación**: La documentación existente está en inglés, las plantillas están en español. ¿Quieres que la documentación final sea en **español** (alineada a las plantillas) o en **inglés** (alineada al código)? ¿O bilingüe?

2. **Normalización de directorio**: ¿Confirmas renombrar `Docs/` → `docs/` (minúscula)? Esto requiere actualizar todas las referencias en el repo pero es la convención universal.

3. **Plantillas P3 (opcionales)**: Para un proyecto académico de desarrollador único, los documentos NDA, SOW, y Plan de Gestión son típicos de entornos corporativos. ¿Quieres crearlos igualmente por completitud, o los descartamos?

4. **Business Model Canvas**: ¿VAAET tiene una dimensión comercial/de negocio o es puramente académico? Esto determina si el Canvas aporta valor.

5. **Profundidad del Model Card**: ¿Quieres un Model Card básico (métricas, limitaciones, uso previsto) o uno completo estilo HuggingFace con análisis de fairness y carbon footprint?

### Preguntas Técnicas

6. **`pyproject.toml`**: ¿Quieres que configure `src/` como un paquete Python instalable (con `pip install -e .`)? Esto eliminará el hack de `sys.path` en notebooks.

7. **CI/CD**: ¿Ya usas GitHub o algún otro proveedor Git? Esto determina si creo workflows de GitHub Actions o equivalente.

8. **Model Registry**: Para el versionado de modelos, ¿prefieres una solución ligera (Google Drive + naming convention) o quieres que implemente algo como DVC o MLflow?

---

## Verificación

### Validación post-ejecución

- [ ] Todos los enlaces en README.md apuntan a archivos que existen
- [ ] Todos los documentos en `docs/` usan la terminología correcta (19 features, no 14)
- [ ] Las 24 plantillas tienen su documento equivalente o una justificación de exclusión
- [ ] Los tests pasan (`pytest tests/`)
- [ ] No hay archivos huérfanos referenciados pero inexistentes

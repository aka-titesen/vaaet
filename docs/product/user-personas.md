<!-- context: VAAET/docs/product/user-personas.md — Perfiles de usuario.
Complementa PRD.md y USE_CASES.md. -->

# Perfiles de Usuario (User Personas) — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 4.2.2 |
| **Estado** | Aprobado |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

---

## Persona 1: Operador SISE

| Atributo | Detalle |
|---|---|
| **Nombre representativo** | Carlos, 35 años |
| **Rol** | Operador del Sistema Inteligente de Seguridad (SISE) |
| **Formación** | Técnico en sistemas / Fuerza de seguridad |
| **Experiencia técnica** | Baja-Media. Usa aplicaciones de monitoreo, no programa. |
| **Objetivo principal** | Monitorear el estado del tráfico en tiempo real y detectar incidentes |
| **Frustraciones** | Videos largos sin poder ver rápidamente el estado del tráfico; faltas de alertas tempranas ante accidentes |
| **Necesidades** | Dashboard claro, alertas visuales, clasificación automática de estados, video anotado descargable |
| **Interacción con VAAET** | Usa inferencia en Colab: sube video → obtiene estado del tráfico + video anotado. En el futuro, usará la Web App. |
| **Métrica de éxito** | Tiempo de detección de incidentes < 2 minutos de procesamiento |

---

## Persona 2: Investigador de Tráfico

| Atributo | Detalle |
|---|---|
| **Nombre representativo** | Dra. Lucía, 42 años |
| **Rol** | Investigadora en ingeniería de transporte |
| **Formación** | Doctorado en ingeniería civil / transporte |
| **Experiencia técnica** | Media. Usa Python, pandas, notebooks. No es experta en ML. |
| **Objetivo principal** | Analizar patrones de tráfico históricos para publicaciones académicas |
| **Frustraciones** | Falta de datos cuantitativos sobre el puente; conteo manual es lento y propenso a errores |
| **Necesidades** | Datos tabulares exportables (CSV/BD), métricas por minuto, reproducibilidad de resultados |
| **Interacción con VAAET** | Usa adquisición, entrenamiento e inferencia para estudiar el pipeline completo |
| **Métrica de éxito** | Datos con granularidad por minuto y velocidades con MAE < 5 km/h |

---

## Persona 3: Ingeniero de Tráfico Municipal

| Atributo | Detalle |
|---|---|
| **Nombre representativo** | Martín, 38 años |
| **Rol** | Ingeniero de tráfico en la Municipalidad de Corrientes |
| **Formación** | Ingeniería civil |
| **Experiencia técnica** | Baja. Usa Excel, informes en PDF. |
| **Objetivo principal** | Planificar mejoras viales basadas en datos de volumen y velocidad |
| **Frustraciones** | Tomar decisiones de infraestructura sin datos objetivos sobre el flujo vehicular del puente |
| **Necesidades** | Reportes resumidos, gráficos de tendencias, conteos por tipo de vehículo, alertas de congestión |
| **Interacción con VAAET** | Recibe reportes generados por Persona 1 o 2. En el futuro, accede al dashboard de la Web App. |
| **Métrica de éxito** | Informes mensuales con datos de volumen y velocidad por franja horaria |

---

## Persona 4: Agente de IA (Copilot / Asistente)

| Atributo | Detalle |
|---|---|
| **Nombre representativo** | — (Agente de código) |
| **Rol** | Asistente de desarrollo basado en IA |
| **Formación** | Modelo de lenguaje entrenado en código |
| **Experiencia técnica** | Alta, pero sin contexto específico del proyecto |
| **Objetivo principal** | Asistir en la implementación, refactorización y documentación del código |
| **Frustraciones** | Deuda de contexto: no entiende la arquitectura sin documentación clara |
| **Necesidades** | `AGENTS.md`, `llms.txt`, ADRs actualizados, contratos de datos explícitos |
| **Interacción con VAAET** | Lee archivos de contexto → propone cambios → ejecuta tests → solicita revisión humana |
| **Métrica de éxito** | Cambios propuestos que pasan todos los tests y respetan los ADRs |

---

## Matriz Persona × Módulo

| Persona | Adquisición | Entrenamiento | Inferencia | Web App (futuro) |
|---|---|---|---|---|
| Operador SISE | ❌ | ❌ | ✅ (uso directo) | ✅ (uso directo) |
| Investigador | 📖 (referencia) | ✅ (uso directo) | ✅ (uso directo) | 📊 (consulta datos) |
| Ingeniero Municipal | ❌ | ❌ | 📊 (recibe reportes) | ✅ (uso directo) |
| Agente de IA | 📖 (referencia) | 🔧 (desarrollo) | 🔧 (desarrollo) | 🔧 (desarrollo) |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23

<!-- context: VAAET/docs/product/product-requirements.md — Requisitos del producto.
Complementa SAD.md (arquitectura) y KPIs/KPIs.md (métricas). -->

# Documento de Requisitos del Producto (PRD) — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 4.5.3 |
| **Fecha de Creación** | 2025-03-06 |
| **Estado** | Aprobado |
| **Responsable Técnico** | Facundo Nicolás González |
| **Fecha de Emisión** | 2026-07-23 |

---

## 1. Resumen Ejecutivo

### 1.1 Definición del Problema

El monitoreo del tráfico vehicular en el Puente Gral. Manuel Belgrano (Corrientes, Argentina) se realiza actualmente de forma manual por operadores SISE. Este método es lento, subjetivo y no genera datos cuantitativos que permitan análisis retrospectivo o planificación vial informada. Los accidentes y congestiones se detectan con demora, y no existe un registro histórico granular del flujo vehicular.

### 1.2 Solución Propuesta

VAAET procesa video de vigilancia para extraer telemetría por minuto y resolver tres estados estables. Mantiene cuatro categorías públicas porque Accident puede ser confirmado por una persona, pero nunca es una salida automática del MLP. Opera sobre Google Colab y PostgreSQL opcional.

### 1.3 Valor Diferenciador

- **Pipeline physics-first**: Estimación de velocidad basada en óptica computacional con compensación de movimiento, no en modelos de caja negra
- **Selección adaptativa de modelo**: 5 variantes de YOLO 11 seleccionadas automáticamente por duración del video
- **Detector conservador de posible incidente**: genera un candidato separado; nunca publica Accident automáticamente
- **19 features de calidad**: Incluyen señales de calidad de medición, no solo datos crudos
- **Costo mínimo**: Runtime gratuito (Colab Free), BD opcional ($15/mes)
- **Degradación silenciosa**: Funciona sin BD, sin GPU, y ante videos con frames corruptos

---

## 2. Objetivos y Métricas de Éxito (KPIs)

| Métrica | Definición | Objetivo |
|---|---|---|
| **F1-macro del clasificador** | F1-Score promedio de Normal, Reduced y Congested sobre holdout humano | ≥ 0.88 |
| **F1-Score de detección** | Precisión de detección vehicular YOLO 11 | ≥ 0.97 (objetivo declarado) |
| **MAE de velocidad** | Error absoluto medio de velocidad estimada | < 5 km/h (objetivo, sin benchmark) |
| **ID Switches de tracking** | Cambios de identidad por video | Minimizar (sin umbral formal) |
| **Disponibilidad del pipeline** | Capacidad de procesar un video exitosamente | > 95% (sesiones de Colab) |

Ver [KPIs](../quality/kpis.md) para la guía completa de validación.

---

## 3. Alcance del Producto

### 3.1 Funcionalidades Incluidas (En Alcance)

- **Percepción**: Detección YOLO 11 + tracking SORT + estimación de velocidad physics-first
- **Inteligencia**: Feature engineering v2 + MLP de tres estados + política jerárquica; cuatro estados públicos
- **Persistencia**: Telemetría y clasificaciones en PostgreSQL (opcional)
- **Visualización**: Video anotado con bounding boxes y HUD + dashboard de métricas
- **Datos sintéticos**: Congested sólo como augmentación de train; Accident sólo como estrés técnico
- **HITL**: feedback persistido en inferencia y consumido exclusivamente por entrenamiento

### 3.2 Fuera de Alcance

- Backend web o API REST (reservado para fase futura)
- Análisis de patentes individuales
- Predicción de tráfico futuro
- Soporte multi-idioma en la interfaz
- Alertas push en tiempo real

---

## 4. Historias de Usuario

### 4.1 Sistema de Priorización

- **P0 (Crítico)**: Bloquea el flujo principal. Sin esto, el sistema no es usable.
- **P1 (Alto)**: Flujo secundario de alto valor. Impacta a >30% de los usuarios.
- **P2 (Medio)**: Mejora la experiencia o la eficiencia.

### 4.2 Catálogo de Historias

#### US-001: Procesar Video de Tráfico — P0

**Como** operador SISE, **quiero** subir un video del puente y obtener un análisis automático, **para** no depender del conteo manual.

**Criterios de Aceptación:**
- **Dado** un video con formato `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`
- **Cuando** el operador lo procesa en el workflow de inferencia
- **Entonces** obtiene telemetría por minuto con conteos, velocidad promedio, y estado del tráfico

#### US-002: Clasificar Estado del Tráfico — P0

**Como** operador SISE, **quiero** conocer el estado del tráfico de cada minuto procesado, **para** detectar condiciones anómalas rápidamente.

**Criterios de Aceptación:**
- **Dado** telemetría procesada con 19 features
- **Cuando** el sistema clasifica cada registro
- **Entonces** asigna un estado estable 0–2; un posible incidente mantiene Congested y sólo un override humano validado produce Accident

#### US-003: Entrenar el Clasificador — P0

**Como** investigador, **quiero** entrenar el MLP con datos de telemetría, **para** clasificar tres estados estables y evaluar incidentes por separado.

**Criterios de Aceptación:**
- **Dado** telemetría raw con augmentación sintética trazable sólo en train
- **Cuando** el sistema completa el entrenamiento con class weights limitados, validation explícita y EarlyStopping
- **Entonces** el bundle sólo es promovible si cumple F1-macro ≥0,88 y todos los gates de soporte, calibración y procedencia

#### US-004: Persistir Resultados — P1

**Como** investigador, **quiero** que los resultados se guarden en una base de datos, **para** poder analizarlos históricamente.

**Criterios de Aceptación:**
- **Dado** credenciales de BD configuradas vía variables de entorno
- **Cuando** el sistema persiste telemetría y clasificaciones
- **Entonces** los registros se insertan idempotentemente en `vaaet_ml.telemetry_features` y `vaaet_ml.traffic_predictions`, sin alterar feedback

---

## 5. Requisitos Técnicos y No Funcionales

### 5.1 Arquitectura

- Pipeline CT/CI de MLOps Nivel 1 con 3 módulos secuenciales
- Código compartido en `src/` (13 módulos Python)
- Notebooks como orquestadores (Colab)
- Ver [arquitectura](../architecture/software-architecture.md) para el diseño completo

### 5.2 Seguridad

- Credenciales por variables de entorno exclusivamente
- Consultas SQL parametrizadas (prevención de inyección)
- No se procesan datos personales (patentes, identidades)

### 5.3 Compatibilidad

- Python 3.8+
- Google Colab Free (runtime principal)
- Google Colab Pro (compatible, no requerido)
- Desarrollo local (validación parcial)

---

## 6. Plan de Proyecto

| Hito | Entregable | Fecha | Estado |
|---|---|---|---|
| M1: Percepción | Pipeline YOLO 11 + SORT + velocidad | 2025-03-15 | ✅ |
| M2: Inteligencia | Clasificador MLP + auto-etiquetado | 2025-07-14 | ✅ |
| M3: Persistencia | PostgreSQL + upsert idempotente | 2025-07-14 | ✅ |
| M4: Testing | Suite ampliada a 20 archivos Python | 2026-06-30 | ✅ |
| M5: Documentación | 25+ documentos (estándares 2026) | 2026-07-23 | ✅ |
| M6: Web App MVP | Dashboard de tráfico | TBD | 📋 Planificado |

---

## 7. Análisis de Riesgos

Ver [matriz de riesgos](../quality/risk-matrix.md) para la matriz completa de 10 riesgos identificados.

Riesgos principales:
- **R-005**: Clases Accidente/Congestionado nunca observadas en datos reales (Severidad: Crítica)
- **R-001**: Desconexión de Colab durante procesamiento largo (Severidad: Crítica)
- **R-009**: Exposición accidental de credenciales (Severidad: Crítica)

---

## Entornos Compatibles

- **Google Colab Free**: Runtime principal y objetivo de diseño
- **Google Colab Pro**: Compatible, no requerido
- **Desarrollo local**: Entorno de validación parcial; la ejecución end-to-end en Colab sigue siendo validación manual

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23
Documentos de referencia: [arquitectura](../architecture/software-architecture.md), [SRS](software-requirements.md), [KPIs](../quality/kpis.md)

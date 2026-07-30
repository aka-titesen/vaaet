<!-- context: VAAET/docs/SRS.md — Especificación de Requisitos de Software.
Sigue el estándar IEEE Std 830-1998. Complementa PRD.md (requisitos de producto)
y SAD.md (arquitectura). -->

# Especificación de Requisitos de Software (SRS) — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 3.0.0 |
| **Fecha de Creación** | 2025-03-06 |
| **Estado** | Aprobado |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

## Ficha del Documento

| Fecha | Revisión | Autor | Verificado |
|---|---|---|---|
| 2025-03-06 | 1.0 | Facundo Nicolás González | — |
| 2025-07-14 | 2.0 | Facundo Nicolás González | — |
| 2026-07-23 | 3.0 | Facundo Nicolás González | — |

---

## 1. Introducción

### 1.1 Propósito

Este documento define los requisitos funcionales y no funcionales del sistema VAAET de forma técnica y unívoca, dirigido a desarrolladores, arquitectos y sistemas de IA que interactúen con el repositorio.

### 1.2 Alcance

VAAET es un sistema de análisis vehicular que procesa video de vigilancia del Puente Gral. Manuel Belgrano para detectar vehículos, estimar velocidades, y clasificar el estado del tráfico. El sistema opera como un pipeline CT/CI de MLOps ejecutado en Google Colab.

### 1.3 Personal Involucrado

| Nombre | Rol | Responsabilidades |
|---|---|---|
| Facundo Nicolás González | Desarrollador principal y arquitecto | Diseño, implementación, testing y documentación |

---

## 2. Descripción General

### 2.1 Perspectiva del Producto

VAAET es un sistema independiente que se integra con infraestructura SISE (Sistema Inteligente de Seguridad) existente. No reemplaza sistemas de vigilancia sino que agrega una capa de inteligencia analítica sobre el video capturado.

### 2.2 Funcionalidad del Producto

El sistema realiza cuatro funciones principales:
1. **Detección y clasificación** de vehículos en 5 categorías
2. **Estimación de velocidad** individual con compensación de movimiento de cámara
3. **Clasificación del estado del tráfico** en 4 estados
4. **Persistencia** de telemetría y clasificaciones en base de datos relacional

### 2.3 Restricciones, Suposiciones y Dependencias

- **Runtime**: Google Colab Free/Pro (GPU T4/V100 cuando disponible)
- **Red**: Conexión a internet para descarga de modelos YOLO y acceso a AWS RDS
- **Video**: Formato MP4, nomenclatura estricta `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`
- **BD**: PostgreSQL 12+ en AWS RDS (opcional, el sistema degrada sin ella)

---

## 3. Requisitos Específicos

### 3.1 Requisitos Funcionales

| ID | Requisito Funcional | Descripción Detallada | Prioridad |
|---|---|---|---|
| RF-001 | Validación de video | El sistema debe validar que el nombre del archivo cumpla el formato `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`. Archivos con formato inválido se rechazan con error descriptivo. | P0 |
| RF-002 | Selección adaptativa de modelo | El sistema debe seleccionar automáticamente la variante YOLO 11 (n/s/m/l/x) según la duración del video extraída del nombre de archivo. | P0 |
| RF-003 | Detección de vehículos | El sistema debe detectar vehículos en cada frame usando YOLO 11 con umbral de confianza ≥ 0.5 y NMS IoU ≤ 0.4, clasificándolos en: auto, camión, colectivo, motocicleta, bicicleta. | P0 |
| RF-004 | Tracking persistente | El sistema debe mantener IDs únicos por vehículo usando SORT con distancia euclidiana máxima de 100px y mismo tipo de vehículo. | P0 |
| RF-005 | Estimación de velocidad | El sistema debe estimar velocidad individual por vehículo usando: compensación de flujo óptico, corrección de perspectiva por zona Y, filtros de plausibilidad por tipo de vehículo, y agregación robusta por minuto. | P0 |
| RF-006 | Detección de estacionarios | El sistema debe clasificar vehículos como estacionarios mediante conjunción AND de 6 criterios estadísticos con histéresis de entrada/salida. | P1 |
| RF-007 | Feature engineering | El sistema debe transformar 9 campos crudos de telemetría en 19 features de calidad según `src/config.py:FEATURE_COLS`. | P0 |
| RF-008 | Clasificación de tráfico | El sistema debe clasificar cada minuto en uno de 4 estados (Normal, Reducido, Congestionado, Accidente) usando el MLP entrenado + gate conservador de accidentes. | P0 |
| RF-009 | Persistencia en BD | El sistema debe persistir telemetría y clasificaciones en PostgreSQL via `src/persistence.py` con upsert idempotente. Opcional — degradación silenciosa si no hay BD. | P1 |
| RF-010 | Video anotado | El sistema debe generar un video de salida con bounding boxes, tipo + ID, velocidad, y HUD informativo. | P1 |
| RF-011 | Soporte multi-cámara | El sistema debe detectar automáticamente layouts de 1, 2 o 4 cámaras y procesar cada ROI independientemente. | P1 |
| RF-012 | Auto-etiquetado | El sistema debe asignar etiquetas de estado del tráfico usando reglas de ingeniería calibradas al Puente Belgrano (ver `src/config.py:LABELING_THRESHOLDS`). | P0 |
| RF-013 | Generación de datos sintéticos | El sistema debe generar secuencias sintéticas de Accidente y Congestión para compensar la ausencia de estas clases en datos reales. IDs sintéticos ≥ 50.001. | P1 |
| RF-014 | Scaffold HITL | El Módulo 2 debe incluir una celda experimental para correcciones humanas y re-entrenamiento, sin ser parte del flujo validado. | P2 |

**Detalle técnico del requisito RF-005 (Estimación de velocidad):**

- **Entradas:** Historial de centroides del track, FPS del video, altura del frame, vector de movimiento global (flujo óptico), tipo de vehículo
- **Secuencia de operaciones:** Acumulación de desplazamiento → compensación de movimiento de cámara → corrección de perspectiva por zona → conversión píxeles/metro → filtro de plausibilidad por tipo → agregación robusta (trim 15%, outlier 3.5σ)
- **Salidas:** Velocidad suavizada en km/h o `None` si la medición no es confiable

### 3.2 Requisitos No Funcionales (RNF)

#### 3.2.1 Rendimiento y Eficiencia

- El sistema debe procesar video en tiempo razonable en Google Colab Free (GPU T4)
- Frame skipping y memory cleanup deben estar implementados para videos largos
- La persistencia en BD no debe bloquear el procesamiento de frames

#### 3.2.2 Seguridad y Privacidad

- Las credenciales de BD deben obtenerse exclusivamente por variables de entorno o `getpass`
- Ningún secreto debe ser impreso en outputs de celdas
- El sistema no debe extraer, almacenar ni procesar patentes individuales
- Solo datos agregados por minuto se persisten en BD

#### 3.2.3 Fiabilidad y Disponibilidad

- El sistema debe continuar procesando ante fallo de BD (degradación silenciosa)
- El sistema debe continuar ante frames corruptos (skip al siguiente)
- El sistema debe continuar ante ausencia de detecciones (usar promedios históricos)
- El sistema debe continuar ante fallo de flujo óptico (usar velocidad sin compensación)

#### 3.2.4 Mantenibilidad y Portabilidad

- Todo código compartido debe residir en `src/` con type hints y docstrings
- Los notebooks deben ser orquestadores que importan funciones de `src/`
- El sistema debe ser compatible con Python 3.8+ y Google Colab Free/Pro
- `config.py` debe ser la única fuente de verdad para constantes y umbrales

---

## 4. Apéndices

### 4.1 Glosario

| Término | Definición |
|---|---|
| **CT/CI** | Continuous Training / Continuous Inference — patrón MLOps |
| **HITL** | Human-in-the-Loop — validación humana de clasificaciones |
| **SORT** | Simple Online and Realtime Tracking — algoritmo de tracking |
| **NMS** | Non-Maximum Suppression — eliminación de detecciones duplicadas |
| **Gate conservador** | Regla heurística que anula la predicción del modelo cuando la evidencia de accidente es fuerte |
| **Feature engineering** | Transformación de campos crudos en variables predictivas de calidad |
| **SMOTE** | Synthetic Minority Over-sampling Technique — balanceo de clases |

### 4.2 Referencias

- [PRD.md](PRD.md) — Requisitos del producto
- [SAD.md](SAD.md) — Arquitectura de software
- [DATA_MODEL.md](DATA_MODEL.md) — Modelo de datos
- [docs/adr/](adr/) — Architecture Decision Records

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23

<!-- context: VAAET/docs/SAD.md — Documento de Arquitectura de Software.
Reemplaza al antiguo DDS.md. Complementa SRS.md y DATA_MODEL.md. -->

# Documento de Arquitectura de Software (SAD) — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 3.0.0 |
| **Fecha de Creación** | 2025-03-06 |
| **Estado** | Aprobado |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

---

## 1. Introducción y Alcance

### 1.1 Propósito

Este documento define la arquitectura técnica del sistema VAAET, documentando las decisiones de diseño, los patrones utilizados, y la topología de los componentes. Está dirigido a desarrolladores, arquitectos, y agentes de IA que necesiten comprender la estructura interna del sistema.

### 1.2 Alcance del Sistema

VAAET es un pipeline CT/CI (Continuous Training / Continuous Inference) de MLOps Nivel 1 que procesa video de vigilancia para detectar vehículos, estimar velocidades, y clasificar el estado del tráfico. Opera como notebooks de Google Colab que orquestan módulos Python compartidos.

---

## 2. Arquitectura y Diseño de Sistemas

### 2.1 Paradigmas y Patrones

- **Arquitectura global**: Pipeline CT/CI MLOps con tres módulos secuenciales
- **Principios de codificación**: SOLID, KISS, YAGNI, DRY
- **Patrón de capas**:
  - **Capa de orquestación**: Notebooks (`.ipynb`) — interfaz Colab, flujo de ejecución
  - **Capa de dominio**: `src/` — lógica de negocio, feature engineering, clasificación
  - **Capa de percepción**: `src/perception/` — detección, tracking, estimación de velocidad
  - **Capa de persistencia**: `src/persistence.py` + `src/db.py` — PostgreSQL opcional

### 2.2 Diagrama de Arquitectura

```mermaid
flowchart TB
    subgraph "Capa de Orquestación (Notebooks)"
        M1[Módulo 1: data_preparation.ipynb]
        M2[Módulo 2: traffic_analyzer.ipynb]
    end

    subgraph "Capa de Dominio (src/)"
        CFG[config.py — Fuente única de verdad]
        FEAT[features.py — Feature engineering 19 cols]
        LAB[labeling.py — Auto-etiquetado 4 estados]
        CLAS[classification.py — Inferencia MLP + gate]
        CONT[contracts.py — Contratos de datos]
        SYN[synthetic.py — Datos sintéticos]
        CAL[calibration.py — Calibración del puente]
    end

    subgraph "Capa de Percepción (src/perception/)"
        DET[detector.py — YOLODetector]
        TRK[tracker.py — SORTTracker]
        SPD[speed.py — Velocidad physics-first]
        OPF[optical_flow.py — Flujo óptico]
        PIP[pipeline.py — Pipeline de telemetría]
    end

    subgraph "Capa de Persistencia"
        DB[db.py — Engine factory]
        PER[persistence.py — Upsert idempotente]
        RDS[(PostgreSQL AWS RDS)]
    end

    M1 --> CFG
    M1 --> FEAT
    M1 --> LAB
    M1 --> SYN
    M1 --> PER

    M2 --> CFG
    M2 --> PIP
    M2 --> CLAS
    M2 --> PER

    PIP --> DET
    PIP --> TRK
    PIP --> SPD
    PIP --> OPF

    CLAS --> FEAT
    CLAS --> LAB

    PER --> DB
    DB --> RDS

    CONT -.->|valida| FEAT
    CONT -.->|valida| PER
    CONT -.->|valida| CLAS
```

### 2.3 Reglas de Dependencia

```
src/config.py          ← Todos los módulos dependen (single source of truth)
src/perception/*       ← Solo Módulo 2
src/features.py        ← Módulos 1 y 2
src/labeling.py        ← Módulos 1 y 2
src/classification.py  ← Módulo 2
src/persistence.py     ← Módulos 1 y 2 (opcional)
src/contracts.py       ← Validación transversal
src/synthetic.py       ← Solo Módulo 1
```

---

## 3. Datos y Persistencia

### 3.1 Estrategia de Persistencia

| Componente | Tipo / Motor | Justificación Técnica |
|---|---|---|
| Almacenamiento principal | PostgreSQL 12+ (AWS RDS) | Consistencia ACID para telemetría temporal con integridad referencial |
| Almacenamiento de modelos | Google Drive / local | Artefactos `.keras` y `.joblib` exportados por Módulo 1 |
| Almacenamiento de video | Ephemeral (Colab) | Videos no se persisten — solo datos agregados por minuto |

### 3.2 Esquema de Base de Datos

Ver [DATA_MODEL.md](DATA_MODEL.md) para el diccionario completo de 3 tablas.

**Resiliencia**: El sistema usa `CREATE TABLE IF NOT EXISTS` como mecanismo de migración. No hay ORM — consultas SQL directas con parámetros vía SQLAlchemy `text()`.

---

## 4. Flujos de Datos

### 4.1 Flujo de Producción (Módulo 2)

```mermaid
sequenceDiagram
    participant U as Usuario (Colab)
    participant P as Pipeline de Percepción
    participant F as Feature Engineering
    participant C as Clasificador MLP
    participant G as Gate de Accidentes
    participant DB as PostgreSQL (opcional)

    U->>P: Sube video .mp4
    P->>P: YOLO 11 → detecciones por frame
    P->>P: SORT → tracks persistentes
    P->>P: Velocidad → physics-first con compensación
    P->>U: DataFrame telemetría (9 campos × N minutos)
    U->>F: Ejecuta feature engineering
    F->>F: 9 campos → 19 features
    F->>C: Features escaladas
    C->>C: MLP predict → estado + confianza
    C->>G: Predicción + features
    G->>G: Evalúa evidencia de accidente
    G->>U: Estado final + confianza + metadata
    U->>DB: Persiste (opcional)
    DB-->>U: Confirmación o degradación silenciosa
```

### 4.2 Flujo de Entrenamiento (Módulo 1)

```mermaid
sequenceDiagram
    participant U as Usuario (Colab)
    participant BD as traffic_data (BD)
    participant SYN as Generador Sintético
    participant F as Feature Engineering
    participant L as Auto-etiquetado
    participant SM as SMOTE
    participant MLP as Entrenamiento MLP

    U->>BD: SELECT telemetría cruda
    BD-->>U: ~2.000 registros
    U->>SYN: Inyectar accidentes + congestión
    SYN-->>U: +200 registros sintéticos
    U->>F: Feature engineering (9 → 19)
    F-->>U: DataFrame con 19 features
    U->>L: Auto-etiquetado (4 estados)
    L-->>U: DataFrame etiquetado
    U->>SM: SMOTE en train set
    SM-->>MLP: Train set balanceado
    MLP-->>U: Artefactos .keras + .joblib
```

---

## 5. Seguridad y Operaciones

### 5.1 Autenticación y Credenciales

- Credenciales de BD por variables de entorno exclusivamente
- En Colab, se usa `getpass` para entrada segura
- No hay autenticación de usuarios (runtime de notebook individual)

### 5.2 Resiliencia

- **Degradación silenciosa**: Si la BD no está disponible, el pipeline continúa sin persistencia
- **Frame skip**: Frames corruptos se saltan sin detener el procesamiento
- **Fallback de velocidad**: Sin flujo óptico, se usa velocidad sin compensación de cámara

### 5.3 CI/CD

- **GitHub Actions**: Tests automáticos en Python 3.10/3.11/3.12
- **Compilación de notebooks**: Verificación sintáctica con `ast.parse()`
- **Verificación de enlaces**: Detección de enlaces rotos en documentación

---

## 6. Registro de Decisiones Arquitectónicas (ADR)

| ID | Decisión | Contexto / Justificación | Estado |
|---|---|---|---|
| ADR-001 | Notebook monolítico | Diseño inicial para una sola fase | Supersedido por ADR-009 |
| ADR-002 | Selección adaptativa de YOLO 11 | 5 variantes por duración de video | Aceptado |
| ADR-003 | SORT sobre DeepSORT | Menor consumo de recursos en Colab | Aceptado |
| ADR-004 | MLP como suavizador de velocidad | Regularización hacia la media | Aceptado (histórico) |
| ADR-005 | PostgreSQL en AWS RDS | Persistencia relacional ACID | Aceptado |
| ADR-006 | Detección conservadora de estacionarios | AND de 6 criterios con histéresis | Aceptado |
| ADR-007 | Google Colab como runtime | Costo cero, GPU gratuita | Aceptado |
| ADR-008 | TensorFlow/Keras para clasificador | Ecosistema maduro para Colab | Aceptado |
| ADR-009 | Arquitectura modular de tres módulos | Desacoplamiento, testabilidad, SOLID | Aceptado |
| ADR-010 | Pipeline MLOps con 19 features | Señales de calidad, contratos, proveniencia | Aceptado |

Ver carpeta completa: [docs/adr/](adr/)

---

## 7. Apéndices y Referencias

| Referencia | Título | Enlace |
|---|---|---|
| REF-01 | Especificación de Requisitos (SRS) | [SRS.md](SRS.md) |
| REF-02 | Modelo de Datos | [DATA_MODEL.md](DATA_MODEL.md) |
| REF-03 | Linaje de Datos | [DATA_LINEAGE.md](DATA_LINEAGE.md) |
| REF-04 | Plan de Pruebas | [TEST_PLAN.md](TEST_PLAN.md) |
| REF-05 | Model Card | [MODEL_CARD.md](MODEL_CARD.md) |
| REF-06 | Sesgos y Limitaciones | [BIAS_AND_LIMITATIONS.md](BIAS_AND_LIMITATIONS.md) |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23

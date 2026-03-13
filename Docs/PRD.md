<!-- context: VAAET/docs/PRD.md — Product requirements for the VAAET system.
Complements DDS.md (technical design) and KPIs.md (performance metrics). -->

# PRD — VAAET: Advanced Vehicular Traffic Analysis System

## Executive Summary

VAAET is an academic vehicular traffic analysis system for the General Manuel Belgrano Bridge. The active implementation combines YOLO 11 detection, SORT tracking, optical-flow-assisted camera-motion compensation, physics-first speed estimation, and a TF/Keras traffic-state classifier with a conservative accident gate.

The system is optimized for Google Colab Free, keeps notebooks as orchestrators, and pushes reusable logic into `src/`. It aims for defensible academic rigor rather than production-infrastructure completeness.

## Current Implementation Status

- `archive/00_bootstrap/` is historical only and no longer part of the runtime path
- The active speed path is physics-first with optical-flow compensation, plausibility filters, and robust per-minute aggregation
- Optional MLP speed fusion still exists in code as a dormant capability, but it is not wired by default in the active telemetry pipeline
- HITL/retraining remains an experimental notebook scaffold, not a validated operational feedback loop
- Notebook code compiles and parity tests pass, but end-to-end Google Colab execution is still validated manually

---

## Unique Value Proposition

- **Robust academic pipeline**: Combines YOLO 11 detection, Optical Flow, conservative stationary handling, and a traffic-state classifier without overengineering the Colab workflow
- **Dynamic model selection**: Automatically selects the optimal YOLO 11 model by video duration
- **Secure persistence**: Integration with PostgreSQL AWS RDS without exposing credentials
- **Clear outputs**: Processed video with overlays, metrics, and minimalist informational panel
- **Colab optimization**: Adapted to Google Colab Free/Pro resources and limitations
- **Dynamic context handling**: Supports camera changes, zoom, multi-view, and real bridge conditions
- **Modularity and scalability**: Shared `src/` modules, compact notebooks, clear outputs

---

## Three-Module Architecture

### Module 0 — Bootstrap (ARCHIVED)

Historical YOLO 11 + OpenCV + SORT pipeline (`archive/00_bootstrap/01_legacy_collection.ipynb`) that generated the original `traffic_data` table. **Never runs again.**

### Module 1 — Data Preparation (one-time)

`notebooks/01_data_prep/data_preparation.ipynb` — Quality-aware feature engineering from legacy telemetry into a 19-feature classifier dataset, auto-labeling of 4 traffic states, SMOTE balancing, MLP training, and model export.

### Module 2 — Production (ongoing)

`notebooks/02_production/traffic_analyzer.ipynb` — YOLO 11 + SORT + physics-first speed estimation + trained MLP classifier + optional DB persistence + experimental HITL scaffold.

---

## Requirements Compliance Summary

1. **Video upload**: Only accepts files with format `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`
2. **Automatic YOLO 11 selection**: Chooses between yolo11x/l/m/s/n by duration (<1h, 1-3h, 3-6h, 6-12h, >12h)
3. **Optional persistence**: User decides whether to persist to PostgreSQL AWS RDS; valid data each minute; no hardcoded credentials
4. **Speed calculation**: Uses a physics-first pipeline with Optical Flow compensation, plausibility filters, conservative stationary handling, and robust per-minute aggregation; optional MLP fusion is not part of the default runtime path
5. **Multi-camera and perspective**: Automatic layout detection (1, 2, 4 views); calibrable homography; adapts to camera and zoom changes
6. **Robust tracking**: Persistent tracking (SORT), unique IDs, exclusion of out-of-frame vehicles
7. **Historical and outputs**: Informational panel and persistence use recent averages when no readings; processed video with overlays and automatic download
8. **Colab optimization**: Frame skipping, memory cleanup, support for free/pro environments
9. **Modularity and robustness**: Shared `src/` modules, auxiliary functions, error handling
10. **Clear outputs**: Concise outputs, success/error messages with emoji prefixes
11. **Secure credential management**: Environment variables only; never exposes sensitive data
12. **Aligned database**: Persistence in 3 tables per required schema
13. **Dynamic context**: Adapts to mobile cameras, zoom, variable angles, and real bridge conditions
14. **Traffic state classification**: TF/Keras MLP classifies each minute into 4 states from 19 engineered features plus a conservative accident gate

---

## YOLO 11 Detection Model

- **Models**: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
- **Selection by duration**: ≤1h→x, 1-3h→l, 3-6h→m, 6-12h→s, >12h→n
- **Auto-diagnosis**: Automatic download of missing weights

## Dependencies

- **Core**: numpy, pandas, sqlalchemy, psycopg2-binary, joblib
- **Data Preparation**: tensorflow, scikit-learn, imbalanced-learn, matplotlib, seaborn
- **Production**: ultralytics, opencv-python, tensorflow, scikit-learn

## PostgreSQL Persistence (optional)

- **Frequency**: One record per minute (avg_speed, counts by type, total)
- **Schema**: 3 tables — `traffic_data` (legacy), `telemetry_raw`, `traffic_classifications`
- **Security**: Environment variables; no credentials exposed; prompt if missing

---

## Use Cases

- SISE operational monitoring
- Traffic engineering
- Urban planning
- Academic research
- Emergency management

---

## KPIs

See [KPIs/KPIs.md](KPIs/KPIs.md) for detailed metrics and validation guide.

- Detection precision, speed accuracy, stationary detection, processing efficiency
- Traffic classification: F1-macro ≥ 0.85, recall per class

---

## Compatible Environments

- **Google Colab Free**: Primary runtime and design target
- **Google Colab Pro**: Compatible but not required
- **Local development**: Partial validation environment; full runtime behavior still needs notebook-specific smoke tests

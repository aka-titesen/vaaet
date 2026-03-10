<!-- context: VAAET/docs/PRD.md — Product requirements for the VAAET system.
Complements DDS.md (technical design) and KPIs.md (performance metrics). -->

# PRD — VAAET: Advanced Vehicular Traffic Analysis System

## Executive Summary

VAAET is a hybrid artificial intelligence system for vehicular traffic analysis on the General Manuel Belgrano Bridge, integrating YOLO 11 detection, Optical Flow, MLP smoothing, and secure PostgreSQL persistence. It resolves historical issues of prior systems such as erroneous speed assignment to stationary vehicles, inconsistent classification, and lack of robustness against dynamic SISE camera conditions.

The system is optimized for Google Colab, is modular, scalable, secure, and fulfills all functional and quality requirements defined by stakeholders.

---

## Unique Value Proposition

- **Robust hybrid system**: Combines YOLO 11 detection, Optical Flow, and MLP for maximum precision and stationary exclusion
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

`notebooks/01_data_prep/data_preparation.ipynb` — Feature engineering from `traffic_data` (9 → 14 columns), auto-labeling of 4 traffic states, SMOTE balancing, MLP training, model export.

### Module 2 — Production (ongoing)

`notebooks/02_production/traffic_analyzer.ipynb` — YOLO 11 + SORT + speed estimation + trained MLP classifier + DB persistence + self-improving HITL feedback loop.

---

## Requirements Compliance Summary

1. **Video upload**: Only accepts files with format `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`
2. **Automatic YOLO 11 selection**: Chooses between yolo11x/l/m/s/n by duration (<1h, 1-3h, 3-6h, 6-12h, >12h)
3. **Optional persistence**: User decides whether to persist to PostgreSQL AWS RDS; valid data each minute; no hardcoded credentials
4. **Hybrid speed calculation**: Combines physics, Optical Flow Farneback, and MLP smoothing; stationary exclusion; limits by type
5. **Multi-camera and perspective**: Automatic layout detection (1, 2, 4 views); calibrable homography; adapts to camera and zoom changes
6. **Robust tracking**: Persistent tracking (SORT), unique IDs, exclusion of out-of-frame vehicles
7. **Historical and outputs**: Informational panel and persistence use recent averages when no readings; processed video with overlays and automatic download
8. **Colab optimization**: Frame skipping, memory cleanup, support for free/pro environments
9. **Modularity and robustness**: Shared `src/` modules, auxiliary functions, error handling
10. **Clear outputs**: Concise outputs, success/error messages with emoji prefixes
11. **Secure credential management**: Environment variables only; never exposes sensitive data
12. **Aligned database**: Persistence in 3 tables per required schema
13. **Dynamic context**: Adapts to mobile cameras, zoom, variable angles, and real bridge conditions
14. **Traffic state classification**: TF/Keras MLP classifies each minute into 4 states from 14 engineered features

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

- **Google Colab Free/Pro**: Primary runtime with GPU
- **Local with GPU**: CUDA optional; CPU supported (slower)
- **Cloud alternative**: AWS/Azure/GCP with GPU

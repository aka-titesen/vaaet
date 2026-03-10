# AGENTS.md — Agentic Context for VAAET

> This file is the long-term memory for AI agents operating on this repository.
> Read it in full before making any modifications to the project.

---

## Project Identity

**VAAET** (Video Advanced Analysis of Traffic) is a computer vision system for analyzing vehicular traffic on the **General Manuel Belgrano Bridge** (Corrientes, Argentina). It detects, classifies, tracks, and estimates the speed of vehicles using SISE surveillance camera footage.

### Fundamental Architecture (ADR-009)

Three-module pipeline with shared `src/` modules:

- **Module 0 — Bootstrap (archived)**: `archive/00_bootstrap/01_legacy_collection.ipynb` — Historical YOLO 11 + OpenCV + SORT pipeline that generated the original `traffic_data` table. **Never runs again.**
- **Module 1 — Data Preparation**: `notebooks/01_data_prep/data_preparation.ipynb` — One-time feature engineering + auto-labeling + SMOTE + MLP training → exports `.keras` and `.joblib` artifacts
- **Module 2 — Production**: `notebooks/02_production/traffic_analyzer.ipynb` — YOLO 11 + SORT + speed estimation + trained MLP classifier + DB persistence + self-improving feedback loop
- **Shared code**: `src/` — Reusable Python modules (config, db, features, labeling, perception) imported by Modules 1 and 2
- **Runtime**: Google Colab (free GPU). NO servers, REST APIs, microservices, or containers
- **Persistence**: PostgreSQL on AWS RDS (optional). 3 tables: `traffic_data` (legacy), `telemetry_raw` + `traffic_classifications` (active)
- **No classic CI/CD**: No build pipeline, no deploy. "Deployment" means opening notebooks in Colab and running cells

### Technology Stack

| Layer | Technology |
|---|---|
| Object detection | YOLO 11 (Ultralytics) — 5 variants by video duration |
| Computer vision | OpenCV — video I/O, optical flow, annotations |
| Numerical computation | NumPy — vectorized operations, statistics |
| ML (speed smoothing) | scikit-learn `MLPRegressor` — NOT a real CNN |
| Traffic classification | TensorFlow/Keras — MLP Sequential (evolvable to LSTM) |
| Data analysis | Pandas + SQLAlchemy — feature engineering and persistence |
| Class balancing | imbalanced-learn SMOTE — synthetic oversampling |
| Database | PostgreSQL via `psycopg2-binary` / SQLAlchemy (AWS RDS) |
| Runtime | Google Colab (primary) or Python 3.8+ local |

---

## Project Structure

```
vaaet/
├── archive/
│   └── 00_bootstrap/
│       ├── 01_legacy_collection.ipynb   # Module 0: Historical YOLO pipeline (FROZEN)
│       └── README.md                    # Explains deprecated status
├── notebooks/
│   ├── 01_data_prep/
│   │   └── data_preparation.ipynb       # Module 1: Feature eng. + model training
│   └── 02_production/
│       └── traffic_analyzer.ipynb       # Module 2: YOLO + classifier + feedback
├── src/
│   ├── __init__.py
│   ├── config.py                        # Single source of truth: constants, paths, thresholds
│   ├── db.py                            # SQLAlchemy engine factory, credential handling
│   ├── features.py                      # Feature engineering (9 → 14 columns)
│   ├── labeling.py                      # Auto-labeling rules (4 traffic states)
│   └── perception/
│       ├── __init__.py
│       ├── detector.py                  # YOLODetector wrapper
│       ├── tracker.py                   # SORTTracker wrapper
│       └── speed.py                     # Physics-based speed estimation
├── models/
│   ├── perception/                      # YOLO weights (downloaded at runtime, gitignored)
│   └── intelligence/                    # Module 1 artifacts (.keras, .joblib, gitignored)
├── data/
│   ├── raw/                             # DB backups (gitignored)
│   ├── processed/                       # Feature CSVs (gitignored)
│   └── samples/                         # Example data
├── docs/
│   ├── PRD.md                           # Product requirements
│   ├── DDS.md                           # Software design
│   ├── USER_GUIDE.md                    # User guide
│   ├── DATA_LINEAGE.md                  # Data lineage
│   ├── BIAS_AND_LIMITATIONS.md          # Biases and limitations
│   ├── KPIs/KPIs.md                     # Metrics and validation
│   ├── adr/                             # Architecture Decision Records (9 ADRs)
│   │   ├── ADR-001 to ADR-007           # Legacy (superseded by ADR-009)
│   │   ├── ADR-008-tensorflow-keras-traffic-classifier.md
│   │   └── ADR-009-modular-three-stage-architecture.md
│   └── diagrams/                        # Mermaid diagrams
├── README.md
├── AGENTS.md                            # This file
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── llms.txt
├── llms-full.txt
└── .gitignore
```

---

## Cell Execution Order

### Module 0 — Bootstrap (ARCHIVED — DO NOT RUN)

Located in `archive/00_bootstrap/01_legacy_collection.ipynb`. Historical-only reference.
Contains 21 cells including `VAAETHybrid` class, video processing, and synthetic demo generator.

### Module 1 — Data Preparation (`notebooks/01_data_prep/data_preparation.ipynb`)

Run once to train the traffic classifier. Re-run only when retraining with new data.

| Cell | Content | Dependencies |
|---|---|---|
| 0 | Environment setup (Colab clone/cd, local no-op) | None |
| 1 | Dependencies + imports (TF/Keras, pandas, etc.) | Cell 0 |
| 2 | DB connection + telemetry extraction from `traffic_data` | Cell 1 |
| 3 | Feature engineering (9 → 14 columns) via `src/features.py` | Cell 2 |
| 4 | Auto-labeling (engineering rules → 4 states) via `src/labeling.py` | Cell 3 |
| 5 | SMOTE + Train/Test split | Cells 3, 4 |
| 6 | MLP model definition + training | Cell 5 |
| 7 | Evaluation + model export (.keras, .joblib) | Cell 6 |
| 8 | Create DB tables + persist results | Cells 2, 7 |

### Module 2 — Production (`notebooks/02_production/traffic_analyzer.ipynb`)

Run for ongoing traffic analysis. Processes new video clips and classifies traffic state.

| Cell | Content | Dependencies |
|---|---|---|
| 0 | Environment setup (Colab clone/cd) | None |
| 1 | Dependencies + load trained model + scalers | Cell 0 |
| 2 | Video input + perception pipeline (`process_clip()`) | Cell 1 |
| 3 | Feature engineering + classification (`classify_telemetry()`) | Cells 1, 2 |
| 4 | Persist results to DB (`persist_classifications()`) | Cells 1, 3 |
| 5 | Re-training with HITL feedback (`retrain_with_feedback()`) | Cells 1, 4 |
| 6 | Visualization dashboard (`show_dashboard()`) | Cells 3, 4 |

---

## Validation Contract

No build or CI/CD. The functional equivalent is:

### Module 0 (Bootstrap — archived)
1. **DO NOT RUN** — This notebook is frozen. Reference only.

### Module 1 (Data Preparation)
2. **Smoke test**: Run Cell 1 without import errors (TF/Keras)
3. **Data**: Cell 2 loads DataFrame with >0 rows from `traffic_data`
4. **Features**: Cell 3 produces DataFrame with 14 columns, no NaN
5. **Classification**: Cell 7 shows F1-macro ≥ 0.85
6. **Artifacts**: `traffic_classifier.keras`, `feature_scaler.joblib`, `label_mapping.joblib` exist in `models/intelligence/`

### Module 2 (Production)
7. **Smoke test**: Run Cell 1 without errors; model loads successfully
8. **Perception**: Cell 2 processes a video clip and produces a telemetry DataFrame
9. **Classification**: Cell 3 classifies telemetry into one of 4 states
10. **Persistence**: Cell 4 persists to `telemetry_raw` and `traffic_classifications` (if DB available)

---

## Architectural Boundaries (DO NOT MODIFY without ADR)

1. **Module 0 is FROZEN** — never modify `archive/00_bootstrap/01_legacy_collection.ipynb`
2. **`src/config.py` is the SINGLE source of truth** for constants, thresholds, and paths
3. **`src/db.py` is the SINGLE point of DB configuration** — credentials via environment variables only
4. **Speed fusion is 70% physics + 30% MLP** — do not alter without experimental evidence
5. **`is_stationary()` criteria use AND-conjunction** — do not relax to OR without ADR
6. **Video filename format is strict**: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`
7. **Tables `telemetry_raw` and `traffic_classifications` require ADR-008** — do not modify schema without new ADR
8. **14 features are canonical** — defined in `src/config.py:FEATURE_COLS`. Do not add/remove without updating all modules
9. **4 traffic states**: Normal (0), Reduced (1), Congested (2), Accident (3) — defined in `src/config.py:STATE_LABELS`
10. **Shared `src/` modules must remain notebook-importable** — no CLI entrypoints, no `if __name__` blocks

---

## Error Handling Patterns

- **Silent degradation**: If DB fails, the system continues without persistence. If optical flow fails, uses physics-only speed calculation
- **Try/except with emoji**: All exceptions are caught and printed with emoji prefixes (🔴 error, ✅ success, ⚠️ warning). Exceptions are NOT propagated
- **No formal logging**: Uses `print()` with emojis, not Python's `logging` module
- **Plausibility filters**: Speeds outside [2, 120] km/h are silently discarded

---

## Governance System: Always / Ask / Never

### ✅ Always (do without supervision)

- Fix Python syntax errors
- Update docstrings and comments
- Add type hints to existing functions
- Add narrative markdown cells to notebooks
- Format code (PEP 8)
- Update documentation that is outdated relative to code
- Fix inconsistencies between `src/config.py` and module implementations
- Update `AGENTS.md` when project structure changes

### 🟡 Ask (requires human approval)

- Change calibration parameters (`BRIDGE_CONFIG`, `bridge_calibration`, `perspective_zones`)
- Modify speed calculation logic (`estimate_speed()` or `is_stationary()`)
- Add new dependencies to the project
- Change the schema of any database table
- Modify YOLO model selection logic
- Refactorings affecting >20% of a notebook cell or `src/` module
- Change confidence or NMS thresholds
- Change auto-labeling thresholds for traffic states
- Modify MLP/LSTM classifier architecture
- Add or remove features from `FEATURE_COLS`

### 🔴 Never (absolute restrictions)

- Hardcode AWS RDS credentials (host, password, etc.)
- Modify `archive/00_bootstrap/01_legacy_collection.ipynb`
- Delete `test_sistema()` or the synthetic demo generator from Module 0
- Modify the `traffic_data` table without drafting an ADR first
- Break compatibility with Google Colab Free Tier
- Remove strict video filename validation
- Commit `.pt` files (YOLO models) to the repository
- Print credentials in cell outputs
- Modify `telemetry_raw` or `traffic_classifications` without ADR
- Remove HITL fields from `traffic_classifications`
- Commit `.keras`, `.joblib`, or CSVs from `data/processed/`

---

## Stop Criteria and Handoff

The agent MUST stop and request human intervention when:

1. **Wants to change the detection model** (e.g., replace YOLO 11 with RT-DETR)
2. **Wants to modify the 70/30 fusion formula** for speed estimation
3. **Wants to alter the ultra-conservative criteria** of `is_stationary()`
4. **Wants to change any database table schema**
5. **Detects documentation-code inconsistencies** that cannot be resolved with certainty
6. **A change requires access to AWS RDS** or real credentials
7. **A change affects Colab Free Tier performance** (e.g., increasing inference resolution)

---

## Key Architecture Decision Records

Consult ADRs in `docs/adr/` before proposing changes that contradict these decisions:

| ADR | Decision | Status |
|---|---|---|
| ADR-001 | Monolithic notebook over Python modules | Superseded by ADR-009 |
| ADR-002 | YOLO 11 with adaptive selection by duration | Superseded by ADR-009 |
| ADR-003 | SORT over DeepSORT/ByteTrack | Superseded by ADR-009 |
| ADR-004 | MLP as smoother (not primary estimator) | Superseded by ADR-009 |
| ADR-005 | PostgreSQL (AWS RDS) over SQLite/local | Superseded by ADR-009 |
| ADR-006 | Ultra-conservative stationary detection | Superseded by ADR-009 |
| ADR-007 | Google Colab as primary runtime | Superseded by ADR-009 |
| ADR-008 | TF/Keras traffic classifier + two tables + auto-labeling | Active |
| ADR-009 | Modular three-stage architecture with shared src/ | **Active** |

---

## Domain Context

- **Bridge**: General Manuel Belgrano, 1700m length, 8.3m roadway width
- **Cameras**: SISE dynamic cameras at 60m height, with zoom, pan, night vision
- **Vehicle types**: car, truck, bus, motorcycle, bicycle
- **Typical speeds**: 40-80 km/h normal flow, 0-20 km/h congestion
- **Persistence**: One record per minute with average speed and counts by type
- **Traffic states**: Normal, Reduced, Congested, Accident
- **Features**: 14 engineered columns from 9 raw telemetry fields

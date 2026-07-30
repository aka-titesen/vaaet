<!-- context: VAAET/docs/DATA_LINEAGE.md — Data lineage documentation.
Complements DDS.md (technical design) and BIAS_AND_LIMITATIONS.md (biases). -->

# Data Lineage — VAAET

This document describes the origin, transformation, and destination of all data flowing through the VAAET system.

---

## 1. Data Sources

### Video Input

| Attribute | Value |
|---|---|
| **Origin** | SISE surveillance cameras on Gral. Manuel Belgrano Bridge |
| **Format** | MP4 (H.264) |
| **Typical resolution** | 1920x1080 (Full HD) |
| **FPS** | 30 fps (configurable by camera) |
| **Naming convention** | `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4` |
| **Owner** | SISE System (National Highway Directorate) |
| **Access** | Restricted — videos are not distributed with the project |

### Pre-trained Models

| Model | Source | Training Dataset | Relevant Classes |
|---|---|---|---|
| yolo11n/s/m/l/x.pt | Ultralytics Hub | MS COCO 2017 | car, truck, bus, motorcycle, bicycle |

**Note**: Models are downloaded automatically at runtime and are NOT versioned in the repository.

---

## 2. Module 0 — Bootstrap Pipeline (ARCHIVED)

Located in `archive/00_bootstrap/01_legacy_collection.ipynb`. This pipeline generated the historical `traffic_data` table and **never runs again**.

```
Video MP4
  |
  +-- [1] Name validation -------- Rejects if format doesn't match
  |
  +-- [2] Frame extraction ------- OpenCV VideoCapture @ 30fps
  |
  +-- [3] YOLO 11 detection ------ Per frame:
  |     |                            Input: BGR frame (numpy array)
  |     |                            Output: list of (bbox, class, confidence)
  |     |                            Filters: conf > 0.5, NMS IoU < 0.4
  |     +-- Only vehicular classes: car, truck, bus, motorcycle, bicycle
  |
  +-- [4] SORT tracking --------- Matching by Euclidean distance
  |     |                            Input: current frame detections
  |     |                            Output: track_id for each detection
  |     |                            Threshold: 100px, same vehicle type
  |     +-- Stores centroid history in deque(30)
  |
  +-- [5] Hybrid speed ---------- Per active track:
  |     |   [5a] Optical Flow (Lucas-Kanade) -> global motion vector
  |     |   [5b] Camera compensation -> subtract global motion
  |     |   [5c] Euclidean displacement -> distance in pixels
  |     |   [5d] Perspective correction -> factor by Y zone
  |     |   [5e] Conversion -> pixels/meter x factor -> km/h
  |     |   [5f] MLP prediction -> 10 features -> smoothed speed
  |     |   [5g] Fusion -> 0.7 x physics + 0.3 x MLP
  |     |   [5h] Plausibility filter -> [2, 120] km/h
  |     +-- Speeds outside range silently discarded
  |
  +-- [6] Stationary classification -- AND of 6 statistical criteria
  |     |                                If stationary -> speed = 0
  |     +-- Requires minimum 200 frames of observation
  |
  +-- [7] Visual annotation ------ Draw on frame:
  |     |   Bounding boxes, type + ID, speed, informational HUD
  |     +-- Write frame to output video (OpenCV VideoWriter)
  |
  +-- [8] Persistence (every 60s) -- INSERT into PostgreSQL:
        |   clip_id, record_time, avg_speed, count_* by type, total
        +-- Optional — silent failure if no DB configured
```

---

## 3. MLP Speed Smoother Training Data

| Attribute | Value |
|---|---|
| **Type** | Random synthetic data |
| **Generation** | `np.random.rand(100, 10)` for features, `np.random.rand(100) * 80 + 20` for targets |
| **Seed** | Not fixed — each execution generates different data |
| **Purpose** | Scaffold to initialize MLPRegressor. NOT real training |
| **Impact** | MLP acts as a regularizer toward the mean (~60 km/h); contribution capped at 30% |

See [ADR-004](adr/ADR-004-mlp-como-suavizador.md) for the rationale.

---

## 4. Module 0 Output Data

### Annotated Video

| Attribute | Value |
|---|---|
| **Format** | MP4 (mp4v codec) |
| **Content** | Original frames + bounding boxes + type/ID + speed + HUD |
| **Destination** | Auto-download to user device (Colab) |
| **Retention** | Ephemeral — lost when Colab session closes |

### Database Records (`traffic_data`)

| Field | Type | Description |
|---|---|---|
| `clip_id` | TEXT | Identifier derived from video filename |
| `record_time` | TIMESTAMP | Timestamp of the recorded minute |
| `avg_speed` | NUMERIC(5,2) | Average speed of moving vehicles (km/h) |
| `count_car` | INTEGER | Cars detected in the minute |
| `count_truck` | INTEGER | Trucks detected in the minute |
| `count_bus` | INTEGER | Buses detected in the minute |
| `count_motorcycle` | INTEGER | Motorcycles detected in the minute |
| `count_bicycle` | INTEGER | Bicycles detected in the minute |
| `total_vehicles` | INTEGER | Total vehicles in the minute |

**Frequency**: One record every 60 seconds of processed video.
**Destination**: `traffic_data` table in PostgreSQL (AWS RDS).
**Retention**: Indefinite (depends on RDS instance policy).

---

## 5. Module 1 — Data Preparation Pipeline

`notebooks/01_data_prep/data_preparation.ipynb` — Runs once to train the traffic classifier.

```
traffic_data (PostgreSQL)
  |
  +-- [1] SQL query ----------- SELECT 9 fields + id, ordered by record_time
  |
  +-- [2] DataFrame ----------- ~2000 records x 10 columns (pandas)
  |
  +-- [3] Feature Engineering -- 9 raw fields -> 14 features via src/features.py:
  |     |   heavy_vehicle_ratio, delta_speed, delta_count,
  |     |   transition_flag, speed_variance, hour_of_day, weather_condition
  |     +-- Drop first rows with NaN from diff()
  |
  +-- [4] Auto-Labeling ------- Engineering rules -> 4 states via src/labeling.py:
  |     |   Accident(3) -> Congested(2) -> Reduced(1) -> Normal(0)
  |     +-- NOT human ground truth. Engineering proxy.
  |
  +-- [5] SMOTE --------------- Balance training set (train only)
  |     |   StandardScaler fit on train, transform on both
  |     +-- Test set retains original distribution
  |
  +-- [6] MLP Training -------- Dense(64) -> Dense(32) -> Softmax(4)
  |     |   EarlyStopping + ReduceLROnPlateau, seed=42
  |     +-- Exports: traffic_classifier.keras, feature_scaler.joblib
  |
  +-- [7] Evaluation ---------- F1-macro, confusion matrix, recall per class
  |
  +-- [8] Persistence --------- 2 new tables:
        |   telemetry_raw: 14 features + FK to traffic_data(id)
        |   traffic_classifications: prediction + confidence + HITL
        +-- Optional — silent failure if no DB
```

### Module 1 Artifacts

| Artifact | Path | Format | Gitignored |
|---|---|---|---|
| Trained model | `models/intelligence/traffic_classifier.keras` | Keras native | Yes |
| Scaler | `models/intelligence/feature_scaler.joblib` | joblib | Yes |
| Label mapping | `models/intelligence/label_mapping.joblib` | joblib | Yes |
| Feature dataset | `data/processed/traffic_telemetry.csv` | CSV | Yes |

### Training Data

| Attribute | Value |
|---|---|
| **Source** | Real telemetry from `traffic_data` (~2000 records) |
| **Backup** | `data/raw/traffic_data.backup` (pg_dump binary) |
| **Labels** | Auto-labeling with engineering rules (NOT ground truth) |
| **Balancing** | SMOTE on training set |
| **Partition** | 80/20 stratified, seed=42 |

---

## 6. Module 2 — Production Pipeline

`notebooks/02_production/traffic_analyzer.ipynb` — Runs for ongoing traffic analysis with feedback loop.

```
Video MP4 (new clip)
  |
  +-- [1] Perception pipeline -- YOLODetector + SORTTracker + speed estimation
  |     |   via src/perception/ modules
  |     +-- Produces telemetry DataFrame (9 raw fields per minute)
  |
  +-- [2] Feature engineering -- 9 -> 14 features via src/features.py
  |
  +-- [3] Classification ------- Load trained model + scaler from models/intelligence/
  |     |   Scale features, predict state + confidence
  |     +-- 4 states: Normal(0), Reduced(1), Congested(2), Accident(3)
  |
  +-- [4] Persistence ---------- INSERT into telemetry_raw + traffic_classifications
  |     +-- Optional — silent failure if no DB
  |
  +-- [5] HITL Feedback -------- SISE operator corrections
        |   Updates is_human_validated, human_override_state
        +-- Retraining with corrected labels (self-improving loop)
```

---

## 7. Privacy and Data Security

### Potentially Sensitive Data

- **License plates**: SISE camera videos may capture plates. VAAET does NOT extract, store, or process individual plates
- **Individual frames**: NOT stored in DB. Only aggregated per-minute data
- **Temporal location**: Timestamps in tables enable traffic pattern inference by hour

### Credentials

- AWS RDS credentials obtained via environment variables or `getpass`
- NEVER printed in cell outputs
- NEVER hardcoded in code
- NEVER versioned in Git (`.gitignore` excludes `.env`)

### Data NOT Collected

- Driver identity
- Individual license plates
- Images of people
- Individual tracking data outside the processed video

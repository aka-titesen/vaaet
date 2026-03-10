<!-- context: VAAET/docs/DDS.md — Software Design Document for the VAAET system.
Complements PRD.md (requirements). Module 0 is archived in archive/00_bootstrap/,
Module 1 in notebooks/01_data_prep/, Module 2 in notebooks/02_production/.
Decisions are in docs/adr/. -->

# Software Design Document (DDS): VAAET System

---

## 1. System Architecture

VAAET's architecture is designed as a **three-module sequential processing pipeline** with shared `src/` modules, encapsulated in Jupyter Notebook environments for reproducibility and intermediate result inspection. The architecture follows **SOLID, YAGNI, and KISS** principles with high cohesion and low coupling.

See [ADR-009](adr/ADR-009-modular-three-stage-architecture.md) for the rationale behind the modular three-stage architecture.

### Logical Data Flow Diagram

```mermaid
flowchart TD
    A[Video Input<br/>bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4] --> B[Validate name<br/>and extract duration]
    B --> C{Duration}
    C -->|< 1h| D1[yolo11x.pt]
    C -->|1-3h| D2[yolo11l.pt]
    C -->|3-6h| D3[yolo11m.pt]
    C -->|6-12h| D4[yolo11s.pt]
    C -->|> 12h| D5[yolo11n.pt]
    
    D1 & D2 & D3 & D4 & D5 --> E[Load YOLO 11 model]
    
    E --> F[Initialize perception pipeline<br/>src/perception/]
    F --> G[Read video frame]
    
    G --> H[Detect vehicles<br/>YOLODetector.detect]
    H --> I[Match tracks<br/>SORTTracker.update]
    I --> J[Calculate speed<br/>estimate_speed - 70/30 fusion]
    J --> K{Stationary?<br/>AND-conjunction 6 criteria}
    
    K -->|Yes| L[Mark as stationary<br/>Speed = 0]
    K -->|No| M[Record speed<br/>in track history]
    
    L & M --> N[Draw annotations<br/>Bounding boxes + HUD]
    N --> O[Write annotated frame<br/>to output video]
    
    O --> P{Every 60s?}
    P -->|Yes| Q[Persist to PostgreSQL<br/>AWS RDS - optional]
    P -->|No| R{More frames?}
    Q --> R
    
    R -->|Yes| G
    R -->|No| S[Finalize video]
    S --> T[Download annotated video<br/>Auto-download in Colab]
```

### Infrastructure Architecture

```mermaid
flowchart LR
    A[User] -->|Upload .mp4| B[Google Colab]
    B -->|Inference| C[GPU T4/V100]
    C -->|Detections| B
    B -->|INSERT per minute| D[(AWS RDS<br/>PostgreSQL)]
    B -->|Annotated video| A
    E[Ultralytics Hub] -->|.pt model| B
```

See [detailed Colab+AWS architecture diagram](diagrams/colab-aws-architecture.md).

---

## 2. Detailed Component Design and Processing Flows

### 2.1. Vehicle Detection and Recognition Flow

**Objective:** Transform raw pixel data from each frame into a structured list of vehicular objects with position, class, and confidence.

**Detailed Process:**

1. **Frame Ingestion and Pre-processing:** Each frame, read by OpenCV as a BGR `numpy.ndarray`, is sent to the inference engine. The `ultralytics` library abstracts pre-processing including color space conversion (BGR to RGB), pixel value normalization ([0, 1]), and resizing to the model's training resolution (e.g., 640x640) with aspect ratio preservation and padding.

2. **CNN Inference:** The `YOLO` model instance processes the pre-processed frame tensor. YOLO's single-shot detector architecture performs a single pass to simultaneously predict bounding boxes and class probabilities.

3. **Post-processing and Filtering (NMS):**
   - **Confidence threshold filter**: Discards predictions with confidence below `0.5`
   - **Non-Max Suppression (NMS)**: Groups overlapping boxes (IoU threshold `0.4`) and keeps only the highest-confidence box per group

4. **Structured Data Extraction:** The final output is a clean list of detected objects, each containing: bounding box coordinates `[x1, y1, x2, y2]`, predicted class label (e.g., 'car'), and confidence score.

**Key Libraries and Techniques:**
- **`ultralytics`**: High-level YOLO 11 implementation
- **Deep Learning / CNN**: Object detection algorithm
- **NMS**: Post-processing algorithm ensuring one box per object

### 2.2. Per-Vehicle Speed Calculation Flow

**Objective:** Robustly estimate each vehicle's speed, compensating for perspective nonlinearities and camera dynamics.

**Detailed Process:**

1. **Trajectory History Maintenance:** For each tracked vehicle, a fixed-size `collections.deque` stores the last N centroid coordinates `(cx, cy)`, enabling O(1) insertion and deletion.

2. **Camera Motion Compensation (Virtual Stabilization):**
   - Optical Flow computed via `cv2.calcOpticalFlowPyrLK` (pyramidal Lucas-Kanade)
   - Robust statistical filtering (median vector via `numpy.median`) estimates global camera motion
   - Global motion vector is **subtracted** from vehicle displacement, isolating true vehicle movement

3. **Displacement Calculation and Perspective Correction:**
   - Cumulative Euclidean displacement computed in pixel space
   - Mapped to real-world space via `get_perspective_factor()` — a nonlinear scale factor based on Y-coordinate, acting as an **implicit homography approximation**

4. **Speed Calculation and Fusion:**
   - Physical distance (meters) / time interval (from FPS) → initial speed in km/h
   - Combined with MLP prediction: **70% physics + 30% MLP** (convex weighted combination)
   - Plausibility filter: speeds outside [2, 120] km/h silently discarded

**Key Libraries and Techniques:**
- **OpenCV**: `cv2.calcOpticalFlowPyrLK()` for Lucas-Kanade Optical Flow
- **NumPy**: Vectorial operations, norms, statistics
- **scikit-learn**: `MLPRegressor` for speed refinement
- **Optical Flow**: Camera motion estimation and compensation
- **Data Fusion**: Physics-based + data-driven model combination

### 2.3. Stationary Vehicle Detection Flow

**Objective:** Reliably classify stopped vehicles with high precision (few false positives) to avoid misclassifying slow traffic as stationary.

**Process (`is_stationary()`):**

1. **Minimum temporal window**: Only activates after `min_stationary_observation` frames (~200) of tracking history
2. **Statistical trajectory analysis**: Computes position variance (`numpy.std`), total displacement (Euclidean), and max/average inter-frame movement
3. **Decision criterion**: A vehicle is classified as stationary **if and only if** ALL statistical criteria are met simultaneously (AND-conjunction). This makes the classifier **ultra-conservative** — high precision at the cost of potentially lower recall.

See [ADR-006](adr/ADR-006-deteccion-estacionarios-conservadora.md) for the rationale.

### 2.4. Strategies for Reducing Error Margin in Dynamic Environments

- **Camera Motion Compensation (Optical Flow)**: Primary strategy against camera non-stationarity
- **Adaptive Perspective Correction**: Dynamic scale factor based on Y-coordinate, flexible across zoom changes
- **Physical Plausibility Filtering**: Post-hoc filter eliminates tracking error outliers
- **Temporal Speed Smoothing**: Weighted moving average acts as a low-pass filter

---

## 3. Database Design

See [ADR-005](adr/ADR-005-postgresql-aws-rds.md) for PostgreSQL vs SQLite rationale.

### Schema

Three tables with FK chain: `traffic_data` -> `telemetry_raw` -> `traffic_classifications`.

```sql
-- Module 0 legacy table
CREATE TABLE IF NOT EXISTS traffic_data (
    id SERIAL PRIMARY KEY,
    clip_id TEXT NOT NULL,
    record_time TIMESTAMP NOT NULL,
    avg_speed NUMERIC(5,2) NOT NULL,
    count_car INTEGER NOT NULL,
    count_truck INTEGER NOT NULL,
    count_bus INTEGER NOT NULL,
    count_motorcycle INTEGER NOT NULL,
    count_bicycle INTEGER NOT NULL,
    total_vehicles INTEGER NOT NULL,
    UNIQUE (clip_id, record_time)
);

-- Module 1/2 engineered features
CREATE TABLE IF NOT EXISTS telemetry_raw (
    id SERIAL PRIMARY KEY,
    source_record_id INTEGER REFERENCES traffic_data(id),
    record_time TIMESTAMP NOT NULL,
    avg_speed NUMERIC(5,2),
    total_vehicles INTEGER,
    count_car INTEGER, count_truck INTEGER, count_bus INTEGER,
    count_motorcycle INTEGER, count_bicycle INTEGER,
    heavy_vehicle_ratio NUMERIC(5,4),
    delta_speed NUMERIC(6,2), delta_count INTEGER,
    transition_flag SMALLINT DEFAULT 0,
    speed_variance NUMERIC(6,2),
    hour_of_day SMALLINT, weather_condition SMALLINT DEFAULT 0,
    UNIQUE (source_record_id)
);

-- Module 1/2 classifications with HITL fields
CREATE TABLE IF NOT EXISTS traffic_classifications (
    id SERIAL PRIMARY KEY,
    telemetry_id INTEGER REFERENCES telemetry_raw(id),
    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    traffic_state SMALLINT NOT NULL,
    state_label TEXT NOT NULL,
    confidence NUMERIC(5,4) NOT NULL,
    model_version TEXT NOT NULL,
    is_human_validated BOOLEAN DEFAULT FALSE,
    human_override_state SMALLINT,
    validated_at TIMESTAMP,
    UNIQUE (telemetry_id, model_version)
);
```

See [ERD diagram](diagrams/erd.md) and [Phase 2 ERD](diagrams/erd-phase2.md) for visual detail.

### Connection Pattern

- **Library**: SQLAlchemy + `psycopg2-binary` (via `src/db.py`)
- **Write frequency**: One INSERT every 60 seconds of processed video
- **Connection pooling**: Not implemented — each write opens/closes connection
- **Failure handling**: Silent degradation — processing continues without persistence
- **Credentials**: Environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`)

---

## 4. Multi-Camera Design

The system auto-detects camera layout and processes each ROI independently.

| Layout | Detection | ROIs |
|---|---|---|
| 1 camera | Undivided frame | Full frame |
| 2 cameras | Central vertical split | Left half + right half |
| 4 cameras | Quadrant split | 4 independent quadrants |

See [multi-camera diagram](diagrams/multi-camera-layout.md).

### Per-ROI Processing

Each ROI is processed with its own tracking instance:
- Vehicle IDs are independent across ROIs
- Speeds calculated with per-view perspective
- Aggregated metrics combine data from all ROIs

---

## 5. Synthetic Video Generator

Module 0 (archived) includes a synthetic demo generator for portfolio demonstrations.

| Scenario | Description |
|---|---|
| `light` | Light traffic, few vehicles |
| `normal` | Normal traffic flow |
| `busy` | Dense traffic |
| `mixed` | Mixed conditions |
| `stationary_test` | Stopped vehicles for `is_stationary()` validation |

Vehicle distribution: car 65%, truck 15%, bus 8%, motorcycle 10%, bicycle 2%.

---

## 6. Error Handling

### General Strategy

The system adopts **silent degradation**: errors are caught, reported via `print()` with emoji, and processing continues when possible.

| Component | Error | Behavior |
|---|---|---|
| PostgreSQL | Connection failure | Continues without persistence |
| PostgreSQL | INSERT failure | Skips the minute, continues |
| Optical Flow | No feature points | Uses speed without camera compensation |
| YOLO | No detections | Uses historical averages |
| Tracking | No match | Creates new track |
| MLP | Prediction outside [5, 100] | Uses physics-only speed (100%) |
| Video | Corrupt frame | Skips to next frame |
| File | Invalid name | Aborts processing (fatal error) |

### Emoji Patterns

```
✅  Successful operation
⚠️  Warning (non-fatal)
🔴  Error (recoverable or fatal)
📊  Result/metric
🎬  Processing start/end
📥  File download
```

---

## 7. Related Architecture Decisions

| Decision | ADR | Status |
|---|---|---|
| Modular three-stage architecture | [ADR-009](adr/ADR-009-modular-three-stage-architecture.md) | **Active** |
| TF/Keras traffic classification | [ADR-008](adr/ADR-008-tensorflow-keras-traffic-classifier.md) | Active |
| Monolithic notebook | [ADR-001](adr/ADR-001-notebook-monolitico.md) | Superseded by ADR-009 |
| YOLO 11 adaptive | [ADR-002](adr/ADR-002-yolo11-seleccion-adaptativa.md) | Superseded by ADR-009 |
| SORT vs DeepSORT | [ADR-003](adr/ADR-003-sort-sobre-deepsort.md) | Superseded by ADR-009 |
| MLP as smoother | [ADR-004](adr/ADR-004-mlp-como-suavizador.md) | Superseded by ADR-009 |
| PostgreSQL AWS RDS | [ADR-005](adr/ADR-005-postgresql-aws-rds.md) | Superseded by ADR-009 |
| Conservative stationary detection | [ADR-006](adr/ADR-006-deteccion-estacionarios-conservadora.md) | Superseded by ADR-009 |
| Google Colab runtime | [ADR-007](adr/ADR-007-google-colab-como-runtime.md) | Superseded by ADR-009 |

---

## 8. Intelligence Layer (Modules 1 & 2)

Modules 1 and 2 transform raw telemetry from `traffic_data` into traffic state classifications. Module 1 trains the classifier; Module 2 applies it in production with a feedback loop.

See [ADR-008](adr/ADR-008-tensorflow-keras-traffic-classifier.md) for the full rationale.

### 8.1 Classifier Architecture

Tabular MLP with TensorFlow/Keras `Sequential`, designed to evolve to LSTM in a future phase.

See [intelligence pipeline diagram](diagrams/intelligence-pipeline.md).

```
Input(14 features)
  -> Dense(64, relu) -> BatchNormalization -> Dropout(0.3)
  -> Dense(32, relu) -> BatchNormalization -> Dropout(0.2)
  -> Dense(n_classes, softmax)
```

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Callbacks**: EarlyStopping(patience=15), ReduceLROnPlateau(patience=5, factor=0.5)
- **Seed**: 42 (reproducibility)

### 8.2 Feature Engineering

| Feature | Formula / Source | Type | Justification |
|---|---|---|---|
| `avg_speed` | Direct from `traffic_data` | NUMERIC(5,2) | Primary flow indicator |
| `total_vehicles` | Direct | INTEGER | Absolute volume |
| `count_car` ... `count_bicycle` | Direct (5 fields) | INTEGER | Vehicular composition |
| `heavy_vehicle_ratio` | `(truck+bus)/total.clip(1)` | NUMERIC(5,4) | Heavy vehicle flow impact |
| `delta_speed` | `avg_speed.diff()` | NUMERIC(6,2) | Acceleration/deceleration |
| `delta_count` | `total_vehicles.diff()` | INTEGER | Volume change rate |
| `transition_flag` | `abs(delta_speed)>10 AND abs(delta_count)>5` | SMALLINT | Simultaneous abrupt change |
| `speed_variance` | `avg_speed.rolling(5).std()` | NUMERIC(6,2) | Flow stability |
| `hour_of_day` | `record_time.dt.hour` | SMALLINT | Circadian pattern |
| `weather_condition` | Proxy by hour (nocturnal=1) | SMALLINT | Simulated environmental condition |

### 8.3 Auto-Labeling

Engineering rules that assign states as ground truth proxy. Evaluation order: severity first.

| State | Code | Conditions |
|---|---|---|
| Accident | 3 | `avg_speed < 2` AND `delta_speed < -20` AND 3+ consecutive records |
| Congested | 2 | `avg_speed < 5` AND `total_vehicles > 25` AND 2+ consecutive records |
| Reduced | 1 | `avg_speed in [5, 40]` AND `total_vehicles in [15, 25]` |
| Normal | 0 | Everything else (default) |

### 8.4 Training

- **Scaling**: StandardScaler fit on training, transform on both sets
- **Partition**: 80/20 stratified by class (seed=42)
- **Balancing**: SMOTE on training set only (adaptive k_neighbors)
- **Target metric**: F1-macro >= 0.85

### 8.5 Persistence Schema

Two tables with FK chain: `traffic_data` -> `telemetry_raw` -> `traffic_classifications`.

See [Phase 2 ERD diagram](diagrams/erd-phase2.md) for visual detail.

**HITL (Human-in-the-Loop) fields** in `traffic_classifications`:
- `is_human_validated`: Boolean, default FALSE
- `human_override_state`: State corrected by SISE operator
- `validated_at`: Validation timestamp

Designed for the HITL feedback loop in Module 2; Module 1 records are all automatic.

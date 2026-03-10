<!-- context: VAAET/docs/BIAS_AND_LIMITATIONS.md -- Biases and limitations.
Complements KPIs.md (metrics) and DATA_LINEAGE.md (lineage). -->

# Biases and Limitations -- VAAET

This document describes known biases, technical limitations, and operational constraints of the VAAET system. Its purpose is to provide transparency about conditions under which the system may not operate correctly.

---

## 1. Detection Biases

### 1.1 COCO Dataset Bias

YOLO 11 is pre-trained on MS COCO 2017, whose vehicular class distribution is not representative of Argentine traffic:

| Class | COCO Representation | Expected Impact |
|---|---|---|
| car | High (over-represented) | Reliable detection |
| truck | Medium | Acceptable detection |
| bus | Low | Possible confusion with large trucks |
| motorcycle | Medium | Acceptable, but Argentine motorcycles may differ visually |
| bicycle | Low (under-represented) | **Risk of under-detection**, especially at distance |

**Current mitigation**: None. Relies on YOLO 11 generalization capability.

**Recommended future mitigation**: Fine-tuning with Argentine traffic dataset (especially bicycles and buses).

### 1.2 Environmental Condition Bias

The system has NOT been systematically evaluated under:

| Condition | Risk | Evaluation |
|---|---|---|
| Heavy rain | Pavement reflections, lens droplets | Not evaluated |
| Fog | Reduced contrast and visibility | Not evaluated |
| Night (no illumination) | Low camera sensor signal | Not evaluated |
| Backlight (sunrise/sunset) | Silhouettes, detail loss | Not evaluated |
| Pronounced shadows | Possible false positives | Not evaluated |
| Snow/frost | N/A for Corrientes (subtropical climate) | Not applicable |

### 1.3 Camera Geometry Bias

- SISE cameras are dynamic (pan/tilt/zoom), introducing perspective variability
- Perspective correction uses a simplified model (factor by Y zone), not a full homography
- Vehicles at lateral frame extremes have less precise perspective correction
- Variable zoom can alter the `pixels_per_meter` ratio during video

---

## 2. Technical Limitations

### 2.1 Speed Without Ground Truth

- Pixel-to-meter conversion depends on manually calibrated `pixels_per_meter`
- NO real speed dataset exists for the Belgrano Bridge
- Real MAE is unknown -- the "< 5 km/h" target is a goal without benchmark
- Calibration was estimated from known bridge dimensions (8.3m width, cameras at 60m)

### 2.2 Speed Smoother MLP Trained with Random Data

- The `cnn_validator` component (misnomer -- it is an `MLPRegressor`) is trained with `np.random.rand()`
- Does NOT provide real learning about speed patterns
- Acts as a stochastic regularizer toward the mean (~60 km/h)
- Its contribution is capped at 30% of the fusion, limiting damage
- See [ADR-004](adr/ADR-004-mlp-como-suavizador.md) for details

### 2.3 Tracking Without Visual Re-Identification

- SORT uses Euclidean distance matching to nearest centroid
- If a vehicle is occluded for >1 second, it loses its ID and receives a new one
- In dense traffic with nearby same-type vehicles, incorrect assignments may occur
- See [ADR-003](adr/ADR-003-sort-sobre-deepsort.md)

### 2.4 Stationary Detection

- Requires 200 frames (~6.5s) minimum observation -- no early detection
- Vehicles with micro-movements (engine vibration, wind) may not be detected as stationary
- Fixed pixel thresholds -- do not adapt to resolution or zoom
- See [ADR-006](adr/ADR-006-deteccion-estacionarios-conservadora.md)

---

## 3. Infrastructure Limitations

### 3.1 Google Colab

| Limitation | Impact |
|---|---|
| Max ~12h sessions (Free) | Very long videos may not complete |
| GPU not guaranteed at peak hours | May fall back to CPU (10x slower) |
| Random disconnections | Progress lost -- no checkpointing |
| Ephemeral storage | Output videos lost when session closes |
| No programmatic execution | Cannot automate via API or cron |

### 3.2 AWS RDS

| Limitation | Impact |
|---|---|
| Requires externally provisioned instance | Recurring AWS cost |
| No connection pooling | Each write opens/closes connection |
| No automatic retry | If connection fails, minute record is lost |
| No SSL by default | Plaintext connection (risk on untrusted networks) |
| No schema migrations | CREATE TABLE IF NOT EXISTS as only mechanism |

---

## 4. Code Limitations

These issues exist in the archived Module 0 (`archive/00_bootstrap/01_legacy_collection.ipynb`):

| Issue | Location | Impact |
|---|---|---|
| Dead code in `is_stationary()` | VAAETHybrid class | Second criteria block unreachable after first `return` |
| Duplicate `get_smoothed_average()` | VAAETHybrid class | Second definition overwrites first |
| `pixels_per_meter` inconsistency: 12 vs 15 | Class init vs calibration | Possible speed conversion error |
| `speed_limits` inconsistency | Class init vs calibration | Different filtering ranges |
| `min_track_frames` inconsistency: 10 vs 20 | Different methods | Inconsistent threshold |
| `threading` imported but unused | Imports | Unnecessary dependency |
| `detection_zone` defined but unused | Calibration | Dead code |
| Name "CNN" for an MLP | Class, docs | Incorrect terminology |

**Note**: These issues are frozen in Module 0 and will not be fixed. The shared `src/` modules in Modules 1 and 2 use correct implementations.

---

## 5. Prioritized Improvement Recommendations

1. **Evaluate with real video** -- measure KPIs with ground truth
2. **Fine-tune YOLO** with Argentine traffic data -- improve bicycle detection
3. **Add SSL** to PostgreSQL connection -- security
4. **Implement checkpointing** -- resilience against Colab disconnections
5. **Replace weather proxy** with real meteorological data API
6. **Evolve MLP to LSTM** -- capture temporal patterns in traffic state classification

---

## 6. Traffic Classifier Biases and Limitations (Modules 1 & 2)

### 6.1 Auto-Labeling Bias

Training labels are NOT human ground truth. They are generated with traffic engineering rules (speed thresholds, volume, persistence). This introduces circular bias: the model learns the thresholds that labeled it, not necessarily the real traffic state.

**Current mitigation**: Thresholds are calibrated with Belgrano Bridge domain expertise.

**Future mitigation**: HITL (fields `is_human_validated`, `human_override_state` in `traffic_classifications`) will allow SISE operators to refine labels via the Module 2 feedback loop.

### 6.2 Class Imbalance

| Class | Expected Frequency | Risk |
|---|---|---|
| Normal | ~80% | Over-represented -- model tends to predict Normal |
| Reduced | ~15% | Low representation |
| Congested | ~4% | Very low representation |
| Accident | ~0.1% | Extremely rare -- possible recall 0 without SMOTE |

**Mitigation**: SMOTE generates synthetic minority samples in training set. However, synthetic Accident samples (feature space interpolation) may not be physically realistic.

### 6.3 Temporal Assumptions

Persistence thresholds in auto-labeling assume each record represents exactly 1 minute of observation. If Module 0 write frequency changes (e.g., every 30s or 2min), the Reduced (>60s), Congested (>120s), and Accident (>180s) states become invalid.

### 6.4 Simulated Weather

The `weather_condition` feature is a proxy based on hour of day (0=daytime 6-18h, 1=nighttime). It is NOT real meteorological data. This limits the model's ability to learn weather-related patterns.

### 6.5 No Temporality in MLP

The MLP processes each record independently -- it has no memory of previous records. It cannot learn sequential patterns like "decreasing speed over 5 consecutive minutes". Evolution to LSTM in a future phase will address this limitation.

### 6.6 Generalization

The model is trained exclusively with data from the General Manuel Belgrano Bridge. It is NOT transferable to other bridges, highways, or urban contexts without retraining with local data.

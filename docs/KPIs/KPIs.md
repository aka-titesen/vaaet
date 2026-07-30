<!-- context: VAAET/docs/KPIs/KPIs.md -- Performance metrics and validation guide.
Complements PRD.md (requirements) and BIAS_AND_LIMITATIONS.md (limitations). -->

# Performance Metrics (KPIs) and Validation Guide for VAAET

This document describes the Key Performance Indicators (KPIs) that define the success of the VAAET system and provides a detailed guide for validating the stated precision targets.

## 1. System KPIs

### 1.1 Detection and Classification Precision

- **What**: Measures the system's ability to correctly identify vehicles and assign the correct class (car, truck, bus, motorcycle, bicycle). Target: **97%**.
- **Why**: Foundation of the system. If detection fails, all downstream calculations are incorrect.
- **How**: Measured via **F1-Score** combining Precision and Recall.
  - **Precision**: `TP / (TP + FP)` -- of detected vehicles, what percentage was correct
  - **Recall**: `TP / (TP + FN)` -- of real vehicles, what percentage was detected

### 1.2 Speed Calculation Precision

- **What**: Difference between VAAET-calculated speed and actual vehicle speed.
- **Why**: Critical for traffic flow analysis and decision-making.
- **How**: **Mean Absolute Error (MAE)** = `(1/n) * SUM|RealSpeed - PredictedSpeed|`. Target: MAE < 5 km/h.

### 1.3 Tracking Reliability

- **What**: Ability to maintain consistent unique IDs across frames.
- **Why**: Essential for speed calculation and avoiding double-counting.
- **How**: **Identity Switches (ID Switches)** -- number of ID changes per video. Target: minimize to near-zero.

### 1.4 Processing Efficiency

- **What**: Video processing speed.
- **Why**: Determines viability for processing large video volumes.
- **How**: **Frames Per Second (FPS)** = total frames / processing time.

### 1.5 Stationary Detection Robustness

- **What**: Effectiveness of `is_stationary()` in correctly identifying stopped vehicles.
- **Why**: Prevents average speed contamination by non-moving vehicles.
- **How**: True Positive and False Positive rates for stationary classification.

---

## 2. Validation Guide for 97% Precision Target

### Step 0: Environment Preparation

1. **Test video**: 2-5 minute representative clip from the Belgrano Bridge (not used for training)
2. **Annotation tool**: CVAT, VGG Image Annotator (VIA), or similar

### Step 1: Create Ground Truth

1. Load video in annotation tool
2. Define class labels: `car`, `truck`, `bus`, `motorcycle`, `bicycle`
3. Annotate each frame (or every N frames): draw bounding boxes, assign classes, assign tracking IDs
4. Export annotations (JSON/XML/CSV) -- this is your Ground Truth

### Step 2: Run VAAET

1. Process the same test video with the production notebook
2. Export detections per frame: frame number, bbox [x1,y1,x2,y2], class, track ID, confidence

### Step 3: Compare and Calculate

1. Load both files in a comparison script
2. Match per frame using IoU > 0.5 threshold
3. Classify: TP (correct match), FP (no matching ground truth), FN (missed ground truth)
4. Calculate: `Precision`, `Recall`, `F1-Score = 2*(P*R)/(P+R)`

### Step 4: Interpretation

F1-Score >= 0.97 validates the 97% precision target. If below, Precision/Recall breakdown indicates improvement areas.

---

## 3. Current Measurement Status

> **Important**: KPI targets listed here are **declared objectives** not yet validated with real benchmarks.

| KPI | Target | Measurement Status |
|---|---|---|
| F1-Score Detection | 97% | No real benchmark. Requires manually annotated ground truth |
| MAE Speed | < 5 km/h | No speed ground truth. No reference data for the bridge |
| ID Switches | Minimize | No formal measurement. Requires ground truth with consistent IDs |
| FPS Processing | Variable | Not published. Depends on YOLO model and Colab GPU |
| Stationary Precision | High | No quantitative evaluation. Qualitatively validated with synthetic demos |
| F1-macro Classification | >= 0.85 | Pending first Module 1 execution |
| Recall Accident class | > 0 | Pending. Extremely rare class |

### Validation Prerequisites

1. **Test video** representative of the bridge (2-5 minutes)
2. **Manual annotation** with CVAT or VIA tool (Step 1 of the guide)
3. **Comparison script** with IoU > 0.5 (not currently provided -- must be implemented)
4. **Real speed data** (radar or GPS) for MAE validation -- currently unavailable

### Known Limitations

See [BIAS_AND_LIMITATIONS.md](../BIAS_AND_LIMITATIONS.md) for complete analysis of biases affecting KPIs.

---

## 4. Traffic State Classification KPIs (Modules 1 and 2)

These KPIs evaluate the quality of the MLP classifier.

### 4.1 F1-Score Macro

- **What**: Unweighted average of F1-Score per class. Treats all classes equally regardless of frequency.
- **Why**: Penalizes poor performance on rare classes (Accident ~0.1%) and frequent classes (Normal ~80%) equitably.
- **Target**: >= 0.85
- **How**: `sklearn.metrics.f1_score(y_test, y_pred, average='macro')`

### 4.2 Recall per Class

- **What**: Of all records truly belonging to a class, what percentage did the model detect?
- **Why**: Recall of 0 for Accident means the model NEVER detects the most operationally critical class.
- **Target**: > 0 for all classes present
- **Especially critical**: Accident (recall > 0 is the minimum acceptable)

### 4.3 Confusion Matrix

- **What**: Table showing prediction vs. reality distribution for each class pair.
- **Expected confusions**:
  - Normal <-> Reduced: fuzzy boundary at ~40 km/h
  - Congested <-> Reduced: fuzzy boundary at ~5 km/h
  - Accident rarely confused with Normal (very different speeds)

### 4.4 Current Measurement Status (Modules 1 and 2)

| KPI | Target | Status |
|---|---|---|
| F1-macro | >= 0.85 | Pending first execution |
| Recall Accident | > 0 | Pending (rare class, SMOTE mitigates) |
| Recall Normal | > 0.90 | Pending (majority class) |
| Recall Reduced | > 0.50 | Pending |
| Recall Congested | > 0.50 | Pending |

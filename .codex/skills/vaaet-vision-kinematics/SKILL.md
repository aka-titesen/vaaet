---
name: vaaet-vision-kinematics
description: Guide safe development and review of VAAET vehicle perception. Use for YOLO 11 detection, lightweight SORT tracking, Lucas-Kanade optical flow, camera-motion compensation, physical speed estimation, stationary-state logic, minute telemetry, and proposed vision-calibration improvements.
---

# VAAET Vision Kinematics

## Overview

Preserve the current, Colab-compatible perception path before proposing a more complex one. Keep the implementation evidence-led: temporal consistency, reliable motion estimates, and usable telemetry matter more than adopting a newer tracker or model.

## Preserve the current boundary

Treat `vaaet.vision.analyze_video()` as the common acquisition and inference boundary. Keep notebooks in `vaaet-ml/` as orchestrators and implement shared vision behavior in `vaaet-core/src/vaaet/vision/`.

Maintain the established ordered chain for each clip:

1. Read the next frame and preserve its temporal order.
2. Estimate optical flow and global camera motion.
3. Detect eligible COCO vehicles with the current YOLO wrapper.
4. Associate detections through the lightweight `SORTTracker` and retain per-track history.
5. Estimate, smooth, and qualify speed before updating motion state.
6. Aggregate global flow and complete minute-level telemetry.

Do not change public `Detection`, `SORTTracker`, `estimate_speed`, `OpticalFlowEstimator`, telemetry contracts, thresholds, model weights, or public APIs without explicit authorization. Respect ADR-0021, ADR-0013, ADR-0014, and ADR-0017 through ADR-0019; read ADR-0022 before any serving proposal and use ADR-0002, ADR-0003, and ADR-0006 only as historical context.


## Detection and tracking

Use YOLO 11 only through the existing wrapper. Filter to the supported COCO vehicle classes and download weights into the ephemeral runtime; never commit `.pt` files or other binary weights to Git.

Keep the lightweight SORT association and process one clip in chronological order. Preserve its bounded track histories, use Lucas-Kanade optical flow for scene motion, and compensate vehicle motion for estimated camera motion before deriving kinematics.

Avoid loading YOLO or other heavy models in unit tests. Test interfaces, fixtures, deterministic trajectory logic, rejection paths, and telemetry aggregation instead.

## Estimate kinematics conservatively

Estimate speed from a temporal track trajectory, real FPS and clip duration—not from a single frame. Apply the current zone-based perspective correction, reliability filters, smoothing, and vehicle-type ranges before exposing a value downstream.

Keep rest detection conservative and hysteretic. Represent near-zero motion separately from a confirmed stationary vehicle so noisy tracks do not inflate parked or stopped counts.

Aggregate traffic flow globally and produce minute-level telemetry only from complete minutes. Do not introduce lane segmentation or infer lane-level behavior from the present global pipeline.

## Make extensions evidence-led

Treat homography, bottom contact points, CLAHE, ByteTrack, DeepSORT, MOTA, sustained FPS above 30, and radar-accuracy claims as future proposals—not implemented behavior. Before introducing one, obtain explicit authorization and document the data, profiling, dependency, contract, and ADR impact.

For a future homography, require versioned per-camera calibration, geometric validation, and comparison with ground truth before it can alter telemetry. Do not use it merely because a calibration example exists.

Do not change the tracker, speed or confidence thresholds, or sampling order without authorization. Measure the real runtime and error distribution before claiming a performance or accuracy improvement.

## Log and validate safely

Log the selected YOLO variant, optical-flow quality, discarded tracks, complete processed minutes, and rejection reasons. Redact secrets, connection details, and private local or Drive paths from logs and errors.

For changes in scope, run the core gates from `vaaet-core/AGENTS.md`; add ML notebook gates only when a laboratory workflow changes. Validate GPU assignment, runtime downloads, Drive, database connectivity, and real-video behavior manually in Colab.

---

Reject these antipatterns:

- Derive velocity from an isolated frame or reorder frames inside a clip.
- Replace SORT, alter thresholds, or introduce unprofiled heavy models without authorization.
- Use predictions as validated traffic truth without the established confidence and review paths.
- Commit model weights, record secrets, or reveal private filesystem paths.
- Present projected calibration, throughput, or radar metrics as current local guarantees.

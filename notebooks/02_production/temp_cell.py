# Cell 2b — Annotated Video Output with Full HUD + Classification
#
# This is the PRIMARY execution cell (mirrors legacy process_bridge_video).
# It processes the video, generates the annotated output with HUD,
# classifies each minute, triggers the Colab download popup, and sets
# df_telemetry / df_classified for downstream cells (4, 5, 6).

# Colour palette (matches legacy COLORS)
_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 255, 0),
    "truck": (255, 165, 0),
    "bus": (255, 0, 0),
    "motorcycle": (0, 255, 255),
    "bicycle": (255, 0, 255),
}

_STATE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 200, 0),     # Normal  → green
    1: (0, 200, 255),   # Reduced → yellow/orange
    2: (0, 0, 255),     # Congested → red
    3: (0, 0, 180),     # Accident  → dark red
}


def _create_output_path(video_path: str) -> str:
    """Build output path like legacy: {basename}_VAAET_processed.mp4.

    On Colab: saves to /content/ so it appears in the Files panel.
    On local: saves next to the input video.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    filename = f"{basename}_VAAET_processed.mp4"
    if IN_COLAB:
        return os.path.join("/content", filename)
    else:
        parent = os.path.dirname(os.path.abspath(video_path))
        return os.path.join(parent, filename)


def _show_progress_bar(current: int, total: int, elapsed_s: float) -> None:
    """Render a compact textual progress bar (legacy-style)."""
    if total <= 0:
        return
    progress = min(max(current / total, 0.0), 1.0)
    bar_len = 30
    filled = int(bar_len * progress)
    bar = "█" * filled + "-" * (bar_len - filled)
    fps_eff = current / elapsed_s if elapsed_s > 0 else 0.0
    print(f"\r[{bar}] {progress * 100:5.1f}% | {current:,}/{total:,} frames | {fps_eff:5.1f} fps", end="")


def _label_codes_sorted() -> list[int]:
    """Return sorted numeric class codes available in label_mapping."""
    if label_mapping is None:
        return []
    codes: list[int] = []
    for key in label_mapping.keys():
        try:
            codes.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(set(codes))


def _decode_model_index(pred_idx: int) -> int:
    """Decode model output index to original traffic-state code."""
    idx = int(pred_idx)
    if label_mapping is None:
        return idx
    if idx in label_mapping:
        return idx
    codes = _label_codes_sorted()
    if 0 <= idx < len(codes):
        return int(codes[idx])
    return idx


def _draw_annotations(
    frame: np.ndarray,
    active_tracks: list,
    speed_tracker: SmoothedSpeedTracker,
    motion_state_tracker: TrackMotionStateTracker,
    fps: float,
    frame_h: int,
    global_motion: float,
    flow_tracking_ratio: float,
) -> tuple[dict[int, float], dict[str, int]]:
    """Draw bboxes, type label, and per-vehicle speed on the frame.

    Stationary vehicles are identified FIRST and labelled [S] without
    calling ``estimate_speed`` / ``speed_tracker.update``, preventing
    stale low-speed values from lingering in the smoothing window.

    Returns ``(individual_speeds, frame_quality)`` where frame_quality tracks
    stationary / rejected / recovered samples for minute-level telemetry.
    """
    individual_speeds: dict[int, float] = {}
    frame_quality: dict[str, int] = {"stationary": 0, "rejected": 0, "recovered": 0}
    for track in active_tracks:
        color = _COLORS.get(track.vehicle_type, (255, 255, 255))

        # Bounding box (derive from centroid + fixed half-size as tracker
        # does not expose bbox — matches legacy draw_annotations).
        cx, cy = track.centroid
        half_w, half_h = 50, 35
        cv2.rectangle(frame, (cx - half_w, cy - half_h),
                       (cx + half_w, cy + half_h), color, 2)

        recovered_gap = getattr(track, "recovered_after_gap", 0)
        reliable = is_speed_measurement_reliable(
            track.history,
            flow_tracking_ratio=flow_tracking_ratio,
            recovered_after_gap=recovered_gap,
        )
        speed = None
        if reliable:
            speed = estimate_speed(
                track.history, fps=fps, frame_height=frame_h,
                global_motion=global_motion, vehicle_type=track.vehicle_type,
            )

        stationary_now = motion_state_tracker.update(
            track.track_id,
            track.history,
            candidate_speed=speed,
        )

        if stationary_now:
            frame_quality["stationary"] += 1
            speed_tracker.remove_track(track.track_id)
            cv2.putText(frame, "[S]",
                        (cx - half_w, cy - half_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 255), 2)
        elif not reliable:
            frame_quality["rejected"] += 1
            if recovered_gap > 0:
                frame_quality["recovered"] += 1
            speed_tracker.remove_track(track.track_id)
            tag = "[R]" if recovered_gap > 0 else "[?]"
            tag_color = (0, 165, 255) if recovered_gap > 0 else (180, 180, 180)
            cv2.putText(frame, tag,
                        (cx - half_w, cy - half_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, tag_color, 2)
        else:
            smoothed = speed_tracker.update(track.track_id, speed)
            if smoothed is not None:
                individual_speeds[track.track_id] = smoothed
                cv2.putText(frame, f"{smoothed:.0f}km/h",
                            (cx - half_w, cy - half_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Type label below box
        cv2.putText(frame, track.vehicle_type.upper(),
                    (cx - half_w, cy + half_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return individual_speeds, frame_quality


def _safe_classify_records(records: list) -> tuple:
    """Classify the most recent record, NaN-safe for short inputs.

    engineer_features drops the first row because diff() produces NaN deltas.
    A single-record input produces an empty DataFrame, causing silent failure.
    Fix: duplicate the lone record so diff() yields delta=0 for row 1.

    Returns (state_code, state_label, confidence) or (None, None, None).
    """
    if model is None or scaler is None or label_mapping is None:
        return None, None, None
    if not records:
        return None, None, None
    try:
        # Use last 2 records to give diff() a valid previous row
        df = pd.DataFrame(records[-2:])
        df_feat = engineer_features(df)
        if df_feat.empty:
            # Only 1 record available: duplicate so delta=0 for row 1
            df2 = pd.concat([df.iloc[[0]], df.iloc[[0]]], ignore_index=True)
            df_feat = engineer_features(df2)
        if df_feat.empty:
            return None, None, None
        X = scaler.transform(df_feat[FEATURE_COLS].values[-1:])
        X = np.nan_to_num(X, nan=0.0)  # safety net for any residual NaN
        proba = model.predict(X, verbose=0)
        code_idx = int(proba.argmax(axis=1)[0])
        code = _decode_model_index(code_idx)
        conf = float(proba.max(axis=1)[0])
        lbl = label_mapping.get(code, label_mapping.get(code_idx, "Unknown"))
        return code, lbl, conf
    except Exception:
        return None, None, None


def _add_info_overlay(
    frame: np.ndarray,
    video_time_s: float,
    avg_speed: float,
    total_counts: dict[str, int],
    current_counts: dict[str, int],
    individual_speeds: dict[int, float],
    state_code: int | None,
    state_label: str | None,
    confidence: float | None,
    frame_idx: int,
    n_active_tracks: int,
) -> np.ndarray:
    """Render full HUD overlay with enlarged, readable text."""
    h, w = frame.shape[:2]

    # Left panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (620, 380), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Timestamp
    hrs = int(video_time_s // 3600)
    mins = int((video_time_s % 3600) // 60)
    secs = int(video_time_s % 60)
    cv2.putText(frame, f"TIME: {hrs:02d}:{mins:02d}:{secs:02d}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Average speed (colour-coded)
    if avg_speed > 80:
        spd_color = (0, 0, 255)       # red  – high
    elif avg_speed > 60:
        spd_color = (0, 165, 255)     # orange – moderate-high
    elif avg_speed < 30:
        spd_color = (255, 255, 0)     # cyan – low
    else:
        spd_color = (0, 255, 255)     # yellow – normal

    cv2.putText(frame, f"AVG SPEED: {avg_speed:.1f} km/h",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, spd_color, 2)
    cv2.line(frame, (20, 88), (600, 88), (255, 255, 255), 1)

    # Per-type counters
    cv2.putText(frame, "CUMULATIVE COUNTS:",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y = 140
    total_vehicles = sum(total_counts.values())
    current_total = sum(current_counts.values())

    for vtype in VEHICLE_TYPES:
        tc = total_counts.get(vtype, 0)
        cc = current_counts.get(vtype, 0)
        color = _COLORS.get(vtype, (255, 255, 255))
        if cc > 0:
            txt = f"{vtype.upper()}: {tc} (+{cc})"
            cv2.putText(frame, txt, (35, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (18, y - 6), 4, (0, 255, 0), -1)
        else:
            cv2.putText(frame, f"{vtype.upper()}: {tc}", (35, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 24

    total_color = (0, 255, 0) if current_total > 0 else (255, 255, 0)
    cv2.putText(frame, f"TOTAL DETECTED: {total_vehicles}",
                (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, total_color, 2)
    y += 26
    moving = len(individual_speeds)
    stationary = max(current_total - moving, 0)
    if current_total > 0:
        cv2.putText(frame, f"ACTIVE NOW: {current_total}",
                    (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO CURRENT ACTIVITY",
                    (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    y += 26
    if stationary > 0:
        cv2.putText(frame, f"STATIONARY: {stationary}",
                    (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 255), 2)

    # Status indicator dot
    if moving > 0:
        st_color, st_text = (0, 255, 0), f"ACTIVE ({moving})"
    elif current_total > 0:
        st_color, st_text = (0, 255, 255), f"STATIONARY ({stationary})"
    else:
        st_color, st_text = (0, 0, 255), "NO DETECTIONS"
    cv2.circle(frame, (590, 30), 10, st_color, -1)
    cv2.putText(frame, st_text, (440, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, st_color, 1)

    # Right panel: individual speeds
    overlay2 = frame.copy()
    if individual_speeds:
        panel_h = min(280, 70 + len(individual_speeds) * 28)
        cv2.rectangle(overlay2, (630, 10), (w - 10, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay2, 0.3, 0)
        cv2.putText(frame, f"INDIVIDUAL SPEEDS ({len(individual_speeds)}):",
                    (640, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        sy = 65
        for idx, (tid, spd) in enumerate(individual_speeds.items()):
            if idx >= 8:
                cv2.putText(frame, f"... and {len(individual_speeds) - 8} more",
                            (640, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (128, 128, 128), 1)
                break
            s_col = (0, 0, 255) if spd > 80 else ((255, 255, 0) if spd < 20 else (200, 200, 200))
            cv2.putText(frame, f"#{tid}: {spd:.0f}km/h",
                        (640, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_col, 1)
            sy += 28
    else:
        cv2.rectangle(overlay2, (630, 10), (w - 10, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay2, 0.3, 0)
        cv2.putText(frame, "NO MOVING", (640, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        cv2.putText(frame, "VEHICLES", (640, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

    # Traffic-state banner (bottom)
    banner_h = 70
    if state_code is not None and state_label is not None:
        banner_color = _STATE_COLORS.get(state_code, (100, 100, 100))
        cv2.rectangle(frame, (0, h - banner_h), (w, h), banner_color, -1)
        cv2.putText(frame, f"TRAFFIC STATE: {state_label.upper()}"
                    + (f"  ({confidence:.0%})" if confidence else ""),
                    (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
    else:
        # Show a muted "awaiting classification" banner so the area isn't blank
        cv2.rectangle(frame, (0, h - banner_h), (w, h), (60, 60, 60), -1)
        cv2.putText(frame, "TRAFFIC STATE: ANALYZING...",
                    (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (160, 160, 160), 2)

    # Tech info (bottom-left above banner)
    tech_y = h - banner_h - 15
    cv2.putText(frame, f"Frame: {frame_idx} | Active tracks: {n_active_tracks}",
                (20, tech_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (128, 128, 128), 1)

    return frame


def generate_annotated_video(
    video_path: str,
    output_path: str | None = None,
    model_variant: str | None = None,
    max_frames: int | None = None,
) -> tuple[str, pd.DataFrame]:
    """Generate annotated video with full HUD, per-minute classification, and
    traffic-state overlay — matching the legacy notebook output.

    Args:
        video_path: Input .mp4 path.
        output_path: Output path.  If ``None``, auto-generated as
            ``{basename}_VAAET_processed.mp4`` in /content/ (Colab) or
            next to the input (local).
        model_variant: YOLO variant (auto-selected if ``None``).
        max_frames: Process at most this many frames (``None`` = full video).

    Returns:
        ``(output_path, df_classified)`` — annotated video path and classified
        DataFrame (one row per minute with traffic_state + confidence).
    """
    # Output path 
    if output_path is None:
        output_path = _create_output_path(video_path)

    # Setup
    if validate_filename(video_path):
        duration = extract_duration(video_path)
        print(f"📎 Valid bridge filename — duration: {duration:.0f}s")
    else:
        print("⚠️ Non-standard filename — extracting duration from metadata")
        try:
            duration = extract_duration(video_path)
        except ValueError:
            duration = 300.0
            print(f"⚠️ Could not determine duration, using {duration:.0f}s default")

    if model_variant is None:
        model_variant = select_model_variant(duration)

    detector = YOLODetector(model_variant=model_variant)
    detector.load()
    tracker = SORTTracker()
    flow_estimator = OpticalFlowEstimator()
    speed_tracker = SmoothedSpeedTracker(window_size=10)
    motion_state_tracker = TrackMotionStateTracker()

    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_minute = int(fps * 60)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    # Accumulators
    frame_idx = 0
    minute_counts: dict[str, int] = {v: 0 for v in VEHICLE_TYPES}
    minute_speeds: list[float] = []
    minute_quality: dict[str, int] = {"stationary": 0, "rejected": 0, "recovered": 0}
    counted_tracks: set[int] = set()
    cumulative_counts: dict[str, int] = {v: 0 for v in VEHICLE_TYPES}

    # Current minute classification state (displayed on HUD)
    cur_state_code: int | None = None
    cur_state_label: str | None = None
    cur_confidence: float | None = None

    # Early classification trigger: classify after EARLY_CLASSIFY_SECS of
    # video so the banner appears quickly (especially for short clips).
    EARLY_CLASSIFY_SECS = 15
    early_classified = False

    records: list[dict] = []

    print(f"\n{'=' * 60}")
    print(f"🌉 VAAET — TRAFFIC ANALYSIS SYSTEM")
    print(f"Puente General Manuel Belgrano")
    print(f"{'=' * 60}")
    print(f"📁 Input:  {os.path.basename(video_path)}")
    print(f"📁 Output: {output_path}")
    print(f"📊 Video:  {duration:.1f}s @ {fps:.1f}fps ({total_frames:,} frames)")
    print(f"📐 Resolution: {frame_w}x{frame_h}")
    print(f"🧠 Model: {model_variant}")
    if max_frames:
        print(f"⚠️  Capped at {max_frames:,} frames")
    print(f"📈 Progress:")

    wall_start = _time.time()
    last_progress_wall = wall_start

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        # 1. Optical flow
        global_motion = flow_estimator.update(frame)

        # 2. Detect
        detections = detector.detect(frame)

        # 3. Track
        det_tuples = [(d.centroid, d.vehicle_type) for d in detections]
        active_tracks = tracker.update(det_tuples)
        for pruned_track_id in tracker.last_pruned_track_ids:
            speed_tracker.remove_track(pruned_track_id)
            motion_state_tracker.remove_track(pruned_track_id)
            counted_tracks.discard(pruned_track_id)
        flow_tracking_ratio = (
            flow_estimator.last_tracking_ratio
            if flow_estimator.last_total_points > 0
            else 1.0
        )

        # 4. Per-frame counts (for HUD "active now")
        current_frame_counts: dict[str, int] = {v: 0 for v in VEHICLE_TYPES}
        for track in active_tracks:
            current_frame_counts[track.vehicle_type] = (
                current_frame_counts.get(track.vehicle_type, 0) + 1
            )

        # 5. Draw annotations + collect individual speeds
        #    (speed_tracker.update is called ONCE per track inside here;
        #     low-confidence and recovered tracks are filtered before telemetry.)
        individual_speeds, frame_quality = _draw_annotations(
            frame, active_tracks, speed_tracker, motion_state_tracker, fps, frame_h,
            global_motion, flow_tracking_ratio,
        )

        # 6. Accumulate for telemetry
        #    NOTE: speeds already computed in _draw_annotations — do NOT call
        #    speed_tracker.update again (would corrupt the moving average).
        for spd in individual_speeds.values():
            minute_speeds.append(spd)
        for key, value in frame_quality.items():
            minute_quality[key] = minute_quality.get(key, 0) + value

        for track in active_tracks:
            if track.track_id not in counted_tracks:
                if track.mark_counted():
                    minute_counts[track.vehicle_type] = (
                        minute_counts.get(track.vehicle_type, 0) + 1
                    )
                    counted_tracks.add(track.track_id)

        # Early classification trigger 
        # Classify after EARLY_CLASSIFY_SECS so the banner appears quickly
        # for short clips instead of waiting for the first full minute.
        if (
            not early_classified
            and cur_state_code is None
            and frame_idx >= int(fps * EARLY_CLASSIFY_SECS)
            and (minute_speeds or any(minute_counts.values()))
        ):
            avg_speed_e = robust_speed_summary(minute_speeds) if minute_speeds else 0.0
            tmp_rec_e = {
                "record_time": datetime.now(),
                "avg_speed": round(avg_speed_e, 2),
                "count_car": minute_counts.get("car", 0),
                "count_truck": minute_counts.get("truck", 0),
                "count_bus": minute_counts.get("bus", 0),
                "count_motorcycle": minute_counts.get("motorcycle", 0),
                "count_bicycle": minute_counts.get("bicycle", 0),
                "total_vehicles": sum(minute_counts.values()),
            }
            code, lbl, conf = _safe_classify_records(records + [tmp_rec_e])
            if code is not None:
                cur_state_code = code
                cur_state_label = lbl
                cur_confidence = conf
                early_classified = True
                print(f"\n   🏷️  Early classification @ {frame_idx / fps:.0f}s: "
                      f"{lbl} ({conf:.0%})")

        # 7. HUD overlay
        avg_speed = robust_speed_summary(minute_speeds) if minute_speeds else 0.0
        frame = _add_info_overlay(
            frame,
            video_time_s=frame_idx / fps,
            avg_speed=avg_speed,
            total_counts={v: cumulative_counts.get(v, 0) + minute_counts.get(v, 0)
                          for v in VEHICLE_TYPES},
            current_counts=current_frame_counts,
            individual_speeds=individual_speeds,
            state_code=cur_state_code,
            state_label=cur_state_label,
            confidence=cur_confidence,
            frame_idx=frame_idx,
            n_active_tracks=len(active_tracks),
        )

        writer.write(frame)
        frame_idx += 1

        # Progress bar (every 2 s)
        wall_now = _time.time()
        if wall_now - last_progress_wall >= 2.0:
            _show_progress_bar(frame_idx, total_frames, wall_now - wall_start)
            last_progress_wall = wall_now

        # Stats every 30 s of video time
        if frame_idx % int(fps * 30) == 0:
            cum_tot = sum(cumulative_counts.values()) + sum(minute_counts.values())
            active_n = len(active_tracks)
            print(f"\n📊 Stats @ {frame_idx / fps:.0f}s video time:")
            print(f"   📈 Total vehicles so far: {cum_tot}")
            print(f"   ⚡ Current avg speed: {avg_speed:.1f} km/h")
            print(f"   🧪 Rejected speeds this minute: {minute_quality['rejected']} | recovered: {minute_quality['recovered']}")
            print(f"   🎯 Active tracks: {active_n}")
            for vt in VEHICLE_TYPES:
                c = cumulative_counts.get(vt, 0) + minute_counts.get(vt, 0)
                if c > 0:
                    print(f"   • {vt.upper()}: {c}")

        # Minute boundary: emit telemetry + classify
        if frame_idx % frames_per_minute == 0:
            total = sum(minute_counts.values())
            rec = {
                "record_time": datetime.now(),
                "avg_speed": round(avg_speed, 2),
                "count_car": minute_counts.get("car", 0),
                "count_truck": minute_counts.get("truck", 0),
                "count_bus": minute_counts.get("bus", 0),
                "count_motorcycle": minute_counts.get("motorcycle", 0),
                "count_bicycle": minute_counts.get("bicycle", 0),
                "total_vehicles": total,
            }
            records.append(rec)
            print(f"\n   📊 Minute {len(records)}: {avg_speed:.1f} km/h, {total} vehicles")
            if minute_quality["rejected"] or minute_quality["recovered"]:
                print(f"   🧪 Filtered speeds: {minute_quality['rejected']} | recovered tracks: {minute_quality['recovered']}")

            # Classify this minute
            _code, _lbl, _conf = _safe_classify_records(records)
            if _code is not None:
                cur_state_code, cur_state_label, cur_confidence = _code, _lbl, _conf
                print(f"   🏷️  State: {_lbl} ({_conf:.0%})")

            # Accumulate and reset
            for vt in VEHICLE_TYPES:
                cumulative_counts[vt] = cumulative_counts.get(vt, 0) + minute_counts.get(vt, 0)
            minute_counts = {v: 0 for v in VEHICLE_TYPES}
            minute_speeds.clear()
            minute_quality = {"stationary": 0, "rejected": 0, "recovered": 0}
            counted_tracks.clear()

    # Flush partial minute
    if frame_idx % frames_per_minute != 0 and (minute_speeds or any(minute_counts.values())):
        avg_speed = robust_speed_summary(minute_speeds) if minute_speeds else 0.0
        total = sum(minute_counts.values())
        records.append({
            "record_time": datetime.now(),
            "avg_speed": round(avg_speed, 2),
            "count_car": minute_counts.get("car", 0),
            "count_truck": minute_counts.get("truck", 0),
            "count_bus": minute_counts.get("bus", 0),
            "count_motorcycle": minute_counts.get("motorcycle", 0),
            "count_bicycle": minute_counts.get("bicycle", 0),
            "total_vehicles": total,
        })
        # Classify the partial minute
        _code, _lbl, _conf = _safe_classify_records(records)
        if _code is not None:
            cur_state_code, cur_state_label, cur_confidence = _code, _lbl, _conf
            print(f"\n   🏷️  Final partial minute: {_lbl} ({_conf:.0%})")

        for vt in VEHICLE_TYPES:
            cumulative_counts[vt] = cumulative_counts.get(vt, 0) + minute_counts.get(vt, 0)

    cap.release()
    writer.release()

    # Final summary (legacy style)
    _show_progress_bar(total_frames, total_frames, _time.time() - wall_start)
    wall_total = _time.time() - wall_start
    grand_total = sum(cumulative_counts.values())

    print(f"\n\n🎉 PROCESSING COMPLETE!")
    print(f"⏱️  Wall time: {wall_total / 60:.1f} min")
    print(f"📊 Frames processed: {frame_idx:,}")
    print(f"🎯 Processing speed: {frame_idx / wall_total:.1f} fps")
    print(f"📈 Detection summary:")
    for vt in VEHICLE_TYPES:
        c = cumulative_counts.get(vt, 0)
        if c > 0:
            print(f"   • {vt.upper()}: {c} vehicles")
    print(f"🚗 Total unique vehicles: {grand_total}")
    print(f"📁 Output: {output_path}")

    # Build classified DataFrame
    df_records = pd.DataFrame(records)
    df_classified_out: pd.DataFrame = df_records
    if model is not None and scaler is not None and label_mapping is not None and not df_records.empty:
        try:
            df_feat = engineer_features(df_records)
            if df_feat.empty and not df_records.empty:
                # Single-record clip: duplicate to satisfy diff()
                _df2 = pd.concat([df_records.iloc[[0]], df_records.iloc[[0]]], ignore_index=True)
                df_feat = engineer_features(_df2).iloc[-len(df_records):]
            X = scaler.transform(df_feat[FEATURE_COLS].values)
            X = np.nan_to_num(X, nan=0.0)
            proba = model.predict(X, verbose=0)
            pred_idx = proba.argmax(axis=1).astype(int)
            pred_codes = np.array([_decode_model_index(idx) for idx in pred_idx], dtype=int)
            df_feat["traffic_state"] = pred_codes
            df_feat["state_label"] = [label_mapping.get(int(c), "Unknown") for c in pred_codes]
            df_feat["confidence"] = proba.max(axis=1).round(4)
            df_classified_out = df_feat
            print("\n✅ Classification summary:")
            for code in sorted(df_classified_out["traffic_state"].unique()):
                lbl = label_mapping.get(code, "Unknown")
                cnt = (df_classified_out["traffic_state"] == code).sum()
                print(f"   {lbl:>10}: {cnt} records")
        except Exception as exc:
            print(f"⚠️ Final classification failed: {exc}")

    # Colab: trigger browser download popup
    if IN_COLAB and os.path.isfile(output_path):
        from google.colab import files as _colab_dl  # type: ignore[import-untyped]
        print(f"\n📥 Downloading annotated video...")
        _colab_dl.download(output_path)
        print(f"✅ Download complete: {output_path}")

    return output_path, df_classified_out


# EXECUTION — runs automatically when VIDEO_PATH is set
try:
    if VIDEO_PATH and os.path.isfile(VIDEO_PATH):
        _out_path, df_classified = generate_annotated_video(VIDEO_PATH)
        # Also set df_telemetry for Cell 3 compatibility
        df_telemetry = df_classified
        print(f"\n📁 Video saved: {_out_path}")
    else:
        print("⚠️ No video file available. Set VIDEO_PATH in Cell 1b or upload a clip.")
        df_telemetry = None
        df_classified = None
except Exception as e:
    print(f"🔴 Error processing clip: {e}")
    import traceback
    traceback.print_exc()
    df_telemetry = None
    df_classified = None
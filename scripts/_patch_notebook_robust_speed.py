import json
import pathlib
import shutil

nb_path = pathlib.Path(
    r"d:/dev/learning/vaaet/notebooks/02_production/traffic_analyzer.ipynb"
)
backup_path = nb_path.with_suffix(".ipynb.robust.bak")
shutil.copy2(nb_path, backup_path)

nb = json.loads(nb_path.read_text(encoding="utf-8"))


def get_cell_by_header(prefix: str) -> dict:
    for cell in nb["cells"]:
        source = cell.get("source", [])
        src = "".join(source) if isinstance(source, list) else str(source)
        first_line = src.splitlines()[0] if src else ""
        if first_line.startswith(prefix):
            return cell
    raise RuntimeError(f"Cell starting with {prefix!r} not found")


cell_imports = get_cell_by_header("# Cell 1 — Dependencies + Load Trained Model")
cell_main = get_cell_by_header(
    "# Cell 2b — Annotated Video Output with Full HUD + Classification"
)

imports_src = "".join(cell_imports["source"])
main_src = "".join(cell_main["source"])

old_import = "from src.perception.speed import estimate_speed, is_stationary, SmoothedSpeedTracker\n"
new_import = (
    "from src.perception.speed import (\n"
    "    TrackMotionStateTracker,\n"
    "    SmoothedSpeedTracker,\n"
    "    estimate_speed,\n"
    "    is_speed_measurement_reliable,\n"
    "    is_stationary,\n"
    "    robust_speed_summary,\n"
    ")\n"
)
if new_import in imports_src and old_import not in imports_src:
    print("Notebook already has robust speed imports. No patch needed.")
    raise SystemExit(0)
assert old_import in imports_src, "Import line not found"
imports_src = imports_src.replace(old_import, new_import, 1)

old_signature = (
    "def _draw_annotations(\n"
    "    frame: np.ndarray,\n"
    "    active_tracks: list,\n"
    "    speed_tracker: SmoothedSpeedTracker,\n"
    "    fps: float,\n"
    "    frame_h: int,\n"
    "    global_motion: float,\n"
    ") -> dict[int, float]:\n"
)
new_signature = (
    "def _draw_annotations(\n"
    "    frame: np.ndarray,\n"
    "    active_tracks: list,\n"
    "    speed_tracker: SmoothedSpeedTracker,\n"
    "    motion_state_tracker: TrackMotionStateTracker,\n"
    "    fps: float,\n"
    "    frame_h: int,\n"
    "    global_motion: float,\n"
    "    flow_tracking_ratio: float,\n"
    ") -> tuple[dict[int, float], dict[str, int]]:\n"
)
assert old_signature in main_src, "Draw signature not found"
main_src = main_src.replace(old_signature, new_signature, 1)

old_doc = "    Returns dict of {track_id: smoothed_speed} for the HUD panel.\n"
new_doc = (
    "    Returns ``(individual_speeds, frame_quality)`` where frame_quality tracks\n"
    "    stationary / rejected / recovered samples for minute-level telemetry.\n"
)
assert old_doc in main_src, "Draw doc return line not found"
main_src = main_src.replace(old_doc, new_doc, 1)

old_init = "    individual_speeds: dict[int, float] = {}\n"
new_init = (
    "    individual_speeds: dict[int, float] = {}\n"
    '    frame_quality: dict[str, int] = {"stationary": 0, "rejected": 0, "recovered": 0}\n'
)
assert old_init in main_src, "Draw init not found"
main_src = main_src.replace(old_init, new_init, 1)

old_draw_block = (
    "        # Stationary check FIRST\n"
    "        if is_stationary(track.history):\n"
    "            # Clear any stale speed history for this track\n"
    "            speed_tracker.remove_track(track.track_id)\n"
    '            cv2.putText(frame, "[S]",\n'
    "                        (cx - half_w, cy - half_h - 12),\n"
    "                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 255), 2)\n"
    "        else:\n"
    "            # Speed estimation only for moving vehicles\n"
    "            speed = estimate_speed(\n"
    "                track.history, fps=fps, frame_height=frame_h,\n"
    "                global_motion=global_motion, vehicle_type=track.vehicle_type,\n"
    "            )\n"
    "            smoothed = speed_tracker.update(track.track_id, speed)\n"
    "            if smoothed is not None:\n"
    "                individual_speeds[track.track_id] = smoothed\n"
    '                cv2.putText(frame, f"{smoothed:.0f}km/h",\n'
    "                            (cx - half_w, cy - half_h - 12),\n"
    "                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)\n"
)
new_draw_block = (
    '        recovered_gap = getattr(track, "recovered_after_gap", 0)\n'
    "        reliable = is_speed_measurement_reliable(\n"
    "            track.history,\n"
    "            flow_tracking_ratio=flow_tracking_ratio,\n"
    "            recovered_after_gap=recovered_gap,\n"
    "        )\n"
    "        speed = None\n"
    "        if reliable:\n"
    "            speed = estimate_speed(\n"
    "                track.history, fps=fps, frame_height=frame_h,\n"
    "                global_motion=global_motion, vehicle_type=track.vehicle_type,\n"
    "            )\n"
    "\n"
    "        stationary_now = motion_state_tracker.update(\n"
    "            track.track_id,\n"
    "            track.history,\n"
    "            candidate_speed=speed,\n"
    "        )\n"
    "\n"
    "        if stationary_now:\n"
    '            frame_quality["stationary"] += 1\n'
    "            speed_tracker.remove_track(track.track_id)\n"
    '            cv2.putText(frame, "[S]",\n'
    "                        (cx - half_w, cy - half_h - 12),\n"
    "                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 255), 2)\n"
    "        elif not reliable:\n"
    '            frame_quality["rejected"] += 1\n'
    "            if recovered_gap > 0:\n"
    '                frame_quality["recovered"] += 1\n'
    "            speed_tracker.remove_track(track.track_id)\n"
    '            tag = "[R]" if recovered_gap > 0 else "[?]"\n'
    "            tag_color = (0, 165, 255) if recovered_gap > 0 else (180, 180, 180)\n"
    "            cv2.putText(frame, tag,\n"
    "                        (cx - half_w, cy - half_h - 12),\n"
    "                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, tag_color, 2)\n"
    "        else:\n"
    "            smoothed = speed_tracker.update(track.track_id, speed)\n"
    "            if smoothed is not None:\n"
    "                individual_speeds[track.track_id] = smoothed\n"
    '                cv2.putText(frame, f"{smoothed:.0f}km/h",\n'
    "                            (cx - half_w, cy - half_h - 12),\n"
    "                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)\n"
)
assert old_draw_block in main_src, "Draw logic block not found"
main_src = main_src.replace(old_draw_block, new_draw_block, 1)

old_return = "    return individual_speeds\n"
new_return = "    return individual_speeds, frame_quality\n"
assert old_return in main_src, "Draw return not found"
main_src = main_src.replace(old_return, new_return, 1)

old_setup = "    speed_tracker = SmoothedSpeedTracker(window_size=10)\n"
new_setup = (
    "    speed_tracker = SmoothedSpeedTracker(window_size=10)\n"
    "    motion_state_tracker = TrackMotionStateTracker()\n"
)
assert old_setup in main_src, "Speed tracker setup not found"
main_src = main_src.replace(old_setup, new_setup, 1)

old_acc = (
    "    minute_speeds: list[float] = []\n"
    "    counted_tracks: set[int] = set()\n"
    "    cumulative_counts: dict[str, int] = {v: 0 for v in VEHICLE_TYPES}\n"
)
new_acc = (
    "    minute_speeds: list[float] = []\n"
    '    minute_quality: dict[str, int] = {"stationary": 0, "rejected": 0, "recovered": 0}\n'
    "    counted_tracks: set[int] = set()\n"
    "    cumulative_counts: dict[str, int] = {v: 0 for v in VEHICLE_TYPES}\n"
)
assert old_acc in main_src, "Accumulator block not found"
main_src = main_src.replace(old_acc, new_acc, 1)

old_track_block = (
    "        # 3. Track\n"
    "        det_tuples = [(d.centroid, d.vehicle_type) for d in detections]\n"
    "        active_tracks = tracker.update(det_tuples)\n"
)
new_track_block = (
    "        # 3. Track\n"
    "        det_tuples = [(d.centroid, d.vehicle_type) for d in detections]\n"
    "        active_tracks = tracker.update(det_tuples)\n"
    "        for pruned_track_id in tracker.last_pruned_track_ids:\n"
    "            speed_tracker.remove_track(pruned_track_id)\n"
    "            motion_state_tracker.remove_track(pruned_track_id)\n"
    "            counted_tracks.discard(pruned_track_id)\n"
    "        flow_tracking_ratio = (\n"
    "            flow_estimator.last_tracking_ratio\n"
    "            if flow_estimator.last_total_points > 0\n"
    "            else 1.0\n"
    "        )\n"
)
assert old_track_block in main_src, "Track block not found"
main_src = main_src.replace(old_track_block, new_track_block, 1)

old_call_block = (
    "        # 5. Draw annotations + collect individual speeds\n"
    "        #    (speed_tracker.update is called ONCE per track inside here;\n"
    "        #     stationary vehicles are skipped entirely — see _draw_annotations)\n"
    "        individual_speeds = _draw_annotations(\n"
    "            frame, active_tracks, speed_tracker, fps, frame_h, global_motion,\n"
    "        )\n"
    "\n"
    "        # 6. Accumulate for telemetry\n"
    "        #    NOTE: speeds already computed in _draw_annotations — do NOT call\n"
    "        #    speed_tracker.update again (would corrupt the moving average).\n"
    "        for spd in individual_speeds.values():\n"
    "            minute_speeds.append(spd)\n"
)
new_call_block = (
    "        # 5. Draw annotations + collect individual speeds\n"
    "        #    (speed_tracker.update is called ONCE per track inside here;\n"
    "        #     low-confidence and recovered tracks are filtered before telemetry.)\n"
    "        individual_speeds, frame_quality = _draw_annotations(\n"
    "            frame, active_tracks, speed_tracker, motion_state_tracker, fps, frame_h,\n"
    "            global_motion, flow_tracking_ratio,\n"
    "        )\n"
    "\n"
    "        # 6. Accumulate for telemetry\n"
    "        #    NOTE: speeds already computed in _draw_annotations — do NOT call\n"
    "        #    speed_tracker.update again (would corrupt the moving average).\n"
    "        for spd in individual_speeds.values():\n"
    "            minute_speeds.append(spd)\n"
    "        for key, value in frame_quality.items():\n"
    "            minute_quality[key] = minute_quality.get(key, 0) + value\n"
)
assert old_call_block in main_src, "Draw call block not found"
main_src = main_src.replace(old_call_block, new_call_block, 1)

old_avg_expr = "float(np.mean(minute_speeds)) if minute_speeds else 0.0"
new_avg_expr = "robust_speed_summary(minute_speeds) if minute_speeds else 0.0"
assert old_avg_expr in main_src, "Average speed expression not found"
main_src = main_src.replace(old_avg_expr, new_avg_expr)

old_stats_line = '            print(f"   ⚡ Current avg speed: {avg_speed:.1f} km/h")\n'
new_stats_line = (
    '            print(f"   ⚡ Current avg speed: {avg_speed:.1f} km/h")\n'
    "            print(f\"   🧪 Rejected speeds this minute: {minute_quality['rejected']} | recovered: {minute_quality['recovered']}\")\n"
)
assert old_stats_line in main_src, "Stats speed line not found"
main_src = main_src.replace(old_stats_line, new_stats_line, 1)

old_minute_print = '            print(f"\\n   📊 Minute {len(records)}: {avg_speed:.1f} km/h, {total} vehicles")\n'
new_minute_print = (
    '            print(f"\\n   📊 Minute {len(records)}: {avg_speed:.1f} km/h, {total} vehicles")\n'
    '            if minute_quality["rejected"] or minute_quality["recovered"]:\n'
    "                print(f\"   🧪 Filtered speeds: {minute_quality['rejected']} | recovered tracks: {minute_quality['recovered']}\")\n"
)
assert old_minute_print in main_src, "Minute print line not found"
main_src = main_src.replace(old_minute_print, new_minute_print, 1)

old_reset = (
    "            minute_counts = {v: 0 for v in VEHICLE_TYPES}\n"
    "            minute_speeds.clear()\n"
    "            counted_tracks.clear()\n"
)
new_reset = (
    "            minute_counts = {v: 0 for v in VEHICLE_TYPES}\n"
    "            minute_speeds.clear()\n"
    '            minute_quality = {"stationary": 0, "rejected": 0, "recovered": 0}\n'
    "            counted_tracks.clear()\n"
)
assert old_reset in main_src, "Minute reset block not found"
main_src = main_src.replace(old_reset, new_reset, 1)

cell_imports["source"] = imports_src.splitlines(keepends=True)
cell_main["source"] = main_src.splitlines(keepends=True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Patched notebook successfully. Backup: {backup_path}")

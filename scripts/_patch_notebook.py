"""
Patch traffic_analyzer.ipynb Cell 2b (index 6):
  1. Replace _classify_current with _safe_classify_records
  2. Update early-trigger call site
  3. Replace minute-boundary inline classification
  4. Replace partial-flush inline classification
  5. Fix final Build-classified-DataFrame block
"""

import json
import pathlib
import shutil

nb_path = pathlib.Path(
    r"d:/dev/learning/vaaet/notebooks/02_production/traffic_analyzer.ipynb"
)
shutil.copy(nb_path, nb_path.with_suffix(".ipynb.bak"))

nb = json.loads(nb_path.read_text(encoding="utf-8"))
cell = nb["cells"][6]
src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

changes = 0

# PATCH 1: replace _classify_current with _safe_classify_records
old1 = (
    "\ndef _classify_current(records, minute_counts, minute_speeds, label_mapping):\n"
    '    """Build a temporary telemetry record and classify it.\n\n'
    "    Returns ``(state_code, state_label, confidence)`` or ``(None, None, None)``.\n"
    '    """\n'
    "    if model is None or scaler is None or label_mapping is None:\n"
    "        return None, None, None\n"
    "    try:\n"
    "        avg_speed = float(np.mean(minute_speeds)) if minute_speeds else 0.0\n"
    "        total = sum(minute_counts.values())\n"
    "        tmp_rec = {\n"
    '            "record_time": datetime.now(),\n'
    '            "avg_speed": round(avg_speed, 2),\n'
    '            "count_car": minute_counts.get("car", 0),\n'
    '            "count_truck": minute_counts.get("truck", 0),\n'
    '            "count_bus": minute_counts.get("bus", 0),\n'
    '            "count_motorcycle": minute_counts.get("motorcycle", 0),\n'
    '            "count_bicycle": minute_counts.get("bicycle", 0),\n'
    '            "total_vehicles": total,\n'
    "        }\n"
    "        # Append temporarily to get feature engineering context (delta, etc.)\n"
    "        full = records + [tmp_rec]\n"
    "        df_tmp = pd.DataFrame(full[-2:]) if len(full) >= 2 else pd.DataFrame(full)\n"
    "        df_feat = engineer_features(df_tmp)\n"
    "        X = scaler.transform(df_feat[FEATURE_COLS].values[-1:])\n"
    "        proba = model.predict(X, verbose=0)\n"
    "        code = int(proba.argmax(axis=1)[0])\n"
    "        conf = float(proba.max(axis=1)[0])\n"
    '        lbl = label_mapping.get(code, "Unknown")\n'
    "        return code, lbl, conf\n"
    "    except Exception:\n"
    "        return None, None, None\n"
)

new1 = (
    "\ndef _safe_classify_records(records: list) -> tuple:\n"
    '    """Classify the most recent record, NaN-safe for short inputs.\n\n'
    "    engineer_features drops the first row because diff() produces NaN deltas.\n"
    "    A single-record input produces an empty DataFrame, causing silent failure.\n"
    "    Fix: duplicate the lone record so diff() yields delta=0 for row 1.\n\n"
    "    Returns (state_code, state_label, confidence) or (None, None, None).\n"
    '    """\n'
    "    if model is None or scaler is None or label_mapping is None:\n"
    "        return None, None, None\n"
    "    if not records:\n"
    "        return None, None, None\n"
    "    try:\n"
    "        # Use last 2 records to give diff() a valid previous row\n"
    "        df = pd.DataFrame(records[-2:])\n"
    "        df_feat = engineer_features(df)\n"
    "        if df_feat.empty:\n"
    "            # Only 1 record available: duplicate so delta=0 for row 1\n"
    "            df2 = pd.concat([df.iloc[[0]], df.iloc[[0]]], ignore_index=True)\n"
    "            df_feat = engineer_features(df2)\n"
    "        if df_feat.empty:\n"
    "            return None, None, None\n"
    "        X = scaler.transform(df_feat[FEATURE_COLS].values[-1:])\n"
    "        X = np.nan_to_num(X, nan=0.0)  # safety net for any residual NaN\n"
    "        proba = model.predict(X, verbose=0)\n"
    "        code = int(proba.argmax(axis=1)[0])\n"
    "        conf = float(proba.max(axis=1)[0])\n"
    '        lbl = label_mapping.get(code, "Unknown")\n'
    "        return code, lbl, conf\n"
    "    except Exception:\n"
    "        return None, None, None\n"
)

assert old1 in src, f"PATCH 1 old string not found!\nExpected snippet:\n{old1[:200]}"
src = src.replace(old1, new1, 1)
changes += 1
print("✅ Patch 1: _safe_classify_records")

# PATCH 2: early trigger — replace _classify_current call
old2 = (
    "            code, lbl, conf = _classify_current(\n"
    "                records, minute_counts, minute_speeds, label_mapping,\n"
    "            )\n"
)

new2 = (
    "            avg_speed_e = float(np.mean(minute_speeds)) if minute_speeds else 0.0\n"
    "            tmp_rec_e = {\n"
    '                "record_time": datetime.now(),\n'
    '                "avg_speed": round(avg_speed_e, 2),\n'
    '                "count_car": minute_counts.get("car", 0),\n'
    '                "count_truck": minute_counts.get("truck", 0),\n'
    '                "count_bus": minute_counts.get("bus", 0),\n'
    '                "count_motorcycle": minute_counts.get("motorcycle", 0),\n'
    '                "count_bicycle": minute_counts.get("bicycle", 0),\n'
    '                "total_vehicles": sum(minute_counts.values()),\n'
    "            }\n"
    "            code, lbl, conf = _safe_classify_records(records + [tmp_rec_e])\n"
)

assert old2 in src, f"PATCH 2 old string not found!\nExpected:\n{old2}"
src = src.replace(old2, new2, 1)
changes += 1
print("✅ Patch 2: early trigger call site")

# PATCH 3: minute-boundary inline classification
# (Read exact content from _mb_block.txt — no typos in real file)
old3 = (
    "# Classify this minute if model is loaded\n"
    "            if model is not None and scaler is not None and label_mapping is not None:\n"
    "                try:\n"
    "                    df_tmp = pd.DataFrame(records[-1:])\n"
    "                    df_feat = engineer_features(df_tmp)\n"
    "                    X = scaler.transform(df_feat[FEATURE_COLS].values)\n"
    "                    proba = model.predict(X, verbose=0)\n"
    "                    cur_state_code = int(proba.argmax(axis=1)[0])\n"
    "                    cur_confidence = float(proba.max(axis=1)[0])\n"
    '                    cur_state_label = label_mapping.get(cur_state_code, "Unknown")\n'
    '                    print(f"   \U0001f3f7\ufe0f  State: {cur_state_label} ({cur_confidence:.0%})")\n'
    "                except Exception as exc:\n"
    '                    print(f"   \u26a0\ufe0f Classification error: {exc}")\n'
)

new3 = (
    "# Classify this minute\n"
    "            _code, _lbl, _conf = _safe_classify_records(records)\n"
    "            if _code is not None:\n"
    "                cur_state_code, cur_state_label, cur_confidence = _code, _lbl, _conf\n"
    '                print(f"   \U0001f3f7\ufe0f  State: {_lbl} ({_conf:.0%})")\n'
)

assert old3 in src, f"PATCH 3 old string not found!\nExpected:\n{old3}"
src = src.replace(old3, new3, 1)
changes += 1
print("✅ Patch 3: minute-boundary classification")

# PATCH 4: partial-flush inline classification
old4 = (
    "        # Classify the partial minute too\n"
    "        if model is not None and scaler is not None and label_mapping is not None:\n"
    "            try:\n"
    "                df_tmp = pd.DataFrame(records[-1:])\n"
    "                df_feat = engineer_features(df_tmp)\n"
    "                X = scaler.transform(df_feat[FEATURE_COLS].values)\n"
    "                proba = model.predict(X, verbose=0)\n"
    "                cur_state_code = int(proba.argmax(axis=1)[0])\n"
    "                cur_confidence = float(proba.max(axis=1)[0])\n"
    '                cur_state_label = label_mapping.get(cur_state_code, "Unknown")\n'
    '                print(f"\\n   \U0001f3f7\ufe0f  Final partial minute: {cur_state_label} ({cur_confidence:.0%})")\n'
    "            except Exception as exc:\n"
    '                print(f"   \u26a0\ufe0f Partial classification error: {exc}")\n'
)

new4 = (
    "        # Classify the partial minute\n"
    "        _code, _lbl, _conf = _safe_classify_records(records)\n"
    "        if _code is not None:\n"
    "            cur_state_code, cur_state_label, cur_confidence = _code, _lbl, _conf\n"
    '            print(f"\\n   \U0001f3f7\ufe0f  Final partial minute: {_lbl} ({_conf:.0%})")\n'
)

assert old4 in src, f"PATCH 4 old string not found!\nExpected:\n{old4}"
src = src.replace(old4, new4, 1)
changes += 1
print("✅ Patch 4: partial-flush classification")

# PATCH 5: final Build-classified-DataFrame block
old5 = (
    "            df_feat = engineer_features(df_records)\n"
    "            X = scaler.transform(df_feat[FEATURE_COLS].values)\n"
    "            proba = model.predict(X, verbose=0)\n"
    '            df_feat["traffic_state"] = proba.argmax(axis=1)\n'
)

new5 = (
    "            df_feat = engineer_features(df_records)\n"
    "            if df_feat.empty and not df_records.empty:\n"
    "                # Single-record clip: duplicate to satisfy diff()\n"
    "                _df2 = pd.concat([df_records.iloc[[0]], df_records.iloc[[0]]], ignore_index=True)\n"
    "                df_feat = engineer_features(_df2).iloc[-len(df_records):]\n"
    "            X = scaler.transform(df_feat[FEATURE_COLS].values)\n"
    "            X = np.nan_to_num(X, nan=0.0)\n"
    "            proba = model.predict(X, verbose=0)\n"
    '            df_feat["traffic_state"] = proba.argmax(axis=1)\n'
)

assert old5 in src, f"PATCH 5 old string not found!\nExpected:\n{old5}"
src = src.replace(old5, new5, 1)
changes += 1
print("✅ Patch 5: final DataFrame classification (NaN-safe)")

# Write back
cell["source"] = src
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"\n🎉 Done — {changes} patches applied. Notebook saved.")
print(f"📁 Backup at: {nb_path.with_suffix('.ipynb.bak')}")

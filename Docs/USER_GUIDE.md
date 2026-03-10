<!-- context: VAAET/docs/USER_GUIDE.md -- User guide for VAAET.
Complements README.md (overview) and DDS.md (technical design). -->

# User Guide -- VAAET

## What is VAAET?

A computer vision system for analyzing vehicular traffic on the General Manuel Belgrano Bridge using YOLO 11, Optical Flow, and MLP. It processes surveillance video, classifies traffic state, and optionally persists results to PostgreSQL.

---

## Quick Start

### Module 1 -- Data Preparation (run once)

1. Open `notebooks/01_data_prep/data_preparation.ipynb` in Google Colab
2. Run all cells in order (Cell 0 through Cell 8)
3. Configure DB credentials in Cell 2 via environment variables: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
4. The system extracts telemetry, engineers 14 features, trains an MLP classifier
5. Artifacts are exported to `models/intelligence/`

### Module 2 -- Production (ongoing)

1. Open `notebooks/02_production/traffic_analyzer.ipynb` in Google Colab
2. Run Cell 0 (environment setup) and Cell 1 (load trained model)
3. Upload your video with name: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`
4. Run Cell 2 to process the clip (YOLO 11 + SORT + speed estimation)
5. Run Cell 3 to classify traffic state
6. Run Cell 4 to persist results to DB (optional)
7. Run Cell 5 for HITL feedback (optional)
8. Run Cell 6 for visualization dashboard

---

## Video Requirements

- **Format**: MP4
- **Name**: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4` (strict). Non-compliant names are rejected.

---

## Automatic Model Selection (YOLO 11)

| Duration | Model |
|---|---|
| <= 1h | yolo11x.pt |
| 1-3h | yolo11l.pt |
| 3-6h | yolo11m.pt |
| 6-12h | yolo11s.pt |
| > 12h | yolo11n.pt |

Note: If local files are named "yolov11*.pt", they are automatically normalized to "yolo11*.pt".

---

## Database (optional)

- Persistence is optional -- the system works without a database
- Uses environment variables if available: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- Frequency: one record per minute with average speed and counts by type
- Schema: 3 tables -- `traffic_data` (legacy), `telemetry_raw` (14 features), `traffic_classifications` (predictions + HITL)
- No JSON/CSV files generated locally

---

## Traffic States

The classifier outputs one of 4 states per minute:

| State | Code | Description |
|---|---|---|
| Normal | 0 | Free flow, typical speeds 40-80 km/h |
| Reduced | 1 | Slower flow, moderate volume |
| Congested | 2 | Very slow flow, high volume |
| Accident | 3 | Near-zero speeds with sudden deceleration |

---

## Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

On Google Colab, dependencies are installed automatically in the first cell.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Missing YOLO weights | Auto-diagnosis downloads them automatically |
| Memory errors on long videos | Frame skipping and memory cleanup are built-in |
| DB connection errors | Check environment variables or run without DB |
| Speeds showing 0 for stationary vehicles | Expected and correct behavior |
| GPU not available on Colab | System falls back to CPU (slower) |
| Session disconnects on Colab | Download processed video before closing |

---

## Output

- **Module 2 Cell 2**: Processed video with bounding boxes, type + ID, speed, informational HUD
- **Module 2 Cell 3**: Traffic state classification (Normal/Reduced/Congested/Accident)
- **Module 2 Cell 6**: Visualization dashboard with charts and metrics

---

## Synthetic Demos

If you don't have real bridge footage, Module 0 (archived) includes a synthetic video generator:

1. Open `archive/00_bootstrap/01_legacy_collection.ipynb`
2. Run Cells 8-9 (Synthetic Video Generator + Demo Runner)

Available scenarios: light, normal, busy, mixed, stationary_test.

---

## Known Limitations

- **Speed without ground truth**: Precision depends on manual calibration of `pixels_per_meter`
- **Tracking without re-ID**: If a vehicle is occluded >1 second, it loses its ID
- **Colab ephemeral**: Files are lost when session closes -- download before closing
- **GPU not guaranteed**: At peak hours on Colab Free, may be assigned CPU only (~10x slower)
- **Auto-labeling is not ground truth**: Labels are engineering proxies, not human-validated

For a complete analysis, see [BIAS_AND_LIMITATIONS.md](BIAS_AND_LIMITATIONS.md).

---

## Related Documentation

- [README.md](../README.md) -- Overview and requirements
- [DDS.md](DDS.md) -- Detailed technical design
- [KPIs/KPIs.md](KPIs/KPIs.md) -- Metrics and validation guide
- [DATA_LINEAGE.md](DATA_LINEAGE.md) -- Data lineage
- [docs/adr/](adr/) -- Architecture Decision Records

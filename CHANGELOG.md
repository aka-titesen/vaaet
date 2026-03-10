# Changelog

All notable changes to the VAAET project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.0.0] - 2025-07-14

### Added

- Three-module architecture with shared `src/` code (ADR-009)
- `src/config.py` — single source of truth for constants, paths, thresholds
- `src/db.py` — SQLAlchemy engine factory with environment variable credentials
- `src/features.py` — shared feature engineering (9 → 14 columns)
- `src/labeling.py` — shared auto-labeling rules (4 traffic states)
- `src/perception/detector.py` — YOLODetector wrapper class
- `src/perception/tracker.py` — SORTTracker wrapper class
- `src/perception/speed.py` — physics-based speed estimation
- Module 2 production notebook: `notebooks/02_production/traffic_analyzer.ipynb`
- Self-improving feedback loop with HITL corrections
- ADR-009: Modular three-stage architecture with shared src/
- AGENTS.md rewritten in English with full architecture documentation

### Changed

- Module 0 (bootstrap) archived to `archive/00_bootstrap/`
- Module 1 renamed and moved to `notebooks/01_data_prep/data_preparation.ipynb`
- ADR-001 through ADR-007 superseded by ADR-009
- ADR-008 fixed: Input(13,) corrected to Input(14,) to match 14 canonical features
- All documentation translated to English
- README.md rewritten with three-module architecture diagram
- CONTRIBUTING.md updated for modular workflow
- Project structure reorganized following SOLID, YAGNI, KISS principles

### Removed

- `vaaet.ipynb` (duplicate of bootstrap notebook)
- Old directory structure: `notebooks/phase_1_perception/`, `notebooks/phase_2_intelligence/`
- `src/utils/` placeholder directory
- Spanish-language documentation (replaced with English)

## [2.0.0] - 2025-03-07

### Added

- Module 1: Traffic state classifier with TensorFlow/Keras MLP
- 4 traffic states: Normal, Reduced, Congested, Accident
- Feature engineering: 9 raw fields → 14 engineered features
- Auto-labeling with traffic engineering rules
- Class balancing with SMOTE (imbalanced-learn)
- Two new tables: `telemetry_raw` (14 features + FK), `traffic_classifications` (prediction + HITL)
- Intelligence pipeline diagram (Mermaid)
- Extended ERD diagram with 3 tables and FK chain
- ADR-008: TensorFlow/Keras for traffic classification

### Changed

- Project restructured: notebooks/, models/, data/, src/, docs/
- Documentation updated: README, DDS, DATA_LINEAGE, KPIs, BIAS_AND_LIMITATIONS, AGENTS, CONTRIBUTING
- requirements.txt expanded with 7 new dependencies (tensorflow, pandas, sqlalchemy, etc.)
- .gitignore updated for *.keras, *.joblib, data/processed/*.csv

## [1.0.0] - 2025-03-06

### Added

- Complete vehicular traffic analysis pipeline for Gral. Manuel Belgrano Bridge
- Detection and classification with YOLO 11 (5 variants: n/s/m/l/x)
- Automatic model selection based on video duration
- Persistent tracking with lightweight SORT
- Hybrid speed calculation: 70% physics + 30% MLP smoother
- Camera motion compensation via Optical Flow (Lucas-Kanade)
- Adaptive perspective correction by Y-coordinate
- Ultra-conservative stationary vehicle detection (AND-conjunction)
- Multi-camera support: automatic layout detection (1, 2, 4 views)
- Optional PostgreSQL persistence (AWS RDS) per minute
- Output video with annotations, overlays, and informational HUD
- Synthetic video generator for portfolio demos
- Universal upload interface (Colab + local)
- Google Colab Free/Pro optimization (frame skipping, memory cleanup)
- Documentation: PRD, DDS, User Guide, KPIs, ADRs
- Documentation infrastructure: AGENTS.md, llms.txt, Mermaid diagrams, data lineage

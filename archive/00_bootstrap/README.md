# Archive — Bootstrap Module (DEPRECATED)

> **Status**: ARCHIVED — This notebook is historical and must NOT be modified.

## What is this?

`01_legacy_collection.ipynb` was the **Phase 1 (Perception)** notebook of VAAET. It used YOLO 11 + OpenCV + SORT to process surveillance video clips from the General Manuel Belgrano Bridge and produce per-minute telemetry records stored in `traffic_data` (PostgreSQL).

## Why is it archived?

Phase 1 fulfilled its purpose: generating the initial dataset of ~2000 telemetry records. That data now lives in `data/raw/traffic_data.backup` and serves as the training foundation for the intelligence layer (Phase 2).

The active project has moved to a **three-module architecture** (see [ADR-009](../../docs/adr/ADR-009-modular-three-stage-architecture.md)):

| Module | Location | Purpose | Runs |
|---|---|---|---|
| **0 — Bootstrap** | `archive/00_bootstrap/` (here) | Initial data generation | Never again |
| **1 — Data Preparation** | `notebooks/01_data_prep/` | Feature engineering + model training | Once |
| **2 — Production** | `notebooks/02_production/` | YOLO + classifier + feedback loop | Always |

## Can I still run it?

Technically yes — it's a self-contained Colab notebook. But there is no reason to: the data it would produce already exists in the backup. Running it again would require video footage from the bridge cameras, which is not distributed with the project.

## Related documentation

- [ADR-001](../../docs/adr/ADR-001-notebook-monolitico.md) through [ADR-007](../../docs/adr/ADR-007-google-colab-como-runtime.md) — Original architectural decisions (now superseded by ADR-009)
- [CHANGELOG v1.0.0](../../CHANGELOG.md) — Original release notes

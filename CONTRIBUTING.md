# Contributing to VAAET

Thank you for your interest in contributing to VAAET. This document establishes the conventions and rules for modifying the project.

## Architecture — Fundamental Rules

1. **Shared code lives in `src/`** — reusable modules (config, db, features, labeling, perception) shared by all notebooks
2. **Notebooks are orchestrators** — they call `src/` functions and provide the Colab UI wrapper
3. **Module 0 (`archive/00_bootstrap/`) is FROZEN** — never modify it
4. **Run on Google Colab** — all changes must be compatible with Colab Free Tier
5. **Read `AGENTS.md`** before starting — contains architectural boundaries and the Always/Ask/Never governance system

## Code Conventions

### General Style

- Python 3.8+ compatible
- PEP 8 formatting (except line length, which may extend in notebooks)
- Type hints on all public functions in `src/` modules
- Docstrings in English for all functions

### Prints with Emoji

The system uses `print()` with emoji prefixes instead of `logging`:

```python
print("✅ Operation successful")
print("⚠️ Warning: parameter out of range")
print("🔴 Error: could not connect to DB")
print("📊 Result: 42 vehicles detected")
```

### Configuration

- **All constants and thresholds**: `src/config.py` (single source of truth)
- **DB credentials**: `src/db.py` via environment variables only
- **Credentials**: NEVER hardcoded. Use environment variables or `getpass`

## Contribution Workflow

1. Read the relevant ADR in `docs/adr/` if your change affects an architectural decision
2. If no ADR exists and your change is significant, draft one before implementing
3. Verify all active notebooks run without errors after your changes
4. Module 1: Verify F1-macro ≥ 0.85 after retraining
5. Module 2: Verify the perception + classification pipeline produces valid output
6. Update corresponding documentation if your change alters observable behavior

## Documentation Structure

| File | Purpose | When to update |
|---|---|---|
| `README.md` | Overview and usage | Changes in features or dependencies |
| `AGENTS.md` | Context for AI agents | Changes in architecture or rules |
| `docs/PRD.md` | Product requirements | New requirements or functional changes |
| `docs/DDS.md` | Technical design | Changes in algorithms or components |
| `docs/USER_GUIDE.md` | User guide | Changes in UX or execution flow |
| `docs/KPIs/KPIs.md` | Metrics | New metrics or benchmarks |
| `docs/adr/` | Architecture decisions | New or revoked decisions |
| `CHANGELOG.md` | Change history | Every PR or significant change |

## ADRs (Architecture Decision Records)

If you want to propose a change that contradicts an existing decision:

1. Read the original ADR in `docs/adr/`
2. Create a new ADR with the next available ADR-XXX number
3. Use status "Proposed" until approved
4. Reference the ADR being superseded

Format: see any existing ADR in `docs/adr/` as a template.

## What NOT to Do

- Hardcode AWS RDS credentials
- Modify `archive/00_bootstrap/01_legacy_collection.ipynb`
- Commit `.pt` files (YOLO models) or `.keras`/`.joblib` artifacts
- Delete `test_sistema()` or the synthetic demo generator from Module 0
- Break compatibility with Colab Free Tier
- Modify DB table schemas without a new ADR
- Remove HITL fields from `traffic_classifications`

## Module-Specific Guidelines

### Module 1 (Data Preparation)

- Run `data_preparation.ipynb` fully after changes
- Verify F1-macro ≥ 0.85 on test set
- Read [ADR-008](docs/adr/ADR-008-tensorflow-keras-traffic-classifier.md) before modifying auto-labeling thresholds or MLP architecture
- Do not commit `*.keras`, `*.joblib`, or `data/processed/*.csv`

### Module 2 (Production)

- Verify perception pipeline produces valid telemetry DataFrame
- Verify classification assigns one of 4 valid states
- Verify persistence writes to both `telemetry_raw` and `traffic_classifications`
- Read [ADR-009](docs/adr/ADR-009-modular-three-stage-architecture.md) for the full architecture specification

### Shared `src/` Modules

- All modules must remain notebook-importable (no CLI entrypoints, no `if __name__` blocks)
- `config.py` is the single source of truth — do not duplicate constants elsewhere
- `db.py` is the single point of DB configuration — do not create alternative connection methods
- 14 features in `FEATURE_COLS` are canonical — do not add/remove without updating all modules

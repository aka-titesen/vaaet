---
name: vaaet-monorepo-architecture
description: Guide safe evolution of VAAET into a single Git monorepo. Use for repository layout, ML/API/web boundaries, DVC ownership, monorepo migration planning, workspace packaging, path-scoped CI, or replacing the current multi-repo decision without contract drift.
---

# VAAET Monorepo Architecture

## Overview

Use one repository root with independently deployable components, not a nested monorepo. Preserve VAAET's ML contracts while creating a clear future boundary between the Python package, API, and web interface.

## Use one Git and DVC root

Keep a single `.git`, `.dvc`, remote configuration, top-level CI surface, and shared architecture documentation. Do not create nested Git repositories, autonomous DVC remotes, or an inner monorepo inside `vaaet-ml/` or `vaaet-app/`.

Use this target layout for a future authorized migration:

```text
vaaet/
├─ vaaet-ml/          # Python package, notebooks, tests, migrations, artifacts
├─ vaaet-app/
│  ├─ web/            # frontend
│  └─ api/            # backend and serving
├─ docs/              # shared ADRs and architecture
├─ .dvc/              # one DVC configuration
└─ .github/           # root CI
```


## Keep components independently bounded

Keep all Python ML behavior in `vaaet-ml/`: the `vaaet-ml` package, notebooks as Colab orchestrators, tests, migrations, and governed artifacts. Do not move notebook behavior into web endpoints or duplicate its feature and inference logic.

Let `vaaet-app/web/` consume only a versioned API. It must not access PostgreSQL, DVC, Google Drive, artifact binaries, or Python modules directly.

Let `vaaet-app/api/` install `vaaet-ml` from the local workspace and use its public contracts to validate and load bundles. Do not reimplement feature engineering, model loading, state policy, or Accident handling. Version an HTTP contract before exposing it; keep framework selection open until explicitly authorized.

## Govern a future migration

Do not move files until an ADR supersedes ADR-0012 while preserving its portable bundle boundary. Inventory paths, notebook setup cells, packaging, DVC metadata, CI, scripts, documentation, and links before the move.

Use traceable `git mv` operations and stage the migration: establish the root layout, relocate the ML project, repair path-aware tooling and links, then introduce application components separately. Do not combine a structural move with feature, model, schema, remote, role, or dependency changes.

Keep DVC at the root and track `vaaet-ml/artifacts/traffic-state/` as the existing atomic four-file bundle. Preserve checksums, manifests, input locks, immutable seed/HITL packages, holdouts, and the configured remotes.

## Preserve the ML serving contract

Keep bundle v2 as the only ML/API exchange: `traffic_classifier.keras`, `feature_scaler.joblib`, `label_mapping.joblib`, and `model-manifest.json`. Require `vaaet.artifacts.validate_manifest()` before deserializing a bundle in the API.

Preserve the 19 `FEATURE_COLS`, learned states `Normal`, `Reduced`, `Congested`, and the human-only publication of `Accident`. Preserve lifecycle, input policy, `production_eligible`, `promotion_blockers`, and artifact eligibility; relocating code never promotes a bundle.

Use the API as the sole web/ML boundary. Do not make the frontend a DVC client or database client, and do not expose a raw filesystem path, a model binary, secrets, or human-review data through the HTTP contract.

## Keep validation and ownership explicit

Keep shared ADRs, security policy, DVC configuration, and root CI at the repository root. Scope CI by changed paths so ML and application checks can run independently while preserving full integration checks at boundary changes.

Validate the relocated ML project with its existing Ruff, pytest, compilation, notebook parsing, Markdown-link, and diff gates. Add app-specific checks only with its future framework. Validate cross-boundary behavior with a verified bundle and a versioned API contract when the API exists.

Reject a migration if package installation, Colab setup, DVC pull, manifest validation, or existing ML tests regress. Keep GPU, Drive, DVC remote, and live PostgreSQL validation manual and explicit.

---

Reject these antipatterns:

- Do not create a second `.git`, nested monorepo, independent DVC remote, or duplicate artifact registry.
- Do not let web import Python, load a model, access DVC, or connect directly to the database.
- Do not copy ML code into the API or bypass manifest validation to accelerate serving.
- Do not move paths and change model behavior, schemas, dependencies, DVC remotes, or permissions in one migration.
- Do not update ADR-0012 silently or add application scaffolding before the architecture decision and scope are approved.

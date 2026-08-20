---
name: vaaet-dvc-registry-comparison
description: Audit and safely compare VAAET model bundles through the DVC registry. Use for DVC and Google Drive artifact history, manifest-first candidate comparison, immutable holdout compatibility, Colab-private ipywidgets evaluation, model eligibility, or registry troubleshooting.
---

# VAAET DVC Registry Comparison

## Overview

Treat DVC plus its configured Google Drive remote as VAAET's model registry. Compare evidence and compatible candidates without mutating the active bundle, exposing artifacts, or turning a Colab runtime into a public service.

## Preserve registry and bundle boundaries

Follow ADR-0011 and ADR-0012. Keep DVC as the registry and version `artifacts/traffic-state/` as one atomic bundle: Keras model, scaler, label mapping, and `model-manifest.json`. Never version the binary files directly in Git or register individual bundle files separately.

Run `vaaet.artifacts.validate_manifest()` before loading Keras or joblib. Preserve DVC remotes, `current.json`, seed snapshots, HITL catalogs, frozen holdouts, and input locks; comparison is read-only and must not update any pointer or registry state.


## Materialize candidates without side effects

Identify each candidate with an explicit Git/DVC revision. Materialize each selection in a unique temporary or run-specific directory, never over the active workspace bundle. Validate both manifests before inspecting metrics or deserializing a model.

Use manifest comparison before expensive evaluation. Show model version, lifecycle, supervision, input policy, `production_eligible`, `promotion_blockers`, data provenance, metrics, human holdout descriptor, and training input lock. Keep a bounded read-only cache and release models no longer selected to remain compatible with Colab Free.

Do not present two versions as quantitatively comparable unless their feature schema, output mapping, input policy, and frozen human-holdout fingerprint are equal. Show incompatible candidates as auditable metadata only.

## Re-evaluate only immutable evidence

Reuse recorded manifest metrics whenever they answer the comparison. Recalculate F1, class support, confusion matrices, or calibration only when both validated candidates use the same compatible frozen human holdout. Do not evaluate against a mutable dataset, proxy-only labels, synthetic validation/test records, or a candidate's training data.

Keep the stable MLP outputs `Normal`, `Reduced`, and `Congested`; Accident remains human-confirmed only. Preserve seed/HITL lifecycle semantics and show pilot or candidate status instead of inferring promotion from a single metric.

Treat comparison latency below five seconds as a real-Colab benchmark. Measure manifest loading, local materialization, and any optional evaluation separately; do not promise a network-dependent latency target.

## Keep the comparison UI private

Use the declared `ipywidgets` extra for a future Colab-private, read-only interface. Provide two candidate selectors and panels for lifecycle/eligibility, provenance, metric and class-support tables, confusion matrices, reliability evidence, and compatibility warnings. Render from validated manifests first and load model weights only for an authorized compatible re-evaluation.

Do not open ports or create public URLs. Do not add MLflow, Streamlit, Gradio, ngrok, LocalTunnel, a remote experiment server, or a web service as part of this workflow. Treat any external UI or tracking service as a future change requiring explicit authorization, security review, dependency approval, and an ADR.

## Audit and verify safely

Keep DVC credentials in its local configuration or approved secrets; never print OAuth material, Drive identifiers, URLs containing tokens, private paths, model-review notes, or raw data. Preserve the manifest's checksums and reject corrupt, incomplete, or mixed bundles.

For changes in scope, run Ruff, pytest, compilation, notebook-cell parsing, Markdown link checks, and `git diff --check`. Validate DVC authentication, Drive access, local materialization, and any private Colab interaction manually.

---

Reject these antipatterns:

- Do not use names such as `model_final_v3` as registry identity or overwrite a Drive bundle in place.
- Do not mix model, scaler, mapping, or manifest files across revisions.
- Do not alter DVC remotes, pointers, snapshots, catalogs, holdouts, locks, or production eligibility from a comparison interface.
- Do not compare metrics across mutable or fingerprint-incompatible holdouts, or use a high F1 score as automatic promotion.
- Do not expose a Colab port, publish a tunnel URL, hardcode credentials, or add external tracking/UI dependencies without approval.

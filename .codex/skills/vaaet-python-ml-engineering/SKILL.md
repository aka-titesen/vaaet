---
name: vaaet-python-ml-engineering
description: Build, review, refactor, or test professional Python for VAAET Data Science, computer vision, MLOps, and notebooks. Use for typed modular code, domain exceptions, logging, pytest, quality gates, notebook-to-module extraction, or maintainable model/data pipelines.
---

# VAAET Python ML Engineering

## Apply this operating model

Build production-quality Python 3.10–3.12 for Data Science, computer vision, MLOps, and notebooks. Optimize for correctness, maintainability, observability, testability, and reproducibility—not abstraction for its own sake.

- Keep notebooks as orchestration and visualization interfaces. Move reusable business logic into `src/vaaet/`; do not mutate `sys.path`.
- Preserve VAAET layers: settings/contracts/artifacts, data, features, vision, inference, evaluation, and training.
- Apply KISS and YAGNI first. Use DRY only when duplication represents one stable concept; do not create a generic framework for one workflow.
- Read applicable ADRs before changing a contract. Do not modify the 19 `FEATURE_COLS`, MLP, thresholds, PostgreSQL schema, public states, or bundle v2 without authorization and an ADR.

## Type and model data explicitly

Annotate public functions, class attributes, return values, and non-obvious local values. Use built-in generic syntax (`list[str]`, `dict[str, float]`), `Path`, `datetime`, `Literal`, `TypedDict`, `Protocol`, `Enum`, and frozen `@dataclass` where they accurately express the domain.

Use concrete NumPy/Pandas or framework types at library boundaries. Validate shape, columns, state codes, checksums, and units at runtime; static typing does not validate data received from video, CSV, PostgreSQL, Drive, or a model.

Avoid `Any`. Allow it only at a third-party boundary that cannot be typed, isolate it in the smallest adapter, validate immediately, and avoid propagating it through the domain. Prefer `Protocol` for a swappable dependency; use an ABC only when shared behavior or lifecycle enforcement is genuinely required.

Model domain results as immutable dataclasses when they have stable fields. Keep transport/persistence dictionaries at the boundary and convert them into validated domain objects promptly.

## Keep responsibilities small and observable

- Let data modules acquire, validate, and persist; let vision modules process frames; let features modules engineer the canonical features; let inference modules classify; let training/evaluation modules control lifecycle and quality.
- Inject external systems—database engines, model providers, clocks, filesystem roots, and download clients—rather than hardcoding connections or global mutable state.
- Keep one explicit responsibility per function. Split functions whose nested branching or error paths obscure correctness; use a complexity target below 10 as a review heuristic, not a substitute for judgment.
- Do not make a notebook cell, a `print`, a mutable DataFrame attribute, or a filesystem convention the sole contract between modules.

Use `vaaet.logging` in reusable code. Log structured, actionable context at the appropriate level: lifecycle events at `INFO`, recoverable degradation at `WARNING`, failed operations at `ERROR`, and bounded diagnostics at `DEBUG`. Never log secrets, DSNs, certificates, private paths, raw credentials, or unredacted exception payloads.

## Fail with domain meaning

Raise the narrowest existing exception from `vaaet.exceptions` or add a documented subtype rooted in the project hierarchy when callers need a different recovery action. Include safe identifiers such as stage, clip ID, record time, contract version, or field name.

Do not catch `Exception` merely to continue. Catch only expected external or validation failures, add useful context with exception chaining, and stop a corrupted pipeline before it produces an artifact or persistence side effect. Use redacted `pipeline_run` metadata for operational lineage.

## Test the contract, not heavy infrastructure

Write or update tests alongside behavior changes:

- Unit-test pure transformations, validation branches, exception paths, and state transitions with small deterministic arrays, frames, and DataFrames.
- Use fixtures, fakes, dependency injection, or mocks for PostgreSQL, Drive, downloads, clocks, and model calls; never require YOLO weights, a GPU, real credentials, or a live database for unit tests.
- Add contract/integration tests for schemas, artifact manifests, persistence idempotency, package checksums, and notebook parity when their boundaries change.
- Assert observable behavior and invariants, not internal implementation details. Keep random seeds explicit.

Treat at least 90% coverage as the quality target for new or materially changed core modules. Do not claim that the current repository meets it without a coverage report. Adding `mypy`, coverage plugins, or CI gates changes dependencies/tooling and requires explicit authorization; introduce a baseline and ratchet coverage rather than failing unrelated historical code immediately.

## Quality gates and Definition of Done

For every implementation, run the repository-required checks:

1. `ruff check src tests scripts`
2. `pytest tests/ -v --tb=short`
3. `python -m compileall -q src tests scripts`
4. Parse code cells from all three notebooks.
5. Check Markdown links and run `git diff --check`.

When static typing is approved and configured, run the scoped MyPy command in CI and require zero errors for its configured scope. Do not run `mypy .` as a blanket repository gate until configuration, third-party stubs, and an incremental baseline have been explicitly adopted.

For a new notebook, keep executable orchestration minimal—preferably under 50 lines per operation cell—and call tested modules. For existing notebooks, extract logic incrementally; do not perform a large refactor without authorization.

## Reject these antipatterns

- Do not build monolithic notebook functions that duplicate `src/vaaet/` logic.
- Do not silently swallow failures with `except Exception: pass` or log-and-continue after data corruption.
- Do not use pervasive `Any`, unvalidated `dict[str, object]`, or untyped model/data boundaries.
- Do not hardcode credentials, connections, model file paths, or environment-specific configuration.
- Do not add a class hierarchy, ABC, dependency, or distributed abstraction without a present and measured need.
- Do not store `.pt`, `.keras`, or `.joblib` binaries in Git.

## Review output

When reviewing or changing code, report: the responsibility boundary, typed input/output contract, validation and exception behavior, logging/lineage impact, tests added or needed, and remaining risk. Prefer a small focused patch over an architectural rewrite.

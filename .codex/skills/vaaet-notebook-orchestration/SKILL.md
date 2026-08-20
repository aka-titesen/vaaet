---
name: vaaet-notebook-orchestration
description: Design, review, simplify, or audit VAAET Jupyter and Google Colab notebooks as thin, sequential, idempotent orchestrators. Use for notebook cell organization, Run All reliability, centralized fail-fast configuration, optional ipywidgets adapters, output hygiene, notebook-to-module boundaries, or detection of notebook antipatterns.
---

# VAAET Notebook Orchestration

Implement the intent of specification `SKL-COLAB-ARCH-009` while preserving VAAET's
package, runtime, and governance contracts.

## Keep the responsibility boundary explicit

- Keep notebooks as user-facing orchestration and visualization entrypoints.
- Put reusable data, vision, feature, inference, training, and persistence logic in
  `src/vaaet/`; import it through `vaaet.*`.
- Use `$vaaet-python-ml-engineering` when extracting or redesigning reusable Python.
- Use `$vaaet-colab-operations` for runtime setup, GPU/RAM, Drive, Secrets, recovery,
  immutable artifacts, and Colab-specific operational behavior.
- Read `AGENTS.md` and the applicable ADRs before changing contracts or workflow behavior.

Do not mutate `sys.path`, import `src.*`, use `requirements.txt`, or add notebook-local
dependency drift. Install project extras from `pyproject.toml`: use a normal installation in
Colab and an editable installation only for local development.

## Organize a linear workflow

Keep one responsibility per cell and preserve this conceptual order:

1. Explain the workflow, inputs, outputs, and safety defaults.
2. Run the single idempotent environment setup cell.
3. Validate one centralized workflow configuration before expensive work.
4. Acquire or select inputs explicitly.
5. Call shared package APIs for the main operation.
6. Perform opt-in persistence or human review.
7. Present and export bounded results.

Place configuration before setup only when it uses plain Python values. Place it immediately
after setup when it requires enums or types imported from `vaaet`. Never require out-of-order
execution. Ensure `Run All` works with safe defaults and without repairing hidden state.

Prefer operation cells below 50 lines. Treat a cell above 50 lines as an extraction review and
a cell above 500 lines as a structural failure. Do not split cohesive setup mechanically merely
to satisfy the soft target; extract reusable behavior instead.

## Make configuration deterministic

- Define each operational option exactly once in a clearly marked configuration cell.
- Default persistence, remote writes, experimental models, and expensive optional behavior to
  disabled unless the governing workflow says otherwise.
- Validate enum values, paths, mutually exclusive options, required profiles, and artifact
  compatibility before processing begins.
- Avoid mutable catch-all dictionaries and variables that change type across cells.
- Move a growing or shared configuration contract to a typed immutable object in `src/vaaet/`.

Treat `ipywidgets` as an optional frontend. Lazy-import it only when UI is authorized. Make the
widgets produce the same validated typed configuration used by the non-interactive path. Keep a
safe default configuration so `Run All` never depends on clicking a button. Do not let widget
callbacks become the only source of truth or business logic.

## Preserve idempotency and observable failures

- Make setup safe to rerun: clone or fast-forward, install once, clear stale imports, and validate
  the installed package origin.
- Make downloads, persistence, review finalization, and artifact publication explicitly
  idempotent through their existing VAAET APIs.
- Fail fast with a clear recovery action when enabled behavior lacks inputs, credentials, schema,
  or compatible artifacts.
- Keep useful progress and final summaries visible, but capture or suppress noisy package output.
- Never print secrets, DSNs, certificates, private review notes, or unredacted exceptions.
- Do not catch broad exceptions merely to continue after a corrupted result.

## Audit before and after edits

Run the bundled auditor against every active notebook:

```bash
python .codex/skills/vaaet-notebook-orchestration/scripts/audit_notebooks.py notebooks
```

The auditor returns nonzero for invalid JSON/Python, cells above 500 lines, forbidden import or
installation patterns, missing or duplicated setup/configuration cells, and configuration values
reassigned outside their owning cell. It reports cells above 50 lines as non-blocking warnings.

After changes, also run the repository-required Ruff, pytest, compileall, notebook AST, Markdown
link, and `git diff --check` gates. Report which logic stayed in the notebook, which logic moved to
`src/vaaet/`, configuration and failure behavior, tests, and any Colab-only validation pending.

## Reject common notebook antipatterns

- Monolithic cells containing reusable OpenCV, model, database, or feature logic.
- Multiple setup paths, repeated installation commands, or hidden import-path mutations.
- Configuration assignments scattered across later cells or Markdown examples copied as code.
- Mandatory widget clicks, implicit persistence, or credentials that activate writes by presence.
- `Run All` flows that require returning to earlier cells or preserving stale runtime variables.
- Unbounded logs, displays inside frame loops, or outputs large enough to destabilize the browser.

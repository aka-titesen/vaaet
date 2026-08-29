---
name: vaaet-python-project-configuration
description: Configure, review, or harden VAAET Python project setup. Use for virtual environments, pyproject dependencies and extras, development quality gates, secrets, security-tool proposals, pre-commit decisions, or compatibility guidance for legacy requirements-based Python projects.
---

# VAAET Python Project Configuration

## Preserve VAAET's source of truth

Use each component's `pyproject.toml` as its sole dependency, build, test, and Ruff configuration source: `vaaet-core/pyproject.toml` for the portable package and `vaaet-ml/pyproject.toml` for the laboratory. The repository root is not an installable package. VAAET supports Python 3.10–3.13; do not widen that range without explicit authorization.

Keep dependency groups intentionally separate by component:

- `vaaet-core`: core runtime plus `vision`, `inference`, and `dev` extras.
- `vaaet-ml`: laboratory runtime plus `training`, `database`, `visualization`, `dvc`, and `dev` extras.

Install only the extras needed for the selected workflow. Do not create `requirements/`, `setup.cfg`, `requirements.txt`, lockfiles, or a second dependency resolver for VAAET unless the user explicitly authorizes a dependency-management change.

## Work in an isolated environment

For local development, create an isolated `.venv` with a supported Python version, activate it for the host shell, then install only the required extras. For a full ML workflow, install `vaaet-core` first and `vaaet-ml` second. Never install project dependencies globally.

Use normal component installation from the repository: editable installs for local development and built wheels or declared component installs for Colab. Keep notebook setup limited to the existing package-installation cell; do not install individual packages in notebook processing cells or loops.

Use `pip check` to diagnose a managed runtime after installation. Do not replace declared dependency ranges with `pip freeze` output. A fully pinned lock is an optional reproducibility mechanism that needs an approved tool and update policy.

## Use one quality toolchain

Use Ruff as VAAET's formatter/linter authority and keep its configuration in `pyproject.toml`. Run the repository gates:

1. Run the `vaaet-core/AGENTS.md` commands when core changes.
2. Run the `vaaet-ml/AGENTS.md` commands when ML changes, after installing core first when required.
3. Run root `pyright --project pyrightconfig.json` for configured typing scope.
4. Parse code cells from all four notebooks in `vaaet-ml/notebooks/`, check Markdown links, and run `git diff --check`.

Do not add Black, isort, Flake8, or `setup.cfg` to VAAET. They duplicate Ruff's role and create conflicting formatting/lint rules.

Treat MyPy, coverage thresholds, pre-commit, Bandit, pip-audit, dependency locks, and expanded CI enforcement as proposals, not implicit edits. Pyright is the configured static-type tool. Before adding any proposal, explain the benefit, package/CI impact, configuration scope, false-positive policy, and maintenance owner; then obtain authorization because it changes dependencies or workflows.

## Keep configuration secure and fail fast

Use VAAET's existing settings and database loaders. Read credentials from Colab Secrets or the local environment, validate required values at the boundary, and fail with a safe domain error when an explicitly enabled integration lacks configuration.

Never commit `.env` files, secrets, DSNs, private endpoints, certificates, model binaries, or environment-specific paths. Never log credential values or store them in pipeline metadata. Do not demand optional database credentials for an offline or local-only workflow.

## Legacy-project compatibility

Use the classical `requirements/base.txt`, `requirements/dev.txt`, and `requirements/prod.txt` layout only when working in a legacy project that cannot use PEP 621/`pyproject.toml`, or whose deployment tooling explicitly requires pip requirements files.

In that context, keep runtime, development, and production packages separate; use a single compatible formatter/linter stack; audit dependencies and source security in CI; and configure pre-commit only after the team agrees on hook latency and versions. Treat `pip freeze` as a generated environment snapshot, not the maintained declaration of direct dependencies.

Do not copy this legacy layout into VAAET merely for familiarity.

## Review and change workflow

Before changing project setup:

1. Inspect `pyproject.toml`, existing automation, and the affected workflow.
2. Identify the exact runtime/development/security problem and the smallest compatible remedy.
3. Ask before changing dependencies, supported Python versions, CI, DVC remotes, or development hooks.
4. Keep configuration changes reproducible, documented in the project source of truth, and covered by the existing checks.
5. Report installations, configuration files changed, security implications, and commands required from developers separately.

Reject duplicate tooling, global installs, unbounded dependency upgrades, secrets in source, notebook-local dependency drift, and silent fallbacks caused by missing required configuration.

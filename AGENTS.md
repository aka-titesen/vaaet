# AGENTS.md — Monorepo VAAET

## Límites de componentes

- `vaaet-ml/` contiene todo el comportamiento Python, notebooks, migraciones,
  tests, datos locales y artefactos ML. Sus reglas detalladas están en
  [`vaaet-ml/AGENTS.md`](vaaet-ml/AGENTS.md).
- `vaaet-app/` está reservado: no agregar API, frontend, framework ni
  dependencias sin un alcance aprobado y un contrato HTTP versionado.
- `docs/`, `.dvc/` y `.github/` pertenecen a la raíz compartida.

## Invariantes

Conservar un único Git y DVC. La Web App futura sólo consumirá una API; no puede
leer PostgreSQL, DVC, Drive, artefactos binarios ni módulos de `vaaet-ml`.
La API deberá validar el manifiesto del bundle v2 antes de deserializarlo.

Leer [ADR-0020](docs/architecture/decisions/0020-single-git-monorepo-and-application-boundary.md)
antes de alterar la estructura y la documentación de `vaaet-ml/AGENTS.md`
antes de cambiar el componente ML.

# AGENTS.md — Monorepo VAAET

## Límites de componentes

- `vaaet-core/` contiene la lógica Python portable de percepción, telemetría,
  clasificación y bundle con import `vaaet`. No puede depender de
  `vaaet_ml`, PostgreSQL, DVC, Drive ni APIs de notebook.
- `vaaet-ml/` contiene laboratorio, entrenamiento, evaluación, notebooks,
  migraciones, datos locales y artefactos ML con import `vaaet_ml`. Sus reglas
  detalladas están en [`vaaet-ml/AGENTS.md`](vaaet-ml/AGENTS.md).
- `vaaet-app/` está reservado: no agregar API, frontend, framework ni
  dependencias sin un alcance aprobado y un contrato HTTP versionado.
- `docs/`, `.dvc/` y `.github/` pertenecen a la raíz compartida.

## Invariantes

Conservar un único Git y DVC. La Web App futura sólo consumirá una API; no puede
leer PostgreSQL, DVC, Drive, artefactos binarios ni módulos Python. La API
deberá validar el manifiesto v2 antes de deserializarlo y sus workers usarán
`vaaet-core`, nunca `vaaet-ml` para serving.

Leer [ADR-0021](docs/architecture/decisions/0021-portable-core-and-ml-laboratory-boundary.md)
antes de alterar la estructura y la documentación de `vaaet-ml/AGENTS.md`
antes de cambiar el componente ML.

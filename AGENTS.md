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
- Una API o Web App que ejecute `vaaet-core[vision]` sólo puede desplegarse por
  una vía explícita: demo pública AGPL-3.0 con código y activos aprobados, o
  aplicación privada/comercial con licencia Ultralytics Enterprise verificada
  fuera de Git. La vía AGPL exige completar el
  [`checklist de demo`](docs/governance/agpl-demo-release-checklist.md). El
  registro público no contiene contratos ni evidencia privada.
- `docs/`, `.dvc/` y `.github/` pertenecen a la raíz compartida. `.dvc/config`
  es neutral y versionado; cada entorno configura exclusivamente en
  `.dvc/config.local` el remoto lógico `vaaet-registry`.

## Invariantes

Conservar un único Git y DVC. Git identifica las versiones de bundle y DVC
almacena su contenido; la guía y ADR-0023 gobiernan el registro portable. La Web
App futura sólo consumirá una API; no puede
leer PostgreSQL, DVC, Drive, artefactos binarios ni módulos Python. La API
deberá validar el manifiesto v2 antes de deserializarlo y sus workers usarán
`vaaet-core`, nunca `vaaet-ml` para serving.

Antes de editar, leer el resumen portable en [`llms.txt`](llms.txt), el índice
de [`docs/`](docs/index.md) y
[ADR-0021](docs/architecture/decisions/0021-portable-core-and-ml-laboratory-boundary.md).
Para cambios de componente, leer también
[`vaaet-core/AGENTS.md`](vaaet-core/AGENTS.md) o
[`vaaet-ml/AGENTS.md`](vaaet-ml/AGENTS.md); ADR-0022 gobierna cualquier
serving futuro con YOLO.

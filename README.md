# VAAET

Monorepo de VAAET para el análisis vehicular y su futura aplicación web.

## Componentes

- [`vaaet-ml/`](vaaet-ml/README.md): paquete Python, notebooks de Google Colab,
  entrenamiento, inferencia, migraciones y artefactos ML.
- [`vaaet-core/`](vaaet-core/README.md): lógica portable de percepción,
  telemetría, clasificación y validación de bundles; expone el import `vaaet`.
- [`vaaet-app/`](vaaet-app/README.md): frontera reservada para una API y Web App
  futuras; no contiene código ejecutable en esta etapa.
- [`docs/`](docs/index.md): documentación, ADRs y contratos compartidos.

La arquitectura está gobernada por
[ADR-0021](docs/architecture/decisions/0021-portable-core-and-ml-laboratory-boundary.md).
La raíz conserva el único repositorio Git, la configuración DVC y la
automatización CI. Para trabajar sobre ML, instalá primero `vaaet-core/` y luego
`vaaet-ml/` desde sus `pyproject.toml` locales.

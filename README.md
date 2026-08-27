# VAAET

Monorepo de VAAET para el análisis vehicular y su futura aplicación web.

## Componentes

- [`vaaet-ml/`](vaaet-ml/README.md): paquete Python, notebooks de Google Colab,
  entrenamiento, inferencia, migraciones y artefactos ML.
- [`vaaet-app/`](vaaet-app/README.md): frontera reservada para una API y Web App
  futuras; no contiene código ejecutable en esta etapa.
- [`docs/`](docs/index.md): documentación, ADRs y contratos compartidos.

La arquitectura está gobernada por
[ADR-0020](docs/architecture/decisions/0020-single-git-monorepo-and-application-boundary.md).
La raíz conserva el único repositorio Git, la configuración DVC y la
automatización CI. Para trabajar sobre ML, ingresá a `vaaet-ml/` e instalá sus
extras desde el `pyproject.toml` del componente.

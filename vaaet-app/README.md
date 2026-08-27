# VAAET App

Este directorio reserva la futura aplicación VAAET. No contiene API, frontend,
framework, dependencias ni acceso directo a datos en esta etapa.

La futura API instalará `../vaaet-ml`, validará el bundle v2 mediante
`vaaet.artifacts.validate_manifest()` antes de deserializarlo y expondrá un
contrato HTTP versionado. La Web App consumirá exclusivamente esa API; no debe
acceder a PostgreSQL, DVC, Google Drive, artefactos binarios ni módulos Python.

La arquitectura está gobernada por
[ADR-0020](../docs/architecture/decisions/0020-single-git-monorepo-and-application-boundary.md).

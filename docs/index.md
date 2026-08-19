# Documentación de VAAET ML 4.5.1

## Puntos de entrada

- [Arquitectura de software](architecture/software-architecture.md)
- [Contrato del bundle](ml/model-artifact-contract.md)
- [ADR de inicio semilla y HITL](architecture/decisions/0017-seed-bootstrap-and-hitl-retraining.md)
- [ADR de holdout humano congelado](architecture/decisions/0018-versioned-frozen-human-holdouts.md)
- [ADR de datasets semilla/HITL inmutables](architecture/decisions/0019-immutable-seed-and-hitl-datasets.md)
- [Model card y gates de promoción](ml/model-card.md)
- [Protocolo de anotación humana](ml/human-annotation-protocol.md)
- [Guía de usuario](operations/user-guide.md)
- [Guía de Google Colab](operations/colab-guide.md)
- [Operación PostgreSQL](operations/postgresql-guide.md)
- [Requisitos de producto](product/product-requirements.md)
- [Plan de pruebas](quality/test-plan.md)
- [Modelo PostgreSQL](architecture/data-model.md)
- [ADR-0015: PostgreSQL seguro e HITL](architecture/decisions/0015-postgresql-namespaces-security-and-hitl.md)
- [ADR-0016: hardening PostgreSQL y ejecuciones](architecture/decisions/0016-postgresql-hardening-and-pipeline-runs.md)

## Mapa documental

- `architecture/`: diseño, datos, linaje, diagramas y decisiones.
- `ml/`: model card, contrato, DVC, sesgos y limitaciones.
- `product/`: requisitos, usuarios, casos de uso y factibilidad.
- `operations/`: despliegue y operación.
- `quality/`: pruebas, KPIs y riesgos.
- `governance/`: planificación, seguridad y registros legales.

Los agentes de IA deben leer `AGENTS.md` y `llms.txt` antes de editar. Los tres
workflows activos son adquisición, entrenamiento e inferencia.

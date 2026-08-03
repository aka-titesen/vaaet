# Documentación de VAAET ML 4.0.0

## Puntos de entrada

- [Arquitectura de software](architecture/software-architecture.md)
- [Contrato del bundle](ml/model-artifact-contract.md)
- [Model card y gates de promoción](ml/model-card.md)
- [Protocolo de anotación humana](ml/human-annotation-protocol.md)
- [Guía de usuario](operations/user-guide.md)
- [Guía de Google Colab](operations/colab-guide.md)
- [Requisitos de producto](product/product-requirements.md)
- [Plan de pruebas](quality/test-plan.md)

## Mapa documental

- `architecture/`: diseño, datos, linaje, diagramas y decisiones.
- `ml/`: model card, contrato, DVC, sesgos y limitaciones.
- `product/`: requisitos, usuarios, casos de uso y factibilidad.
- `operations/`: despliegue y operación.
- `quality/`: pruebas, KPIs y riesgos.
- `governance/`: planificación, seguridad y registros legales.

Los agentes de IA deben leer `AGENTS.md` y `llms.txt` antes de editar. Los tres
workflows activos son adquisición, entrenamiento e inferencia.

# ADR-0012: Límite ML/Web y contrato portable de artefactos

**Estado:** Aceptado  
**Fecha:** 2026-08-01

## Contexto

VAAET evolucionará hacia una Web App con inferencia en tiempo real. El código de
ML y el backend necesitan ciclos de entrega independientes y un límite explícito.

## Decisión

- La URL canónica continúa siendo `https://github.com/zgfnicolas/vaaet`.
- La distribución Python es `vaaet-ml` 4.0.0 y el paquete importable es `vaaet`,
  ubicado en `src/vaaet/`.
- Se adopta multi-repo: este repositorio posee ML y metadata DVC; el futuro repo
  web posee serving.
- Ambos intercambian el bundle de cuatro archivos definido en el
  [contrato](../../ml/model-artifact-contract.md).
- Todo consumidor valida el manifiesto antes de deserializar los binarios.
- No se incorpora API, CLI, servicio, migración de BD, cambio de features/MLP ni
  trigger automático de CT.
- La adquisición de datos bajo demanda se rige por ADR-0013.
- La adquisición de datos bajo demanda se rige por ADR-0013.

## Consecuencias

El contrato evita drift entre training y serving para las 19 features y cuatro
clases canónicas. La publicación del bundle queda fuera de alcance.

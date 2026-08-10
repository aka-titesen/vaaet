# ADR-0017 — Inicio semilla y reentrenamiento HITL

- Estado: aceptado
- Fecha: 2026-08-09
- Versión: VAAET ML 4.3.0
- Complementa: [ADR-0014](0014-hierarchical-traffic-state-and-incident-policy.md) y [ADR-0015](0015-postgresql-namespaces-security-and-hitl.md)

## Contexto

El archivo histórico `traffic_data.backup` contiene telemetría cruda legacy y
no dispone de ground truth humano ni métricas de calidad de telemetría v2. Es
útil para iniciar el ciclo, pero mezclar su preparación con los reentrenamientos
HITL hacía menos clara la procedencia de etiquetas, alentaba recalcular features
ya persistidas y podía presentar métricas proxy como calidad operacional.

Se necesita entregar pronto un modelo piloto que reproduzca de forma
conservadora la matriz condicional, sin confundirlo con un modelo aprobado para
producción, y sustituir gradualmente esa weak supervision por evidencia humana.

## Decisión

- `TrainingMode.SEED_BOOTSTRAP` procesa raw una sola vez, normaliza timestamps,
  calcula las 19 features, asigna etiquetas proxy y exporta un paquete semilla.
- `TrainingMode.HITL_RETRAINING` consume features compatibles y validaciones
  humanas efectivas. No recalcula las features ni acepta autopredicciones como
  targets.
- Ambos modos convergen antes del split y comparten scaler, MLP, calibración,
  evaluación y contrato del bundle.
- La memoria semilla recibe por clase
  `0.5 * max(0, 1 - soporte_humano / soporte_objetivo)` y desaparece al alcanzar
  300 etiquetas Normal, 300 Reduced o 100 Congested.
- La entrada legacy neutraliza exclusivamente las features de calidad ausentes.
  El mismo `input_policy` se declara en el manifiesto y se ejecuta en inferencia.
- Se comparan class weights, oversampling moderado y escenarios sintéticos de
  congestión sólo con train. Validation elige por coste de confusión y falsos
  Congested; test permanece congelado.
- El modelo semilla tiene `deployment_stage=pilot`, supervisión `weak-proxy` y
  `production_eligible=false`. Inferencia exige autorizar explícitamente su uso.
- Accident permanece fuera del MLP. Una regla sólo crea un candidato y conserva
  Congested; únicamente una validación humana puede publicar Accident.

## Consecuencias

- El backup histórico deja de restaurarse en cada reentrenamiento: se reutiliza
  el paquete procesado con procedencia y checksums.
- Las métricas proxy miden fidelidad a las reglas, no eficacia real. La promoción
  continúa bloqueada hasta superar los gates sobre un holdout humano congelado.
- Los sintéticos nunca ingresan en validation/test y su peso efectivo está
  limitado. No se utiliza SMOTE para Accident ni balance 1:1.
- Los bundles piloto son válidos e íntegros, pero no intercambiables
  silenciosamente con artefactos de producción.

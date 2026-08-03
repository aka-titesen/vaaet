# Model card — VAAET Traffic State Classifier

## Identificación

| Campo | Valor |
|---|---|
| Proyecto | VAAET ML 4.0.0 |
| Modelo vigente | `mlp-v2.0` |
| Estado inicial | Experimental / shadow-only hasta cumplir gates |
| Runtime | TensorFlow/Keras, Python 3.10–3.12, Google Colab |
| Artefacto | Bundle DVC contrato v2 |

## Uso previsto

El modelo ayuda a describir por minuto el flujo del Puente General Manuel Belgrano. No es un sistema autónomo de seguridad, no decide respuestas de emergencia y no debe transferirse a otra cámara o vía sin validación local.

## Arquitectura

```text
19 features v2
-> Dense(64) + BatchNorm + Dropout(0.3)
-> Dense(32) + BatchNorm + Dropout(0.2)
-> Dense(3, softmax)
-> confianza/margen + histéresis + transiciones adyacentes
-> detector conservador de posible incidente
-> confirmación humana opcional
```

Las salidas aprendidas son `Normal`, `Reduced` y `Congested`. El estado público `Accident` sólo puede provenir de una confirmación humana validada. Una sospecha automática permanece `Congested` y activa `accident_rule_triggered`.

## Features

Se conserva el orden contractual de 19 features: velocidad y volumen; conteos de cinco tipos; proporción de pesados; deltas, transición, varianza y persistencia por segmento continuo de cada clip; calidad de velocidad, near-zero y stationary sobre tracks únicos; hora y proxy nocturno. La semántica temporal corresponde a `traffic-features-v2`.

Los registros `traffic-telemetry-v1` no contienen evidencia moderna de calidad. Esos valores permanecen desconocidos y sólo permiten reproducir un baseline experimental; nunca se interpretan como calidad perfecta.

## Datos y particiones

- Test: clips reales temporalmente posteriores y congelados.
- Validation: grupos reales completos del período restante.
- Train: grupos restantes; es la única partición que admite sintéticos.
- Scaler, class weights y cualquier balanceo se ajustan sólo con train.
- El baseline usa class weights limitados y no SMOTE 1:1.
- Escenarios sintéticos de Accident no son targets del MLP ni evidencia de eficacia real.

Las etiquetas de reglas son proxies. Las métricas de producción sólo son válidas sobre un holdout real, humano y agrupado conforme al [protocolo de anotación](human-annotation-protocol.md).

## Métricas y promoción

Los objetivos iniciales son F1-macro ≥0,88; precision/recall de Normal ≥0,93; precision Reduced ≥0,88 y recall ≥0,90; precision Congested ≥0,90 y recall ≥0,85; error directo Normal↔Congested ≤1%; ECE ≤0,05. Cada resultado debe incluir soporte, clips, origen, intervalos y matrices absoluta/normalizada.

Sin al menos 100 minutos Congested validados de 20 episodios reales, esa clase es experimental. Sin accidentes reales no se publica recall de Accident. Para candidatos se reportan falsos por hora; el objetivo preliminar es menos de uno cada 100 horas y unas 300 horas negativas sin falsos para evidencia aproximada al 95%.

La promoción manual exige telemetría v2 suficiente, holdout humano, retrospective replay, shadow mode prospectivo y revisión de falsos positivos. El manifiesto conserva `production_eligible` y `promotion_blockers`.

## Limitaciones

- Datos históricos de un único puente, período y configuración de cámara.
- Calidad de velocidad sin ground truth físico exhaustivo.
- `weather_condition` sigue siendo un proxy horario, no meteorología real.
- El MLP es tabular; la temporalidad se incorpora mediante features y política, no LSTM.
- El desempeño con incidentes reales es desconocido hasta adquirir ejemplos confirmados.

## Historial

| Versión | Cambio |
|---|---|
| `mlp-v1.0` | Baseline histórico de 14 features |
| `mlp-v1.1` | Baseline de 19 features y cuatro salidas |
| `mlp-v2.0` | Tres salidas estables, contrato v2 y política jerárquica humana para Accident |

Última revisión: 2026-08-02. Véase [ADR-0014](../architecture/decisions/0014-hierarchical-traffic-state-and-incident-policy.md).

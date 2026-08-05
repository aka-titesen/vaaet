# ADR-0014: Clasificación jerárquica del flujo e incidentes

- Estado: aceptada
- Fecha: 2026-08-02
- Decisores: Facundo Nicolás González

## Contexto

El baseline `mlp-v1.1` aprendía cuatro clases a partir de etiquetas automáticas y datos sintéticos balanceados con SMOTE. El conjunto histórico no contiene accidentes reales y la validación posterior al remuestreo podía mezclar episodios o medir filas sintéticas. Esa evidencia no permite sostener precisión operacional para Accident ni probabilidades calibradas.

## Decisión

`mlp-v2.0` aprende únicamente `Normal`, `Reduced` y `Congested`. Los cuatro
estados públicos permanecen vigentes, pero `Accident` sólo puede resultar de una
validación humana efectiva con estado 3. Desde VAAET 4.1.0 su persistencia
append-only se define en [ADR-0015](0015-postgresql-namespaces-security-and-hitl.md).

La cadena compartida por entrenamiento e inferencia es:

```text
19 features v2 -> scaler -> MLP de tres clases
-> umbral y margen -> transiciones adyacentes e histéresis
-> detector conservador de posible incidente
-> confirmación humana opcional
```

Una evidencia fuerte y persistente de incidente mantiene el estado operativo `Congested` y activa `accident_rule_triggered`. Una medición incompleta o de baja calidad nunca activa la alerta. Ningún camino automático puede publicar el código 3.

Las particiones se forman antes de escalar o balancear. Test usa grupos temporalmente posteriores, validation conserva clips completos y los sintéticos sólo pueden entrar en train con peso reducido. `validation_split` y SMOTE 1:1 dejan de formar parte del baseline.

El contrato de bundle v2 registra tres salidas aprendidas, cuatro estados públicos, umbrales, persistencia, prohibición de Accident automático, elegibilidad y bloqueos de promoción. Un bundle sin telemetría v2 suficiente o sin holdout humano queda en `experimental/shadow-only`.

## Consecuencias

- Se eliminan por diseño los falsos estados Accident automáticos.
- Congested requiere soporte real y validado; hasta entonces sus métricas son preliminares.
- Los registros v1 continúan disponibles para baseline, pero sus campos de calidad son desconocidos y no se imputan como calidad perfecta.
- Se necesita anotación humana y exposición negativa suficiente antes de promoción.
- La inferencia sólo clasifica minutos completos y conserva el último estado estable durante la ventana en curso.

## Criterios de promoción

La promoción es manual. Requiere holdout real y humano, telemetría v2 con cobertura suficiente, ausencia de leakage, métricas con soporte e intervalos, revisión retrospectiva y shadow mode prospectivo. Sin accidentes reales no se publica recall de Accident; se reportan candidatos falsos por hora y la exposición acumulada.

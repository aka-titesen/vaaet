<!-- context: VAAET/docs/architecture/diagrams/speed-calculation.md — Flujo vigente de velocidad physics-first. -->

# Flujo de cálculo de velocidad physics-first

`vaaet-core/src/vaaet/vision/pipeline.py` coordina el análisis ordenado; la
estimación vive en `vaaet-core/src/vaaet/vision/speed.py`. El resultado depende
del historial de cada track y no ofrece un fallback de velocidad inventada.

```mermaid
flowchart TD
    A[Historial reciente de centroides] --> B[Flujo óptico global]
    B --> C[Compensar movimiento de cámara]
    C --> D[Desplazamientos por frame]
    D --> E[Filtrar ruido y anomalías]
    E --> F[Corrección de perspectiva por zona Y]
    F --> G[Convertir píxeles a metros]
    G --> H[Velocidad física km/h]
    H --> I{¿Plausible y confiable?}
    I -->|No| J[Excluir de la síntesis]
    I -->|Sí| K[Suavizado por track]
    A --> L[Analizar movimiento]
    L --> M[Confirmación estacionaria con histéresis]
    K --> N[Acumulador por minuto]
    M --> N
    J --> N
    N --> O[Síntesis robusta y señales de calidad]
    O --> P[Fila de telemetría apta para clasificación]
```

La fusión MLP de velocidad es un path dormido y no integra el runtime activo.

<!-- context: VAAET/Docs/diagrams/intelligence-pipeline.md — Pipeline de la Etapa 2 (Inteligencia).
Referenciado por DDS.md §8, ADR-008, notebook 02_traffic_state_classifier.ipynb. -->

# Pipeline de Inteligencia — Etapa 2

Flujo completo desde la telemetría de Etapa 1 hasta la clasificación de estado de tráfico y persistencia.

```mermaid
flowchart TD
    A[(traffic_data<br/>PostgreSQL)] -->|SQL query| B[DataFrame<br/>~2000 registros × 9 campos]

    B --> C[Ingeniería de Features]
    C --> C1[heavy_vehicle_ratio<br/>delta_speed / delta_count<br/>transition_flag<br/>speed_variance<br/>hour_of_day / weather_condition]
    C1 --> D[DataFrame<br/>14 features]

    D --> E[Auto-Labeling]
    E --> E1{Reglas de ingeniería}
    E1 -->|avg_speed > 40| F0[Normal — 0]
    E1 -->|5-40 km/h + 15-25 veh| F1[Reducido — 1]
    E1 -->|< 5 km/h + >25 veh| F2[Atascado — 2]
    E1 -->|~0 km/h + desaceleración| F3[Accidente — 3]

    F0 & F1 & F2 & F3 --> G[Dataset etiquetado]

    G --> H[Train/Test Split<br/>80/20 estratificado]
    H --> H1[Training Set]
    H --> H2[Test Set<br/>sin modificar]

    H1 --> I[SMOTE<br/>Balanceo de clases]
    I --> J[StandardScaler]

    J --> K[MLP Training<br/>Input 13 → Dense 64 → Dense 32 → Softmax 4]
    K --> K1[EarlyStopping<br/>patience=15]
    K --> K2[ReduceLROnPlateau<br/>patience=5]

    K1 & K2 --> L[Modelo entrenado]

    L --> M[Evaluación]
    H2 --> M
    M --> M1[F1-macro ≥ 0.85]
    M --> M2[Confusion Matrix]
    M --> M3[Recall por clase]

    L --> N[Exportar artefactos]
    N --> N1[traffic_classifier.keras]
    N --> N2[feature_scaler.joblib]
    N --> N3[label_mapping.joblib]

    L --> O[Persistencia]
    D --> O
    O --> P[(telemetry_raw<br/>14 features + FK)]
    O --> Q[(traffic_classifications<br/>predicción + HITL)]
```

## Artefactos Generados

| Artefacto | Ruta | Propósito |
|---|---|---|
| Modelo Keras | `models/intelligence/traffic_classifier.keras` | Clasificador entrenado |
| Scaler | `models/intelligence/feature_scaler.joblib` | Normalización para inferencia |
| Label mapping | `models/intelligence/label_mapping.joblib` | Mapeo código→nombre de estado |
| CSV features | `data/processed/traffic_telemetry.csv` | Dataset con features para reproducibilidad |

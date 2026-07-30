<!-- context: VAAET/docs/diagrams/intelligence-pipeline.md — Module 1 Intelligence Pipeline.
Referenced by DDS.md, ADR-008, notebook 01_data_prep/data_preparation.ipynb. -->

# Intelligence Pipeline — Module 1

Complete flow from Module 0 telemetry to traffic state classification and persistence.

```mermaid
flowchart TD
    A[(traffic_data<br/>PostgreSQL)] -->|SQL query| B[DataFrame<br/>~2000 records x 9 fields]

    B --> C[Feature Engineering]
    C --> C1[heavy_vehicle_ratio<br/>delta_speed / delta_count<br/>transition_flag<br/>speed_variance<br/>hour_of_day / weather_condition]
    C1 --> D[DataFrame<br/>14 features]

    D --> E[Auto-Labeling]
    E --> E1{Engineering rules}
    E1 -->|avg_speed > 40| F0[Normal — 0]
    E1 -->|5-40 km/h + 15-25 veh| F1[Reduced — 1]
    E1 -->|< 5 km/h + >25 veh| F2[Congested — 2]
    E1 -->|~0 km/h + deceleration| F3[Accident — 3]

    F0 & F1 & F2 & F3 --> G[Labeled dataset]

    G --> H[Train/Test Split<br/>80/20 stratified]
    H --> H1[Training Set]
    H --> H2[Test Set<br/>unmodified]

    H1 --> I[SMOTE<br/>Class balancing]
    I --> J[StandardScaler]

    J --> K[MLP Training<br/>Input 14 → Dense 64 → Dense 32 → Softmax 4]
    K --> K1[EarlyStopping<br/>patience=15]
    K --> K2[ReduceLROnPlateau<br/>patience=5]

    K1 & K2 --> L[Trained model]

    L --> M[Evaluation]
    H2 --> M
    M --> M1[F1-macro >= 0.85]
    M --> M2[Confusion Matrix]
    M --> M3[Per-class Recall]

    L --> N[Export artifacts]
    N --> N1[traffic_classifier.keras]
    N --> N2[feature_scaler.joblib]
    N --> N3[label_mapping.joblib]

    L --> O[Persistence]
    D --> O
    O --> P[(telemetry_raw<br/>14 features + FK)]
    O --> Q[(traffic_classifications<br/>prediction + HITL)]
```

## Generated Artifacts

| Artifact | Path | Purpose |
|---|---|---|
| Keras model | `models/intelligence/traffic_classifier.keras` | Trained classifier |
| Scaler | `models/intelligence/feature_scaler.joblib` | Normalization for inference |
| Label mapping | `models/intelligence/label_mapping.joblib` | Code-to-state-name mapping |
| Features CSV | `data/processed/traffic_telemetry.csv` | Feature dataset for reproducibility |

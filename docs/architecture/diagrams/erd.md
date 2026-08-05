# ERD vigente — VAAET ML 4.1.0

El diagrama y diccionario canónicos están en el
[modelo PostgreSQL `vaaet-db-v2`](../data-model.md). La separación vigente es
`vaaet_raw.traffic_data` → `vaaet_ml.telemetry_features` →
`vaaet_ml.traffic_predictions` → `vaaet_feedback.human_validations`.

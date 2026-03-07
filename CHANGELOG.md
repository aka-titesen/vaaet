# Changelog

Todos los cambios notables del proyecto VAAET se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [2.0.0] - 2026-03-07

### Agregado

- Etapa 2: Clasificador de estado de tráfico con TensorFlow/Keras MLP
- 4 estados de tráfico: Normal, Reducido, Atascado, Accidente
- Ingeniería de features: 9 campos crudos → 14 features
- Auto-labeling con reglas de ingeniería de tránsito
- Balanceo de clases con SMOTE (imbalanced-learn)
- Dos tablas nuevas: `telemetry_raw` (14 features + FK), `traffic_classifications` (predicción + HITL)
- Diagrama de pipeline de inteligencia (Mermaid)
- Diagrama ERD ampliado con 3 tablas y FK chain
- ADR-008: TensorFlow/Keras para clasificación de tráfico

### Cambiado

- Proyecto reestructurado: notebooks/, models/, data/, src/, docs/
- Documentación actualizada: README, DDS, DATA_LINEAGE, KPIs, BIAS_AND_LIMITATIONS, AGENTS, CONTRIBUTING
- requirements.txt ampliado con 7 nuevas dependencias (tensorflow, pandas, sqlalchemy, etc.)
- .gitignore actualizado para *.keras, *.joblib, data/processed/*.csv

## [1.0.0] - 2026-03-06

### Agregado

- Pipeline completo de análisis de tráfico vehicular para el Puente Gral. Manuel Belgrano
- Detección y clasificación con YOLO 11 (5 variantes: n/s/m/l/x)
- Selección automática de modelo según duración del video
- Tracking persistente con SORT ligero
- Cálculo híbrido de velocidad: 70% física + 30% MLP suavizador
- Compensación de movimiento de cámara via Optical Flow (Lucas-Kanade)
- Corrección de perspectiva adaptativa por coordenada Y
- Detección ultra-conservadora de vehículos estacionarios (AND-conjunction)
- Soporte multi-cámara: detección automática de layout (1, 2, 4 vistas)
- Persistencia opcional en PostgreSQL (AWS RDS) cada minuto
- Video de salida con anotaciones, overlays y HUD informativo
- Generador de videos sintéticos para demos de portfolio
- Interfaz universal de carga (Colab + local)
- Optimización para Google Colab Free/Pro (frame skipping, limpieza de memoria)
- Documentación: PRD, DDS, Guía de Usuario, KPIs, ADRs
- Infraestructura documental: AGENTS.md, llms.txt, diagramas Mermaid, data lineage

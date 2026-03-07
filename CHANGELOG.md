# Changelog

Todos los cambios notables del proyecto VAAET se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

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

# 📋 PRD - VAAET: Sistema Avanzado de Análisis de Tráfico Vehicular

## ✅ Validación de Cumplimiento de Requisitos

El sistema VAAET cumple con todos los requisitos funcionales y de calidad definidos para el análisis de tránsito en el Puente General Manuel Belgrano, integrando visión artificial avanzada, persistencia segura, outputs claros y modularidad robusta. La solución está alineada con el contexto dinámico del problema y las necesidades de los stakeholders.

### Resumen de Cumplimiento (actualizado)

1. Carga y validación de video: Solo acepta archivos con formato `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`. Si no cumple, aborta e informa.
2. Selección automática de modelo YOLO 11: Elige entre yolo11x/l/m/s/n según duración (<1h, 1-3h, 3-6h, 6-12h, >12h).
3. Persistencia opcional: El usuario decide si persiste en PostgreSQL AWS RDS, con datos válidos cada minuto y sin credenciales hardcodeadas.
4. Cálculo híbrido de velocidad: Combina método real, Optical Flow Farneback y CNN, con exclusión de estacionados y límites por tipo.
5. Multi-cámara y perspectiva: Detección automática de layout (1, 2, 4 vistas), homografía calibrable, y adaptación a cambios de cámara y zoom.
6. Seguimiento robusto: Tracking persistente (SORT), IDs únicos, y exclusión de vehículos fuera de toma.
7. Históricos y outputs: Panel informativo y persistencia usan promedios recientes cuando no hay lecturas; video procesado con overlays, métricas y descarga automática. No se generan JSON/CSV; la persistencia per-minute es opcional en PostgreSQL.
8. Optimización Colab: Frame skipping, limpieza de memoria, soporte para entornos gratuitos/pro.
9. Modularidad y robustez: Código desacoplado, funciones auxiliares, logging y gestión de errores.
10. Notebook compacto: ~8–10 celdas, outputs concisos, mensajes claros de éxito/error.
11. Gestión segura de credenciales: Uso de variables de entorno, nunca expone datos sensibles.
12. Base de datos alineada: Persistencia en tabla `traffic_data` según el esquema requerido.
13. Contexto dinámico: Adaptación a cámaras móviles, zoom, ángulos variables y condiciones reales del puente.

### Notas de implementación actual

- Autodiagnóstico: Hay una celda que verifica Ultralytics y descarga pesos yolo11\*.pt si faltan.
- Selección de modelo: Si el nombre local es "yolov11*.pt", el sistema lo normaliza a "yolo11*.pt" automáticamente.
- Persistencia: Se pregunta una sola vez y prioriza variables de entorno (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD).
- Ejecución: Hay una celda final que ejecuta `process_bridge_video()` y muestra un resumen conciso.

---

## 🎯 Resumen Ejecutivo (Actualizado)

VAAET es un sistema híbrido de inteligencia artificial para el análisis de tráfico vehicular en el Puente General Manuel Belgrano, que integra detección YOLO 11, Optical Flow, CNN y persistencia segura en PostgreSQL. Resuelve los problemas históricos de los sistemas previos, como la asignación errónea de velocidades a vehículos estacionados, la clasificación inconsistente y la falta de robustez ante el contexto dinámico de cámaras SISE.

El sistema está optimizado para Google Colab, es modular, escalable, seguro y cumple con todos los requisitos funcionales y de calidad definidos por el usuario y los stakeholders.

---

### 🔐 Propuesta de Valor Única (Validada)

- Sistema híbrido robusto: Combina detección YOLO 11, Optical Flow y CNN para máxima precisión y exclusión de estacionados.
- Selección dinámica de modelo: Elige automáticamente el modelo YOLO 11 óptimo según la duración del video.
- Persistencia segura: Integración con PostgreSQL AWS RDS, sin exponer credenciales.
- Outputs claros y concisos: Video procesado con overlays, métricas y panel informativo minimalista.
- Optimización Colab: Adaptado a recursos y limitaciones de Google Colab Free/Pro.
- Gestión de contexto dinámico: Soporta cambios de cámara, zoom, multi-vista y condiciones reales del puente.
- Modularidad y escalabilidad: Código desacoplado, funciones auxiliares, notebook compacto y outputs claros.

---

## 🧩 Arquitectura de Solución y Metodología Técnica (resumen actualizado)

- Entorno y autodiagnóstico (celdas 1-2): Detecta Colab/local y verifica/descarga pesos yolo11.
- Carga/validación del video y selección de modelo (celda 3): Valida nombre, extrae duración y elige yolo11\*.
- Inicialización del motor híbrido (celda 4): Clases y utilidades (detección estacionados, límites por tipo, perspectiva).
- Procesamiento híbrido y celda final (celdas 5-6): Detección + tracking + Optical Flow + CNN; ejecución con `process_bridge_video()`.
- Mejoras opcionales (celda avanzada): SORT ligero, Farneback y homografías externas.

### 🧪 YOLO 11: Modelo de Detección de Vanguardia

- Modelos: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x.
- Selección por duración: ≤1h→x, 1–3h→l, 3–6h→m, 6–12h→s, >12h→n.
- Autodiagnóstico: descarga automática de pesos faltantes.

### 🔧 Dependencias y Librerías (actualizadas)

- Principales: ultralytics, opencv-python, numpy, scikit-learn, scipy, torch/torchvision.
- Opcional BD: psycopg2-binary.

### 🗄️ Persistencia en PostgreSQL (opcional)

- Frecuencia: Un registro por minuto (avg_speed, conteos por tipo, total).
- Esquema: Tabla `traffic_data` con UNIQUE(clip_id, record_time).
- Seguridad: Variables de entorno; no se exponen credenciales; prompt único si faltan.

---

## 🧭 Casos de Uso (resumen)

- Monitoreo operacional SISE, ingeniería de tráfico, planificación urbana, investigación académica, gestión de emergencias.

---

## 📈 KPIs (sin cambios sustanciales)

- Precisión de detección, precisión de velocidades, estacionados, eficiencia de procesamiento, disponibilidad, usabilidad, escalabilidad.

---

## 🖥️ Entornos compatibles (resumen)

- Google Colab Free/Pro: configuraciones y límites típicos.
- Local con GPU: CUDA opcional; CPU soportado (más lento).
- Cloud alternativo: AWS/Azure/GCP con GPU.

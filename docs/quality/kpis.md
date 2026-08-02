<!-- context: VAAET/docs/quality/kpis.md — Métricas de rendimiento y guía de validación.
Complementa PRD.md (requisitos) y BIAS_AND_LIMITATIONS.md (limitaciones). -->

# Métricas de Rendimiento (KPIs) y Guía de Validación — VAAET

Este documento describe los Indicadores Clave de Rendimiento (KPIs) que definen el éxito del sistema VAAET y provee una guía detallada para validar los objetivos de precisión declarados.

## 1. KPIs del Sistema

### 1.1 Precisión de Detección y Clasificación

- **Qué**: Mide la capacidad del sistema para identificar correctamente vehículos y asignar la clase correcta (auto, camión, colectivo, motocicleta, bicicleta). Objetivo: **97%**.
- **Por qué**: Base del sistema. Si la detección falla, todos los cálculos posteriores son incorrectos.
- **Cómo**: Medido vía **F1-Score** que combina Precisión y Recall.
  - **Precisión**: `TP / (TP + FP)` — de los vehículos detectados, qué porcentaje fue correcto
  - **Recall**: `TP / (TP + FN)` — de los vehículos reales, qué porcentaje fue detectado

### 1.2 Precisión del Cálculo de Velocidad

- **Qué**: Diferencia entre la velocidad calculada por VAAET y la velocidad real del vehículo.
- **Por qué**: Crítico para el análisis del flujo de tráfico y la toma de decisiones.
- **Cómo**: **Error Absoluto Medio (MAE)** = `(1/n) * Σ|VelocidadReal - VelocidadPredicha|`. Objetivo: MAE < 5 km/h.

### 1.3 Confiabilidad del Tracking

- **Qué**: Capacidad de mantener IDs únicos consistentes entre frames.
- **Por qué**: Esencial para el cálculo de velocidad y evitar doble conteo.
- **Cómo**: **Identity Switches (Cambios de ID)** — número de cambios de ID por video. Objetivo: minimizar a casi cero.

### 1.4 Eficiencia de Procesamiento

- **Qué**: Velocidad de procesamiento de video.
- **Por qué**: Determina la viabilidad para procesar grandes volúmenes de video.
- **Cómo**: **Frames Por Segundo (FPS)** = total de frames / tiempo de procesamiento.

### 1.5 Robustez de Detección de Estacionarios

- **Qué**: Efectividad de `is_stationary()` para identificar correctamente vehículos detenidos.
- **Por qué**: Previene la contaminación de la velocidad promedio por vehículos que no se mueven.
- **Cómo**: Tasas de Verdaderos Positivos y Falsos Positivos para clasificación de estacionarios.

---

## 2. Guía de Validación del Objetivo de 97% de Precisión

### Paso 0: Preparación del Entorno

1. **Video de test**: Clip representativo de 2-5 minutos del Puente Belgrano (no usado para entrenamiento)
2. **Herramienta de anotación**: CVAT, VGG Image Annotator (VIA), o similar

### Paso 1: Crear Ground Truth

1. Cargar el video en la herramienta de anotación
2. Definir etiquetas de clase: `auto`, `camión`, `colectivo`, `motocicleta`, `bicicleta`
3. Anotar cada frame (o cada N frames): dibujar bounding boxes, asignar clases, asignar IDs de tracking
4. Exportar anotaciones (JSON/XML/CSV) — esto es tu Ground Truth

### Paso 2: Ejecutar VAAET

1. Procesar el mismo video de test con el notebook de producción
2. Exportar detecciones por frame: número de frame, bbox [x1,y1,x2,y2], clase, track ID, confianza

### Paso 3: Comparar y Calcular

1. Cargar ambos archivos en un script de comparación
2. Hacer matching por frame usando umbral de IoU > 0.5
3. Clasificar: TP (match correcto), FP (sin ground truth correspondiente), FN (ground truth no detectado)
4. Calcular: `Precisión`, `Recall`, `F1-Score = 2*(P*R)/(P+R)`

### Paso 4: Interpretación

F1-Score ≥ 0.97 valida el objetivo de 97% de precisión. Si está por debajo, el desglose de Precisión/Recall indica las áreas de mejora.

---

## 3. Estado Actual de Medición

> **Importante**: Los objetivos de KPI listados aquí son **metas declaradas** aún no validadas con benchmarks reales.

| KPI | Objetivo | Estado de Medición |
|---|---|---|
| F1-Score Detección | 97% | Sin benchmark real. Requiere ground truth anotado manualmente |
| MAE Velocidad | < 5 km/h | Sin ground truth de velocidad. Sin datos de referencia del puente |
| Cambios de ID | Minimizar | Sin medición formal. Requiere ground truth con IDs consistentes |
| FPS Procesamiento | Variable | No publicado. Depende del modelo YOLO y GPU de Colab |
| Precisión Estacionarios | Alta | Sin evaluación cuantitativa. Validado cualitativamente con demos sintéticos |
| F1-macro Clasificación | ≥ 0.85 | Registrado en el manifiesto de cada bundle |
| Recall clase Accidente | > 0 | Pendiente. Clase extremadamente rara |

### Prerrequisitos de Validación

1. **Video de test** representativo del puente (2-5 minutos)
2. **Anotación manual** con herramienta CVAT o VIA (Paso 1 de la guía)
3. **Script de comparación** con IoU > 0.5 (no provisto actualmente — debe implementarse)
4. **Datos reales de velocidad** (radar o GPS) para validación de MAE — actualmente no disponibles

### Limitaciones Conocidas

Ver [sesgos y limitaciones](../ml/bias-and-limitations.md) para el análisis completo de sesgos que afectan los KPIs.

---

## 4. KPIs de Clasificación de Estados del Tráfico (Módulos 1 y 2)

Estos KPIs evalúan la calidad del clasificador MLP.

### 4.1 F1-Score Macro

- **Qué**: Promedio no ponderado del F1-Score por clase. Trata todas las clases por igual independientemente de su frecuencia.
- **Por qué**: Penaliza el rendimiento pobre en clases raras (Accidente ~0,1%) y clases frecuentes (Normal ~80%) de forma equitativa.
- **Objetivo**: ≥ 0.85
- **Cómo**: `sklearn.metrics.f1_score(y_test, y_pred, average='macro')`

### 4.2 Recall por Clase

- **Qué**: De todos los registros que verdaderamente pertenecen a una clase, ¿qué porcentaje detectó el modelo?
- **Por qué**: Recall de 0 para Accidente significa que el modelo NUNCA detecta la clase más crítica operativamente.
- **Objetivo**: > 0 para todas las clases presentes
- **Especialmente crítico**: Accidente (recall > 0 es el mínimo aceptable)

### 4.3 Matriz de Confusión

- **Qué**: Tabla que muestra la distribución de predicciones vs. realidad para cada par de clases.
- **Confusiones esperadas**:
  - Normal ↔ Reducido: frontera difusa a ~40 km/h
  - Congestionado ↔ Reducido: frontera difusa a ~5 km/h
  - Accidente raramente confundido con Normal (velocidades muy diferentes)

### 4.4 Estado Actual de Medición (Módulos 1 y 2)

| KPI | Objetivo | Estado |
|---|---|---|
| F1-macro | ≥ 0.85 | Pendiente primera ejecución |
| Recall Accidente | > 0 | Pendiente (clase rara, SMOTE mitiga) |
| Recall Normal | > 0.90 | Pendiente (clase mayoritaria) |
| Recall Reducido | > 0.50 | Pendiente |
| Recall Congestionado | > 0.50 | Pendiente |

<!-- context: VAAET/Docs/KPIs/KPIs.md — Métricas de rendimiento y guía de validación.
Complementa PRD.md (requisitos) y BIAS_AND_LIMITATIONS.md (limitaciones). -->

# Métricas de Rendimiento (KPIs) y Guía de Validación para el Sistema VAAET

Este documento describe los Indicadores Clave de Rendimiento (KPIs) que definen el éxito del sistema VAAET y proporciona una guía detallada para validar el objetivo de precisión del 97% mencionado en la documentación del proyecto.

## 1. Indicadores Clave de Rendimiento (KPIs) del Sistema

Los KPIs son las métricas cuantificables que permiten evaluar la efectividad y calidad del sistema en sus tareas principales. Para VAAET, estos se centran en la precisión de la detección, el cálculo de velocidad y la eficiencia del procesamiento.

### 1.1. Precisión de Detección y Clasificación

* **¿Qué es?** Mide la capacidad del sistema para identificar correctamente los vehículos y asignarles la clase correcta (ej. 'car', 'truck', 'bus'). El objetivo es alcanzar un **97%** en esta métrica.
* **¿Por qué es importante?** Es el pilar del sistema. Si la detección falla, todos los cálculos posteriores (conteo, velocidad) serán incorrectos. Una alta precisión garantiza que los datos generados son fiables.
* **¿Cómo se mide?** Se mide utilizando la métrica **F1-Score**, que combina la Precisión y la Exhaustividad (Recall) de las detecciones del modelo YOLO 11.
    * **Precisión (Precision):** De todos los vehículos que el sistema *dice* haber detectado, ¿qué porcentaje era correcto? Su fórmula es `VP / (VP + FP)`. Un valor alto indica pocos falsos positivos.
    * **Exhaustividad (Recall):** De todos los vehículos que *realmente* había en el video, ¿qué porcentaje logró encontrar el sistema? Su fórmula es `VP / (VP + FN)`. Un valor alto indica que el sistema no omite vehículos.
    * **Términos Técnicos:**
        * `Verdadero Positivo (VP)`: El sistema detecta un auto y realmente era un auto.
        * `Falso Positivo (FP)`: El sistema detecta un auto, pero era una sombra o no había nada.
        * `Falso Negativo (FN)`: Había un auto real en el video, pero el sistema no lo detectó.

### 1.2. Precisión del Cálculo de Velocidad

* **¿Qué es?** Mide la diferencia entre la velocidad calculada por el sistema VAAET y la velocidad real de los vehículos.
* **¿Por qué es importante?** La estimación de velocidad es una de las funcionalidades principales. Su precisión es crítica para el análisis del flujo de tráfico y la toma de decisiones.
* **¿Cómo se mide?** Se utiliza el **Error Absoluto Medio (MAE - Mean Absolute Error)**. Se calcula la diferencia absoluta entre la velocidad predicha y la velocidad real para cada vehículo y luego se promedian todos esos errores. La fórmula es `(1/n) * Σ|VelocidadReal - VelocidadPredicha|`.
    * Un objetivo para este KPI sería, por ejemplo, "lograr un MAE inferior a 5 km/h".

### 1.3. Fiabilidad del Tracking de Vehículos

* **¿Qué es?** Mide la capacidad del sistema para asignar un ID único a un vehículo y mantenerlo consistentemente mientras este aparece en pantalla, sin cambiarlo o asignarlo a otro vehículo.
* **¿Por qué es importante?** Un tracking fiable es esencial para el cálculo de velocidad (que depende del historial de posiciones) y para evitar el conteo múltiple del mismo vehículo.
* **¿Cómo se mide?** Se mide principalmente a través de la métrica **Identity Switches (ID Switches)**. Un "ID Switch" ocurre cuando el sistema cambia el ID de un vehículo que ya estaba siendo rastreado. El objetivo es minimizar este número, idealmente a 0 o un valor muy cercano en un video de prueba.

### 1.4. Eficiencia de Procesamiento (Rendimiento)

* **¿Qué es?** Mide la velocidad a la que el sistema puede procesar el video.
* **¿Por qué es importante?** Determina la viabilidad del sistema para aplicaciones en tiempo real o para procesar grandes volúmenes de video en un tiempo razonable.
* **¿Cómo se mide?** Se mide en **Fotogramas Por Segundo (FPS)**. Se calcula dividiendo el número total de fotogramas del video por el tiempo total que tardó el sistema en procesarlo.

### 1.5. Robustez ante Detección de Estacionados

* **¿Qué es?** Mide la efectividad del algoritmo `is_stationary` para identificar correctamente vehículos detenidos y excluirlos del cálculo de velocidad promedio.
* **¿Por qué es importante?** Evita que la velocidad promedio del tráfico se vea incorrectamente reducida por vehículos que no están en movimiento, lo cual era un problema clave que VAAET soluciona.
* **¿Cómo se mide?** Se evalúa contando cuántos vehículos realmente detenidos en el video de prueba fueron marcados como estacionados (Verdaderos Positivos) y cuántos vehículos en movimiento fueron marcados erróneamente como detenidos (Falsos Positivos).

---

## 2. Guía para la Validación del Objetivo de Precisión del 97%

Este es un proceso técnico que requiere rigor para ser válido. Sigue estos pasos para probar la afirmación.

### **Paso 0: Preparación del Entorno y Herramientas**

Antes de empezar, necesitarás:
1.  **Un video de prueba:** Debe ser un clip representativo del Puente General Manuel Belgrano, de unos 2 a 5 minutos, que el sistema no haya usado para entrenar.
2.  **Una herramienta de anotación de video:** Sirve para etiquetar manualmente los objetos en el video. Algunas herramientas de código abierto recomendadas son:
    * **CVAT (Computer Vision Annotation Tool):** Potente, basada en web e ideal para equipos.
    * **VGG Image Annotator (VIA):** Más simple, se ejecuta directamente en el navegador.

### **Paso 1: Creación del "Ground Truth" (La Verdad Absoluta)**

Este es el paso más crucial. Usando la herramienta de anotación, debes crear el conjunto de datos de referencia.

1.  **Carga el video** en la herramienta de anotación.
2.  **Define las etiquetas de clase:** `car`, `truck`, `bus`, `motorcycle`, `bicycle`.
3.  **Anota cada fotograma (o cada N fotogramas, ej. cada 5):**
    * **Dibuja un Bounding Box (caja delimitadora)** alrededor de CADA vehículo visible.
    * **Asigna la etiqueta de clase correcta** a cada caja.
    * **Asigna un ID de seguimiento único** a cada vehículo. Este ID debe ser el mismo para ese vehículo en todos los fotogramas en los que aparece.
4.  **Exporta las anotaciones:** Al final, la herramienta te dará un archivo (generalmente en formato JSON, XML o CSV) que contiene la información de todas las cajas, clases e IDs para cada fotograma. **Este archivo es tu Ground Truth.**

### **Paso 2: Ejecución de VAAET para Obtener las Predicciones**

1.  Toma el **mismo video de prueba (el archivo `.mp4` original, sin las anotaciones)**.
2.  Procesa este video utilizando tu *notebook* `vaaet.ipynb`.
3.  Modifica ligeramente el código para que, en lugar de (o además de) dibujar en el video, **guarde las detecciones de cada fotograma en un archivo de texto o JSON**. Para cada detección, necesitas guardar:
    * El número de fotograma.
    * Las coordenadas del Bounding Box (x1, y1, x2, y2).
    * La clase predicha por YOLO (ej. 'car').
    * El ID de seguimiento asignado por tu sistema.
    * La confianza de la detección.

### **Paso 3: Comparación y Cálculo de la Métrica**

Ahora tienes dos archivos: el Ground Truth (realidad) y las Predicciones (lo que VAAET hizo). Necesitarás un script de Python (usando librerías como `pandas` y `numpy`) para compararlos.

1.  **Carga ambos archivos** en tu script.
2.  **Itera fotograma por fotograma:**
    * Para cada fotograma, compara las cajas del Ground Truth con las cajas de tus Predicciones. Se utiliza una métrica llamada **IoU (Intersection over Union)** para determinar si una caja predicha coincide con una caja real. Generalmente, un IoU > 0.5 se considera una coincidencia.
3.  **Clasifica cada detección:**
    * Si una caja predicha coincide con una caja real de la misma clase, es un **Verdadero Positivo (VP)**.
    * Si una caja predicha no coincide con ninguna caja real, es un **Falso Positivo (FP)**.
    * Si una caja real del Ground Truth no tuvo ninguna caja predicha que coincidiera, es un **Falso Negativo (FN)**.
4.  **Calcula las métricas finales:**
    * `Precision = VP / (VP + FP)`
    * `Recall = VP / (VP + FN)`
    * `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

### **Paso 4: Interpretación del Resultado**

El valor del **F1-Score** será un número entre 0 y 1. Si el resultado es **0.97 o superior**, has validado exitosamente que tu sistema VAAET cumple con el objetivo de precisión del 97%. Si no, el análisis de la Precisión y el Recall te dirá dónde mejorar (por ejemplo, un Recall bajo significa que necesitas hacer tu modelo más sensible para que no omita vehículos).

---

## 3. Estado Actual de Medición

> **Importante**: Los targets de KPIs listados en este documento son **objetivos declarados** que aún no han sido validados con benchmarks reales.

| KPI | Target | Estado de Medición |
|---|---|---|
| F1-Score Detección | 97% | Sin benchmark real. Requiere ground truth anotado manualmente |
| MAE Velocidad | < 5 km/h | Sin ground truth de velocidades. No existen datos de referencia para el puente |
| ID Switches | Minimizar | Sin medición formal. Requiere ground truth con IDs consistentes |
| FPS Procesamiento | Variable | No publicado. Depende de modelo YOLO y GPU asignada por Colab |
| Precisión Estacionarios | Alta | Sin evaluación cuantitativa. Validado cualitativamente con demos sintéticas |

### Prerequisitos para Validación

1. **Video de prueba** representativo del puente (2-5 minutos)
2. **Anotación manual** con herramienta como CVAT o VIA (Paso 1 de la guía)
3. **Script de comparación** IoU > 0.5 (no proporcionado actualmente — debe implementarse)
4. **Datos de velocidad real** (radar o GPS) para validar MAE — actualmente no disponibles

### Limitaciones Conocidas

Para un análisis completo de sesgos y limitaciones que afectan los KPIs, consultar [BIAS_AND_LIMITATIONS.md](../BIAS_AND_LIMITATIONS.md).
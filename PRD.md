# 📋 PRD - VAAET: Sistema Simple de Análisis de Tráfico

## 🎯 Resumen Ejecutivo

**VAAET** (Video Analysis for Advanced Traffic Engineering) es un sistema **simplificado** de análisis de tráfico vehicular que utiliza inteligencia artificial YOLOv11 para procesar videos del Puente General Manuel Belgrano y generar estadísticas en tiempo real.

### 🔑 Valor Único

- **📊 Análisis Automático**: Detección, tracking y velocidad sin intervención
- **🤖 IA Adaptativa**: Selección dinámica de modelos según duración del video
- **☁️ Cloud-Ready**: Optimizado para Google Colab Free y Pro
- **🗄️ Persistencia Simple**: Solo PostgreSQL en AWS RDS (opcional)
- **⚡ Ultra Simple**: 10 celdas ejecutables con "Run All"

---

## � Stakeholders

### 🎯 Usuarios Primarios

- **Investigadores de Tráfico**: Análisis de patrones vehiculares
- **Autoridades de Transporte**: Monitoreo del puente
- **Operadores SISE**: Verificación de funcionamiento de cámaras

### 🤝 Usuarios Secundarios

- **Desarrolladores**: Integración con sistemas existentes
- **Analistas de Datos**: Procesamiento de estadísticas

4. **Detección de Estacionamiento**: Identificación de vehículos estacionados
5. **Almacenamiento Temporal**: Datos por minuto para análisis estadísticos

### Desafíos Técnicos

- **Perspectiva Variable**: Múltiples ángulos y alturas de cámara
- **Condiciones Dinámicas**: Zoom, movimiento, múltiples capturas
- **Visión Cenital**: Vehículos vistos desde el techo
- **Condiciones Ambientales**: Día/noche, clima variable

## 🤖 Tecnología Base

### Modelo de IA: YOLOv11

- **Versiones Dinámicas**: Nano, Small, Medium, Large, Extra Large
- **Selección Automática**: Basada en duración del clip
- **Optimización**: Recursos de Google Colab (Free/Pro)

### Criterios de Selección de Modelo

| Duración del Clip | Modelo YOLOv11 | Justificación               |
| ----------------- | -------------- | --------------------------- |
| ≤ 1 hora          | Extra Large    | Máxima precisión            |
| 1-3 horas         | Large          | Balance precisión/velocidad |
| 3-6 horas         | Medium         | Eficiencia moderada         |
| 6-12 horas        | Small          | Procesamiento rápido        |
| > 12 horas        | Nano           | Máxima velocidad            |

## 📊 Arquitectura de Datos

### Base de Datos PostgreSQL (AWS RDS)

- **Almacenamiento**: Opcional según preferencia del usuario
- **Frecuencia**: Registros por minuto
- **Campos**: Timestamp, velocidades, conteos por tipo, totales
- **Persistencia**: Configurable (enable_db = True/False)

### Estructura de Datos

```sql
traffic_data (
    id SERIAL PRIMARY KEY,
    clip_id TEXT NOT NULL,
    record_time TIMESTAMP NOT NULL,
    avg_speed NUMERIC(5,2) NOT NULL,
    count_car INTEGER NOT NULL,
    count_truck INTEGER NOT NULL,
    count_bus INTEGER NOT NULL,
    count_motorcycle INTEGER NOT NULL,
    count_bicycle INTEGER NOT NULL,
    total_vehicles INTEGER NOT NULL
)
```

## 🎯 Casos de Uso

### Usuarios Objetivo

- **Ingenieros de Tráfico**: Análisis de flujo vehicular
- **Autoridades de Transporte**: Monitoreo y planificación
- **Investigadores**: Estudios de movilidad urbana
- **Operadores SISE**: Supervisión de infraestructura

### Aplicaciones

1. **Monitoreo en Tiempo Real**: Detección de congestiones
2. **Análisis Histórico**: Patrones de tráfico por horarios/días
3. **Planificación Urbana**: Datos para mejoras de infraestructura
4. **Seguridad Vial**: Detección de velocidades anómalas
5. **Mantenimiento**: Identificación de vehículos estacionados

## 🔧 Requisitos Técnicos

### Entorno de Ejecución

- **Plataforma**: Google Colab (Free/Pro)
- **GPU**: Recomendada para YOLOv11
- **RAM**: Variable según modelo seleccionado
- **Almacenamiento**: Temporal para clips y resultados

### Dependencias Principales

- ultralytics (YOLOv11)
- opencv-python
- psycopg2-binary
- numpy, tqdm
- Optimización de memoria y caché

## 📈 Métricas de Éxito

### KPIs Técnicos

- **Precisión de Detección**: > 90% por tipo de vehículo
- **Velocidad de Procesamiento**: Real-time o cerca del real-time
- **Estabilidad**: Procesamiento completo sin errores
- **Eficiencia**: Uso óptimo de recursos Colab

### KPIs de Negocio

- **Disponibilidad**: 24/7 para análisis de clips
- **Escalabilidad**: Manejo de clips de hasta 12+ horas
- **Usabilidad**: Interface simple en Jupyter Notebook
- **Flexibilidad**: Configuración dinámica de parámetros

# 📖 GUÍA DE USUARIO - VAAET

## 🚦 Sistema Simple de Análisis de Tráfico

Esta guía te ayudará a usar el notebook VAAET **simplificado** para analizar tráfico vehicular en el Puente General Manuel Belgrano.

---

## 🎯 Resumen Rápido

**VAAET** es un sistema que usa YOLOv11 para detectar vehículos, calcular velocidades y generar estadísticas de tráfico a partir de videos de las cámaras SISE.

### ⚡ Inicio Rápido (3 minutos)

1. Abrir `vaaet_simple.ipynb` en Google Colab
2. **Runtime → Restart and run all**
3. Cargar tu video cuando se solicite
4. ¡El sistema hace el resto automáticamente!

---

## 📋 Prerequisitos

### 🌐 Entorno Recomendado

- **Google Colab** (Free o Pro)
- **GPU habilitada** (recomendado para mejor rendimiento)
- **Conexión estable a internet**

### 📁 Formato de Video Requerido

- **Formato**: MP4
- **Nomenclatura**: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`
- **Ejemplo**: `bridge_2024-08-09_14-30-00_to_15-30-00.mp4`

> 💡 **Nota**: Si tu video no sigue este formato, el sistema usará configuración automática.

---

## 🔢 Estructura Simplificada (10 Celdas)

### 📖 **Celda 1: Título y Descripción**

- Información general del proyecto
- **Acción**: Solo lectura

### 📦 **Celda 2: Instalación**

- Instala dependencias necesarias (YOLOv11, OpenCV, PostgreSQL)
- **Tiempo**: 1-2 minutos
- **Acción**: Ejecuta automáticamente

### 🔧 **Celda 3: Imports y Configuración**

- Detecta entorno (Colab/Local)
- Verifica GPU disponible
- Configura logging básico
- **Acción**: Ejecuta automáticamente

### 🎯 **Celda 4: Variables y Modelos**

- Configura modelos YOLOv11 (nano → extra_large)
- Define criterios de selección automática
- Inicializa variables globales
- **Acción**: Ejecuta automáticamente

### 📁 **Celda 5: Carga de Video** ⭐

- **Paso más importante para el usuario**
- Sube tu video desde tu computadora
- Valida formato automáticamente
- **Acción**: Subir archivo cuando se solicite

### ⏱️ **Celda 6: Análisis de Duración**

- Analiza duración del video
- Selecciona modelo YOLOv11 óptimo:
  - **≤ 1 hora**: Extra Large (máxima precisión)
  - **1-3 horas**: Large (equilibrado)
  - **3-6 horas**: Medium (eficiente)
  - **6-12 horas**: Small (rápido)
  - **> 12 horas**: Nano (ultra rápido)
- **Acción**: Automática

### 🗄️ **Celda 7: Configuración de BD**

- **SOLO DOS OPCIONES**: PostgreSQL AWS RDS o nada
- **Por defecto**: Deshabilitada
- **Sin JSON, sin CSV** - Solo BD si lo necesitas
- **Acción**: Cambiar `enable_db = True` si quieres BD

### 🚗 **Celda 8: Analizador de Tráfico**

- Define la clase de procesamiento
- Detección, tracking y cálculo de velocidad
- **Acción**: Automática

### 💾 **Celda 9: Función de BD**

- Guarda datos en PostgreSQL AWS RDS
- Solo se ejecuta si `enable_db = True`
- **Acción**: Automática

### 🚀 **Celda 10: Procesamiento Principal** ⭐

- **Ejecuta todo el análisis**
- Genera video procesado
- Guarda en BD si está habilitada
- **Descarga**: Video automáticamente
- **Tiempo**: Variable según duración

---

## ⚡ Flujo de Trabajo Simplificado

### 🎯 Método Recomendado (Ultra Simple)

```
1. 🚀 Runtime > Restart and run all
2. 📁 Subir video cuando se solicite
3. ⚙️  Cambiar enable_db = True si quieres BD
4. ☕ Esperar resultados automáticamente
5. 📥 Descargar video procesado
```

### 🔧 Método Paso a Paso

```
1. ▶️  Ejecutar celdas 1-4 (configuración)
2. 📁 Subir video en celda 5
3. ⏳ Ver análisis automático (celda 6)
4. ⚙️  Configurar BD en celda 7 (opcional)
5. ▶️  Ejecutar celdas 8-10
6. 📥 Descargar resultados
```

---

## � ¿Qué Obtienes?

### 🎥 **Video Procesado**

- **Archivo**: `VAAET_[tu_clip_id].mp4`
- **Contenido**: Video con detecciones, velocidades y contadores

### 💾 **Datos (Solo si BD habilitada)**

- **PostgreSQL**: Tabla `traffic_data` en AWS RDS
- **Campos**: timestamp, velocidad promedio, conteos por vehículo
- **Frecuencia**: Un registro por minuto

### ❌ **Sin Archivos Locales**

- **No JSON**, **No CSV**
- Solo BD PostgreSQL o nada
- Simplifica la gestión de datos

---

## ⏱️ Tiempos de Procesamiento

| Duración Video | Modelo      | Tiempo Estimado\* |
| -------------- | ----------- | ----------------- |
| 30 min         | Extra Large | 15-25 min         |
| 1 hora         | Extra Large | 25-40 min         |
| 2 horas        | Large       | 30-50 min         |
| 4 horas        | Medium      | 45-70 min         |
| 8 horas        | Small       | 60-90 min         |
| 12+ horas      | Nano        | 80-120 min        |

\*En Google Colab con GPU

---

## 🔧 Configuración de BD AWS RDS

### 🔑 **Variables de Entorno (Recomendado)**

```python
import os
db_config = {
    "host": os.getenv('DB_HOST'),
    "port": int(os.getenv('DB_PORT', 5432)),
    "dbname": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD')
}
```

### ⚙️ **Configuración Directa (Demo)**

```python
enable_db = True  # Cambiar a True en celda 7

# En celda 9, configurar:
db_config = {
    "host": "bridge-traffic-db.cb2gcwmaimbx.sa-east-1.rds.amazonaws.com",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "tu-password-seguro"
}
```

---

## 🛡️ Solución de Problemas

### ❌ **Error de Carga de Video**

```
Solución: Verificar formato MP4, reintentar carga
```

### ⚠️ **Error de Memoria**

```
Solución: El sistema reduce resolución automáticamente
```

### 🔄 **Error de BD**

```
Solución: Cambiar enable_db = False, usar solo video
```

### 📱 **Error de Modelo**

```
Solución: Reiniciar runtime, verificar conexión
```

---

## 💡 Consejos de Uso

### 🎯 **Para Mejores Resultados**

- Videos de día funcionan mejor
- Resolución mínima: 640x480
- Duración óptima: 1-4 horas

### ⚡ **Para Mayor Velocidad**

- Usar Google Colab Pro
- Mantener BD deshabilitada si no es necesaria
- Procesar videos en partes si son muy largos

### 🔒 **Para Seguridad**

- Nunca hardcodear passwords
- Usar variables de entorno en producción
- Los videos se procesan localmente en Colab

---

## 🌟 Diferencias con Versión Anterior

| Aspecto           | Versión Anterior    | Versión Simple       |
| ----------------- | ------------------- | -------------------- |
| **Celdas**        | 13 celdas complejas | 10 celdas simples    |
| **Persistencia**  | JSON + CSV + BD     | Solo BD o nada       |
| **Configuración** | Múltiples opciones  | ON/OFF simple        |
| **Ejecución**     | Paso a paso         | "Run all" automático |
| **Archivos**      | 3 tipos de salida   | Solo video + BD      |
| **Complejidad**   | Alta                | Mínima               |

---

¡**Analiza tráfico del Puente General Manuel Belgrano de forma simple con VAAET!** 🚗📊

### 📖 **Celda 1: Título y Descripción**

- Información general del proyecto
- **Acción**: Solo lectura

### ⚙️ **Celda 2: Configuración Inicial**

- Detecta si estás en Colab
- Verifica disponibilidad de GPU
- Configura logging
- **Acción**: Ejecutar automáticamente

### 📦 **Celda 3: Instalación de Dependencias**

- Instala librerías necesarias (YOLOv11, OpenCV, etc.)
- **Tiempo**: 1-2 minutos
- **Acción**: Ejecutar y esperar

### 🔧 **Celda 4: Variables Globales**

- Configura modelos YOLOv11 disponibles
- Define criterios de selección automática
- **Acción**: Ejecutar automáticamente

### 📁 **Celda 5: Carga de Video** ⭐

- **Paso más importante para el usuario**
- Sube tu video desde tu computadora
- Valida el formato automáticamente
- Si el nombre es incorrecto, te permite corregirlo manualmente

#### 🎯 Cómo usar:

1. Ejecutar la celda
2. Hacer clic en "Examinar" cuando aparezca
3. Seleccionar tu archivo MP4
4. Si el nombre es válido: ¡continúa!
5. Si no es válido: llenar fechas/horas manualmente

### ⏱️ **Celda 6: Análisis de Duración**

- Analiza tu video automáticamente
- Selecciona el modelo YOLOv11 óptimo según duración:
  - **≤ 1 hora**: Extra Large (máxima precisión)
  - **1-3 horas**: Large (equilibrado)
  - **3-6 horas**: Medium (eficiente)
  - **6-12 horas**: Small (rápido)
  - **> 12 horas**: Nano (ultra rápido)

### 🗄️ **Celda 7: Configuración de Base de Datos** (Opcional)

- **Por defecto**: Deshabilitada (archivos locales)
- **Si quieres BD**: Marca la casilla y configura credenciales
- **Recomendación**: Mantener deshabilitada para uso básico

### 🏭 **Celda 8: Carga de Modelo**

- Descarga solo el modelo YOLOv11 necesario
- **Tiempo**: 30 segundos - 2 minutos (según modelo)
- **Acción**: Automática

### 🚗 **Celda 9: Inicialización del Analizador**

- Crea el motor de análisis de tráfico
- **Acción**: Automática

### 🎬 **Celda 10: Procesador de Video**

- Define la lógica de procesamiento optimizado
- **Acción**: Automática

### 💾 **Celda 11: Sistema de Persistencia**

- Configura guardado de datos
- **Acción**: Automática

### 🧪 **Celda 12: Tests del Sistema**

- Valida que todo esté funcionando correctamente
- **Tiempo**: 10-30 segundos
- **Acción**: Automática

### 🚀 **Celda 13: Ejecución Final** ⭐

- **Ejecuta todo el análisis**
- **Tiempo**: Variable según duración del video
- **Genera**: Video procesado + datos estadísticos
- **Descarga**: Archivos automáticamente

---

## ⚡ Flujo de Trabajo Recomendado

### 🎯 Para Usuarios Nuevos

```
1. ▶️  Ejecutar celdas 1-4 (configuración)
2. 📁 Cargar video en celda 5
3. ⏳ Esperar análisis automático (celda 6)
4. ⚙️  Configurar BD si es necesario (celda 7)
5. ▶️  Ejecutar celdas 8-12 en secuencia
6. 🚀 Ejecutar celda 13 y esperar resultados
7. 📥 Descargar archivos generados
```

### ⚡ Para Usuarios Experimentados

```
1. ▶️  Runtime > Run All
2. 📁 Subir video cuando se solicite
3. ⚙️  Configurar BD si es necesario
4. ☕ Esperar resultados
```

---

## 📊 ¿Qué Obtienes?

### 🎥 **Video Procesado**

- **Archivo**: `VAAET_ANALYSIS_[tu_clip_id].mp4`
- **Contenido**: Video original con:
  - Cajas delimitadoras alrededor de vehículos
  - Velocidades en tiempo real
  - Contadores por tipo de vehículo
  - Indicadores de vehículos estacionados

### 📈 **Datos Estadísticos**

#### 📄 **Archivo JSON** (`traffic_data_[clip_id].json`)

```json
{
  "clip_id": "bridge_2024-08-09_14-30-00_to_15-30-00",
  "timestamp": "2024-08-09T14:31:00",
  "avg_speed": 45.67,
  "count_car": 12,
  "count_truck": 3,
  "count_bus": 1,
  "count_motorcycle": 2,
  "count_bicycle": 0,
  "total_vehicles": 18
}
```

#### 📊 **Archivo CSV** (`traffic_data_[clip_id].csv`)

- Mismos datos en formato tabular
- Perfecto para análisis en Excel/Python
- Un registro por minuto de video

---

## ⏱️ Tiempos de Procesamiento Estimados

| Duración Video | Modelo Usado | Tiempo Esperado\* |
| -------------- | ------------ | ----------------- |
| 30 minutos     | Extra Large  | 15-25 min         |
| 1 hora         | Extra Large  | 25-40 min         |
| 2 horas        | Large        | 30-50 min         |
| 4 horas        | Medium       | 45-70 min         |
| 8 horas        | Small        | 60-90 min         |
| 12+ horas      | Nano         | 80-120 min        |

\*En Google Colab con GPU. Tiempos en CPU serán 3-5x más lentos.

---

## 🔧 Solución de Problemas

### ❌ **Error: "No se pudo cargar el video"**

**Solución:**

- Verificar que el archivo sea MP4
- Asegurarse de que no esté corrupto
- Intentar con un video más pequeño para probar

### ⚠️ **Warning: "Alto uso de memoria"**

**Solución:**

- El sistema automáticamente reduce la resolución
- Para videos muy largos, considera dividirlos en partes
- Reinicia el runtime si es necesario

### 🔄 **Error: "Timeout de conexión a BD"**

**Solución:**

- Deshabilitar base de datos y usar modo local
- Verificar credenciales si usas BD
- Los datos se guardarán en archivos locales automáticamente

### 🚫 **Error: "Tests fallaron"**

**Solución:**

- Reiniciar el runtime de Colab
- Ejecutar las celdas en orden
- Verificar conexión a internet

### 📱 **Error: "Modelo no se puede descargar"**

**Solución:**

- Verificar conexión a internet
- Reintentar en unos minutos
- El sistema intentará con modelo más pequeño automáticamente

---

## 💡 Consejos de Uso

### 🎯 **Para Mejores Resultados**

- **Videos de día** funcionan mejor que de noche
- **Resolución mínima recomendada**: 640x480
- **Duración óptima**: 1-4 horas por procesamiento

### ⚡ **Para Mayor Velocidad**

- Usar Google Colab Pro (más GPU y RAM)
- Procesar videos en partes si son muy largos
- Usar resoluciones moderadas (no 4K)

### 📊 **Para Análisis**

- Los datos se guardan cada minuto
- Velocidades se promedian usando EMA (suavizado exponencial)
- Vehículos estacionados se detectan automáticamente

### 🔒 **Para Seguridad**

- Nunca hardcodear passwords en el notebook
- Usar variables de entorno para credenciales
- Los videos se procesan localmente en Colab

---

## 📞 Soporte

### 🐛 **Reportar Problemas**

1. Copiar el mensaje de error completo
2. Indicar en qué celda ocurrió
3. Especificar duración y resolución del video
4. Mencionar si usas Colab Free o Pro

### 📚 **Documentación Adicional**

- `PRD.md` - Descripción completa del proyecto
- `Sprints.md` - Plan de desarrollo y características técnicas
- Código está documentado en cada celda

### 🔄 **Actualizaciones**

- Siempre usar la versión más reciente del notebook
- Las mejoras se documentan en los sprints
- El sistema se optimiza continuamente

---

## 🎯 Casos de Uso Comunes

### 👨‍🔬 **Investigador de Tráfico**

```
Objetivo: Analizar patrones de flujo vehicular
Recomendación:
- Videos de 2-4 horas en horarios pico
- Habilitar base de datos para análisis histórico
- Usar modelo Large o Extra Large
```

### 🏛️ **Autoridad de Transporte**

```
Objetivo: Monitoreo continuo del puente
Recomendación:
- Videos de 8-12 horas (día completo)
- Usar modelo Small para procesamiento rápido
- Guardar datos en BD para reportes
```

### 👨‍💼 **Operador SISE**

```
Objetivo: Verificar funcionamiento de cámaras
Recomendación:
- Videos cortos de 30-60 minutos
- Usar modelo Extra Large para máxima precisión
- Modo local sin BD
```

---

## 🌟 Características Destacadas

### 🤖 **Inteligencia Artificial**

- **YOLOv11**: Última versión del detector más avanzado
- **Tracking multi-objeto**: Sigue vehículos entre frames
- **Cálculo de velocidad**: Estimación en tiempo real
- **Detección de estacionamiento**: Identifica vehículos parados

### ⚡ **Optimización Automática**

- **Selección de modelo**: Basada en duración del video
- **Gestión de memoria**: Ajuste automático según recursos
- **Downsampling inteligente**: Para videos largos
- **Caché de modelos**: Evita descargas repetidas

### 🛡️ **Robustez**

- **Manejo de errores**: Recuperación automática
- **Fallback modes**: BD → archivos locales
- **Validación de datos**: Filtros de calidad
- **Cleanup automático**: Gestión de memoria

---

¡**Disfruta analizando el tráfico del Puente General Manuel Belgrano con VAAET!** 🚗📊

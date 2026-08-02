<!-- context: VAAET/docs/USER_GUIDE.md — Guía de usuario.
Complementa README.md (visión general) y SAD.md (arquitectura). -->

# Guía de Usuario — VAAET

## ¿Qué es VAAET?

Un sistema de visión artificial para analizar el tráfico vehicular en el Puente General Manuel Belgrano, utilizando YOLO 11, Flujo Óptico, un pipeline de velocidad physics-first, y un clasificador MLP de TF/Keras para estados del tráfico. Procesa video de vigilancia, clasifica el estado del tráfico, y opcionalmente persiste los resultados en PostgreSQL.

---

## Inicio Rápido

### Módulo 1 — Preparación de Datos (ejecutar una vez)

1. Abrir `notebooks/01_data_prep/data_preparation.ipynb` en Google Colab
2. Ejecutar las celdas requeridas en orden. Las celdas académicas opcionales son `7b` (validación cruzada), `7c` (exportación a Drive), y `8` (persistencia en BD)
3. Configurar credenciales de BD en la Celda 2 vía variables de entorno solo si se desea acceso a la base de datos: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
4. El sistema extrae telemetría, genera 19 features de calidad, y entrena un clasificador MLP
5. Los artefactos se exportan a `models/intelligence/`

### Módulo 2 — Producción (continua)

1. Abrir `notebooks/02_production/traffic_analyzer.ipynb` en Google Colab
2. Ejecutar Celda 0 (setup del entorno) y Celda 1 (cargar modelo entrenado)
3. Subir un video con nombre: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`
4. Ejecutar Celda 2 para procesamiento solo de telemetría, o Celda 2b para video anotado
5. Ejecutar Celda 3 para clasificar el estado del tráfico
6. Ejecutar Celda 4 para persistir resultados en BD (opcional)
7. Tratar la Celda 5 como scaffold experimental de HITL/re-entrenamiento
8. Ejecutar Celda 6 para el dashboard de visualización

---

## Requisitos de Video

- **Formato**: MP4
- **Nombre**: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4` (estricto). Nombres no conformes son rechazados.

---

## Selección Automática de Modelo (YOLO 11)

| Duración | Modelo |
|---|---|
| ≤ 1h | yolo11x.pt |
| 1-3h | yolo11l.pt |
| 3-6h | yolo11m.pt |
| 6-12h | yolo11s.pt |
| > 12h | yolo11n.pt |

Nota: Si los archivos locales tienen nombre "yolov11*.pt", se normalizan automáticamente a "yolo11*.pt".

---

## Base de Datos (opcional)

- La persistencia es opcional — el sistema funciona sin base de datos
- Usa variables de entorno si están disponibles: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- Frecuencia: un registro por minuto con velocidad promedio y conteos por tipo
- Esquema: 3 tablas — `traffic_data` (legado), `telemetry_raw` (telemetría con señales de calidad + proveniencia), `traffic_classifications` (predicciones + gate de accidentes + metadata de validación)

---

## Estados del Tráfico

El clasificador produce uno de 4 estados por minuto:

| Estado | Código | Descripción |
|---|---|---|
| Normal | 0 | Flujo libre, velocidades típicas 40-80 km/h |
| Reducido | 1 | Flujo más lento, volumen moderado |
| Congestionado | 2 | Flujo muy lento, alto volumen |
| Accidente | 3 | Velocidades casi nulas con desaceleración abrupta |

---

## Dependencias

Instalar todas las dependencias:

```bash
pip install -e ".[all]"
# O la forma clásica:
pip install -r requirements.txt
```

En Google Colab, las dependencias se instalan automáticamente en la primera celda.

---

## Resolución de Problemas

| Problema | Solución |
|---|---|
| Faltan pesos de YOLO | Se descargan automáticamente |
| Errores de memoria en videos largos | Frame skipping y memory cleanup están integrados |
| Errores de conexión a BD | Verificar variables de entorno o ejecutar sin BD |
| Velocidades mostrando 0 para estacionarios | Comportamiento esperado y correcto |
| GPU no disponible en Colab | El sistema recurre a CPU (más lento) |
| Sesión se desconecta en Colab | Descargar video procesado antes de cerrar |

---

## Salidas

- **Módulo 2 Celda 2**: Procesamiento solo de telemetría para validación rápida y flujos CSV/DataFrame
- **Módulo 2 Celda 2b**: Video anotado con bounding boxes, tipo + ID, velocidad, y HUD
- **Módulo 2 Celda 3**: Clasificación del estado del tráfico (Normal/Reducido/Congestionado/Accidente)
- **Módulo 2 Celda 6**: Dashboard de visualización con gráficos y métricas

---

## Archivo Histórico

`archive/00_bootstrap/01_legacy_collection.ipynb` se conserva solo como contexto histórico de cómo se obtuvo la telemetría inicial. No forma parte del flujo de trabajo académico activo.

---

## Limitaciones Conocidas

- **Velocidad sin ground truth**: La precisión depende de la calibración manual de `pixels_per_meter`
- **Tracking sin re-identificación**: Si un vehículo es ocluido >1 segundo, pierde su ID
- **Colab efímero**: Los archivos se pierden al cerrar la sesión — descargar antes de cerrar
- **GPU no garantizada**: En horas pico de Colab Free, puede asignarse solo CPU (~10x más lento)
- **Auto-etiquetado no es ground truth**: Las etiquetas son proxies de ingeniería, no validadas por humanos
- **Sin smoke test automatizado en Colab**: Los notebooks compilan y pasan tests de paridad, pero la ejecución end-to-end se valida manualmente

Para un análisis completo, ver [BIAS_AND_LIMITATIONS.md](BIAS_AND_LIMITATIONS.md).

---

## Documentación Relacionada

- [README.md](../README.md) — Visión general y requisitos
- [SAD.md](SAD.md) — Diseño técnico detallado
- [KPIs/KPIs.md](KPIs/KPIs.md) — Métricas y guía de validación
- [DATA_LINEAGE.md](DATA_LINEAGE.md) — Linaje de datos
- [docs/adr/](adr/) — Decisiones arquitectónicas

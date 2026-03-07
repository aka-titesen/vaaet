<!-- context: VAAET/Docs/GUIA_USUARIO.md — Guía de usuario para VAAET.
Complementa README.md (visión general) y DDS.md (diseño técnico). -->

# 📖 GUÍA DE USUARIO - VAAET (actualizada)

## 🚦 ¿Qué es VAAET?

Un notebook para analizar tránsito del Puente General Manuel Belgrano con YOLO 11, Optical Flow y CNN. Procesa un video y genera un video anotado; opcionalmente guarda métricas por minuto en PostgreSQL.

---

## ⚡ Inicio rápido

1. Abre `vaaet.ipynb` (Colab o local).
2. Ejecuta las celdas en orden. Hay una celda de autodiagnóstico que verifica Ultralytics y descarga pesos yolo11 si faltan.
3. Sube tu video con nombre: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4`.
4. El sistema selecciona el modelo YOLO 11 por duración y prepara todo.
5. En la celda final, corre `process_bridge_video()` y espera la descarga del video procesado.

---

## 📁 Requisitos del video

- Formato: MP4.
- Nombre: `bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4` (estricto). Si no cumple, el sistema lo indicará y no continuará.

---

## 🧠 Selección automática de modelo (YOLO 11)

- ≤1h: yolo11x.pt
- 1–3h: yolo11l.pt
- 3–6h: yolo11m.pt
- 6–12h: yolo11s.pt
- > 12h: yolo11n.pt

Nota: Si tus archivos locales se llaman “yolov11*.pt”, se normalizan a “yolo11*.pt”.

---

## 🗄️ Base de datos (opcional)

- Se pregunta una sola vez si quieres persistir.
- Usa env vars si existen: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.
- Frecuencia: un registro por minuto con velocidad promedio y conteos por tipo.
- Esquema: tabla `traffic_data` con UNIQUE(clip_id, record_time).

No se generan JSON/CSV locales.

---

## 🔧 Dependencias

- ultralytics, opencv-python, numpy, scikit-learn, scipy, psycopg2-binary.
- Colab: se instalan automáticamente; local: instala con pip si hace falta.

---

## 🧭 Flujo de ejecución (resumen)

- Autodiagnóstico y dependencias.
- Carga y validación del video; selección YOLO 11.
- Inicialización del motor híbrido.
- (Opcional) Celda de mejoras: seguimiento y Farneback.
- Celda final: `process_bridge_video()`.

---

## 🧰 Consejos y solución de problemas

- Modelos: si faltan pesos, el autodiagnóstico intenta descargarlos.
- Memoria: para videos largos, el sistema aplica frame skipping y limpieza.
- Errores BD: verifica variables de entorno o ejecuta sin BD.
- Velocidades 0 en estacionados: esperado y correcto.

---

## 📦 Salida

- Video MP4 con cajas verdes, texto con borde (sin caja negra), tipo + ID y velocidad (si es coherente). Hub informativo minimalista.
- Descarga automática al finalizar en Colab.

---

## 🎬 Demos Sintéticas

Si no tienes un video real del puente, puedes generar videos de demostración:

1. Ejecuta **Celda 8** (Generador de Videos Sintéticos) — define los escenarios disponibles
2. Ejecuta **Celda 9** (Ejecutor de Demos) — genera 4 videos de demostración automáticamente

### Escenarios disponibles

| Escenario | Descripción |
|---|---|
| `light` | Tráfico ligero, pocos vehículos |
| `normal` | Flujo de tráfico normal |
| `busy` | Tráfico denso |
| `mixed` | Combinación de condiciones |
| `stationary_test` | Vehículos detenidos (valida `is_stationary()`) |

Los videos generados se descargan automáticamente en Colab.

---

## ⚠️ Limitaciones Conocidas

- **Velocidad sin ground truth**: La precisión del cálculo de velocidad depende de la calibración manual de `pixels_per_meter`
- **Tracking sin re-ID**: Si un vehículo es ocluido >1 segundo, pierde su ID
- **Colab efímero**: Los archivos se pierden al cerrar la sesión — descarga el video procesado antes de cerrar
- **GPU no garantizada**: En horas pico de Colab Free, puede asignarse solo CPU (procesamiento ~10x más lento)

Para un análisis completo, consultar [BIAS_AND_LIMITATIONS.md](BIAS_AND_LIMITATIONS.md).

---

## 🔗 Documentación relacionada

- [README.md](../README.md) — Visión general y requisitos
- [DDS.md](DDS.md) — Diseño técnico detallado
- [KPIs.md](KPIs/KPIs.md) — Métricas y guía de validación
- [DATA_LINEAGE.md](DATA_LINEAGE.md) — Linaje de datos
- [ADRs](adr/) — Decisiones arquitectónicas

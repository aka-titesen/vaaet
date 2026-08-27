# Flujo de procesamiento de VAAET

```mermaid
flowchart TD
    V["Video MP4"] --> A["vaaet.vision.analysis.analyze_video"]
    A --> D["YOLO detection"]
    D --> T["SORT tracking"]
    T --> S["Speed and motion quality"]
    S --> M["Minute telemetry"]
    S --> H["Annotated video and HUD"]
    M --> C{"Workflow"}
    C -->|"Data collection"| CSV["Cumulative raw CSV"]
    C -->|"Data collection, opt-in"| RAW["vaaet_raw.traffic_data"]
    C -->|"Inference + bundle"| F["19 engineered features"]
    F --> P["Traffic state + confidence"]
    P --> DB["Optional feedback tables"]
```

Sin proveedor de clasificación, el HUD indica **Telemetry Collection**. Con un
proveedor validado, muestra el estado actual sin acoplar visión con TensorFlow.

La implementación interna aplica estos pasos como Pipe-and-Filter síncrono y
ordenado: `FramePacket → PerceptionPacket → TrackingPacket → MotionPacket →
RenderedFramePacket`. No usa Producer–Consumer, threads ni colas. Si una
medición futura detecta que la preparación de frames limita a un único YOLO GPU,
la única frontera candidata será lectura/preparación ordenada → cola local
acotada → detección, seguida de una restauración estricta del orden antes de
tracking, velocidad, telemetría o render.

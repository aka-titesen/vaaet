<!-- context: VAAET/Docs/diagrams/model-selection.md — Lógica de selección de modelo YOLO.
Referenciado por DDS.md, ADR-002, PRD.md. -->

# Selección Automática de Modelo YOLO 11

Diagrama de decisión de `select_optimal_model()` en Cell 4.

```mermaid
flowchart TD
    A[Video cargado<br/>Duración extraída del nombre] --> B{duration_hours}
    
    B -->|< 1 hora| C[yolo11x.pt<br/>📊 Máxima precisión<br/>~140MB, ~30 FPS en T4]
    B -->|1 - 3 horas| D[yolo11l.pt<br/>📊 Alta precisión<br/>~100MB, ~45 FPS en T4]
    B -->|3 - 6 horas| E[yolo11m.pt<br/>⚖️ Balance<br/>~40MB, ~65 FPS en T4]
    B -->|6 - 12 horas| F[yolo11s.pt<br/>⚡ Alta velocidad<br/>~20MB, ~90 FPS en T4]
    B -->|> 12 horas| G[yolo11n.pt<br/>⚡ Mínimo consumo<br/>~6MB, ~120 FPS en T4]
    
    C & D & E & F & G --> H{¿Modelo disponible<br/>localmente?}
    
    H -->|Sí| I[Cargar modelo local]
    H -->|No| J[Descargar de<br/>Ultralytics Hub]
    
    J --> I
    I --> K[✅ Modelo listo<br/>para inferencia]

    style C fill:#4CAF50,color:#fff
    style D fill:#8BC34A,color:#fff
    style E fill:#FFC107,color:#000
    style F fill:#FF9800,color:#fff
    style G fill:#F44336,color:#fff
```

## Lógica (Cell 4)

```python
def select_optimal_model(duration_hours):
    if duration_hours < 1:     return "yolo11x.pt"
    elif duration_hours <= 3:  return "yolo11l.pt"
    elif duration_hours <= 6:  return "yolo11m.pt"
    elif duration_hours <= 12: return "yolo11s.pt"
    else:                      return "yolo11n.pt"
```

## Consideraciones

- La duración se extrae del **nombre del archivo**, no de los metadatos del video
- Si el nombre del modelo contiene "yolov11" (con 'v'), se normaliza automáticamente a "yolo11"
- FPS estimados en GPU T4 (Colab Free) — varían según resolución de video y carga del servidor

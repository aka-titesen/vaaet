<!-- context: VAAET/docs/diagrams/model-selection.md — YOLO model selection logic.
Referenced by DDS.md, ADR-002, PRD.md. -->

# Automatic YOLO 11 Model Selection

Decision diagram for `select_optimal_model()` in `src/perception/detector.py`.

```mermaid
flowchart TD
    A[Video loaded<br/>Duration extracted from filename] --> B{duration_hours}
    
    B -->|< 1 hour| C[yolo11x.pt<br/>Maximum accuracy<br/>~140MB, ~30 FPS on T4]
    B -->|1 - 3 hours| D[yolo11l.pt<br/>High accuracy<br/>~100MB, ~45 FPS on T4]
    B -->|3 - 6 hours| E[yolo11m.pt<br/>Balanced<br/>~40MB, ~65 FPS on T4]
    B -->|6 - 12 hours| F[yolo11s.pt<br/>High speed<br/>~20MB, ~90 FPS on T4]
    B -->|> 12 hours| G[yolo11n.pt<br/>Minimum resource usage<br/>~6MB, ~120 FPS on T4]
    
    C & D & E & F & G --> H{Model available<br/>locally?}
    
    H -->|Yes| I[Load local model]
    H -->|No| J[Download from<br/>Ultralytics Hub]
    
    J --> I
    I --> K[Model ready<br/>for inference]

    style C fill:#4CAF50,color:#fff
    style D fill:#8BC34A,color:#fff
    style E fill:#FFC107,color:#000
    style F fill:#FF9800,color:#fff
    style G fill:#F44336,color:#fff
```

## Selection Logic

```python
def select_optimal_model(duration_hours: float) -> str:
    if duration_hours < 1:     return "yolo11x.pt"
    elif duration_hours <= 3:  return "yolo11l.pt"
    elif duration_hours <= 6:  return "yolo11m.pt"
    elif duration_hours <= 12: return "yolo11s.pt"
    else:                      return "yolo11n.pt"
```

## Notes

- Duration is extracted from the **filename**, not from video metadata
- If the model name contains "yolov11" (with 'v'), it is automatically normalized to "yolo11"
- Estimated FPS on T4 GPU (Colab Free) — varies by video resolution and server load

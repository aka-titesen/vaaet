<!-- context: VAAET/Docs/diagrams/pipeline-flow.md — Diagrama principal del pipeline de procesamiento.
Referenciado por DDS.md §1, AGENTS.md, llms-full.txt. -->

# Pipeline de Procesamiento VAAET

Flujo completo de datos desde la ingesta del video hasta la salida anotada y persistencia.

```mermaid
flowchart TD
    A[📹 Video Input<br/>bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4] --> B[Validar nombre<br/>y extraer duración]
    B --> C{Duración}
    C -->|< 1h| D1[yolo11x.pt]
    C -->|1-3h| D2[yolo11l.pt]
    C -->|3-6h| D3[yolo11m.pt]
    C -->|6-12h| D4[yolo11s.pt]
    C -->|> 12h| D5[yolo11n.pt]
    
    D1 & D2 & D3 & D4 & D5 --> E[Cargar modelo YOLO 11]
    
    E --> F[Inicializar VAAETHybrid]
    F --> G[Leer frame del video]
    
    G --> H[Detectar vehículos<br/>YOLO inference + NMS]
    H --> I[Matching de tracks<br/>SORT por distancia euclidiana]
    I --> J[Calcular velocidad<br/>Fusión híbrida 70/30]
    J --> K{¿Estacionario?<br/>AND-conjunction 6 criterios}
    
    K -->|Sí| L[Marcar como estacionado<br/>Velocidad = 0]
    K -->|No| M[Registrar velocidad<br/>en historial del track]
    
    L & M --> N[Dibujar anotaciones<br/>Bounding boxes + HUD]
    N --> O[Escribir frame anotado<br/>al video de salida]
    
    O --> P{¿Cada 60s?}
    P -->|Sí| Q[Persistir en PostgreSQL<br/>AWS RDS - opcional]
    P -->|No| R{¿Más frames?}
    Q --> R
    
    R -->|Sí| G
    R -->|No| S[Finalizar video]
    S --> T[📥 Descargar video anotado<br/>Auto-download en Colab]
```

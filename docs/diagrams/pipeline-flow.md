<!-- context: VAAET/docs/diagrams/pipeline-flow.md — Main processing pipeline diagram.
Referenced by SAD.md, AGENTS.md, llms-full.txt. -->

# VAAET Processing Pipeline

Complete data flow from video ingestion to annotated output and persistence.

```mermaid
flowchart TD
    A[Video Input<br/>bridge_YYYY-MM-DD_HH-MM-SS_to_HH-MM-SS.mp4] --> B[Validate filename<br/>and extract duration]
    B --> C{Duration}
    C -->|< 1h| D1[yolo11x.pt]
    C -->|1-3h| D2[yolo11l.pt]
    C -->|3-6h| D3[yolo11m.pt]
    C -->|6-12h| D4[yolo11s.pt]
    C -->|> 12h| D5[yolo11n.pt]
    
    D1 & D2 & D3 & D4 & D5 --> E[Load YOLO 11 model]
    
    E --> F[Initialize perception pipeline]
    F --> G[Read video frame]
    
    G --> H[Detect vehicles<br/>YOLO inference + NMS]
    H --> I[Match tracks<br/>SORT by Euclidean distance]
    I --> J[Estimate speed<br/>Hybrid fusion 70/30]
    J --> K{Stationary?<br/>AND-conjunction 6 criteria}
    
    K -->|Yes| L[Mark as stationary<br/>Speed = 0]
    K -->|No| M[Record speed<br/>in track history]
    
    L & M --> N[Draw annotations<br/>Bounding boxes + HUD]
    N --> O[Write annotated frame<br/>to output video]
    
    O --> P{Every 60s?}
    P -->|Yes| Q[Persist to PostgreSQL<br/>AWS RDS - optional]
    P -->|No| R{More frames?}
    Q --> R
    
    R -->|Yes| G
    R -->|No| S[Finalize video]
    S --> T[Download annotated video<br/>Auto-download in Colab]
```

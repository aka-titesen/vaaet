<!-- context: VAAET/docs/diagrams/multi-camera-layout.md — Multi-camera layout detection.
Referenced by SAD.md, PRD.md. -->

# Multi-Camera Layout Detection

The system auto-detects whether the video contains 1, 2, or 4 camera views and processes each ROI independently.

```mermaid
flowchart TD
    A[Video Frame] --> B{Analyze layout}
    
    B -->|1 view| C[Single Camera]
    B -->|2 views| D[Dual Camera]
    B -->|4 views| E[Quad Camera]
    
    C --> F[ROI: Full frame<br/>0,0 -> W,H]
    
    D --> G[ROI 1: Left half<br/>0,0 -> W/2,H]
    D --> H[ROI 2: Right half<br/>W/2,0 -> W,H]
    
    E --> I[ROI 1: Top-left<br/>0,0 -> W/2,H/2]
    E --> J[ROI 2: Top-right<br/>W/2,0 -> W,H/2]
    E --> K[ROI 3: Bottom-left<br/>0,H/2 -> W/2,H]
    E --> L[ROI 4: Bottom-right<br/>W/2,H/2 -> W,H]
    
    F & G & H & I & J & K & L --> M[VAAET Pipeline<br/>per each ROI]
    M --> N[Merge annotations<br/>in final frame]
```

## Visual Layouts

```mermaid
block-beta
    columns 3
    
    block:single:1
        columns 1
        s1["1 Camera"]
        s2["+---------+"]
        s3["|  Full   |"]
        s4["|  Frame  |"]
        s5["+---------+"]
    end
    
    block:dual:1
        columns 1
        d1["2 Cameras"]
        d2["+----+----+"]
        d3["| L  | R  |"]
        d4["|    |    |"]
        d5["+----+----+"]
    end
    
    block:quad:1
        columns 1
        q1["4 Cameras"]
        q2["+----+----+"]
        q3["| TL | TR |"]
        q4["+----+----+"]
        q5["| BL | BR |"]
    end
```

## Notes

- Layout detection is based on edge analysis and black areas between views
- Each ROI is processed independently with its own tracking instance
- Perspectives can vary between cameras — homographies are applied per ROI
- Aggregated metrics (average speed, counts) combine data from all ROIs

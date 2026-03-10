<!-- context: VAAET/docs/diagrams/speed-calculation.md — Detailed speed calculation flow.
Referenced by DDS.md, ADR-004. -->

# Hybrid Speed Calculation Flow

Detail of the `estimate_speed()` pipeline in `src/perception/speed.py`.

```mermaid
flowchart TD
    A[Track Data<br/>deque of last 30 frame centroids] --> B[Compute global motion<br/>Lucas-Kanade Optical Flow]
    
    B --> C[Compensate camera motion<br/>Subtract median global vector]
    
    C --> D[Compute Euclidean displacement<br/>in pixel space]
    
    D --> E[Get perspective factor<br/>by vehicle Y coordinate]
    
    E --> F{Perspective zone}
    F -->|Y > 0.7 - Near| G1[Factor: 1.8x]
    F -->|0.3-0.7 - Mid| G2[Factor: 1.0x]
    F -->|Y < 0.3 - Far| G3[Factor: 0.6x]
    
    G1 & G2 & G3 --> H[Convert pixels to meters<br/>pixels_per_meter x factor]
    
    H --> I[Physics Speed<br/>distance_meters / dt -> km/h]
    
    A --> J[Extract 10 features<br/>displacement, std_x, std_y,<br/>len, perspective, y_ratio,<br/>mean_dx, mean_dy, max_dx, time]
    
    J --> K[MLPRegressor predict<br/>hidden_layers: 64, 32]
    
    K --> L{MLP output in 5-100?}
    L -->|Yes| M[Valid MLP prediction]
    L -->|No| N[Discard MLP prediction<br/>Use physics only]
    
    I & M --> O[Fusion: 0.7 x Physics + 0.3 x MLP]
    I & N --> O2[Use 100% Physics]
    
    O & O2 --> P{Speed in 2-120 km/h?}
    P -->|Yes| Q[Speed accepted]
    P -->|No| R[Discarded as implausible]
    
    Q --> S[Temporal smoothing<br/>Weighted moving average<br/>0.6 / 0.3 / 0.1 last 3s]
```

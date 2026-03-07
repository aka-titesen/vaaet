<!-- context: VAAET/Docs/diagrams/speed-calculation.md — Flujo detallado de cálculo de velocidad.
Referenciado por DDS.md §2.2, ADR-004. -->

# Flujo de Cálculo de Velocidad Híbrida

Detalle del pipeline `calculate_enhanced_speed()` en `VAAETHybrid`.

```mermaid
flowchart TD
    A[Track Data<br/>deque de centroides últimos 30 frames] --> B[Calcular movimiento global<br/>Lucas-Kanade Optical Flow]
    
    B --> C[Compensar movimiento de cámara<br/>Restar vector mediano global]
    
    C --> D[Calcular desplazamiento euclidiano<br/>en espacio de píxeles]
    
    D --> E[Obtener factor de perspectiva<br/>por coordenada Y del vehículo]
    
    E --> F{Zona de perspectiva}
    F -->|Y > 0.7 - Near| G1[Factor: 1.8x]
    F -->|0.3-0.7 - Mid| G2[Factor: 1.0x]
    F -->|Y < 0.3 - Far| G3[Factor: 0.6x]
    
    G1 & G2 & G3 --> H[Convertir píxeles → metros<br/>pixels_per_meter × factor]
    
    H --> I[Velocidad Física<br/>distancia_metros / Δt → km/h]
    
    A --> J[Extraer 10 features<br/>displacement, std_x, std_y,<br/>len, perspective, y_ratio,<br/>mean_dx, mean_dy, max_dx, time]
    
    J --> K[MLPRegressor predict<br/>hidden_layers: 64, 32]
    
    K --> L{MLP output ∈ [5, 100]?}
    L -->|Sí| M[Predicción MLP válida]
    L -->|No| N[Descartar predicción MLP<br/>Usar solo física]
    
    I & M --> O[Fusión: 0.7 × Física + 0.3 × MLP]
    I & N --> O2[Usar 100% Física]
    
    O & O2 --> P{Velocidad ∈ [2, 120] km/h?}
    P -->|Sí| Q[✅ Velocidad final aceptada]
    P -->|No| R[❌ Descartada por implausibilidad]
    
    Q --> S[Suavizado temporal<br/>Media móvil ponderada<br/>0.6 / 0.3 / 0.1 últimos 3s]
```

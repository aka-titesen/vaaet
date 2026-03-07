<!-- context: VAAET/Docs/diagrams/multi-camera-layout.md — Detección de layouts multi-cámara.
Referenciado por DDS.md, PRD.md requisito 5. -->

# Detección de Layout Multi-Cámara

El sistema auto-detecta si el video contiene 1, 2 o 4 vistas de cámara y procesa cada ROI independientemente.

```mermaid
flowchart TD
    A[Frame del Video] --> B{Analizar layout}
    
    B -->|1 vista| C[Single Camera]
    B -->|2 vistas| D[Dual Camera]
    B -->|4 vistas| E[Quad Camera]
    
    C --> F[ROI: Frame completo<br/>0,0 → W,H]
    
    D --> G[ROI 1: Mitad izquierda<br/>0,0 → W/2,H]
    D --> H[ROI 2: Mitad derecha<br/>W/2,0 → W,H]
    
    E --> I[ROI 1: Superior izq<br/>0,0 → W/2,H/2]
    E --> J[ROI 2: Superior der<br/>W/2,0 → W,H/2]
    E --> K[ROI 3: Inferior izq<br/>0,H/2 → W/2,H]
    E --> L[ROI 4: Inferior der<br/>W/2,H/2 → W,H]
    
    F & G & H & I & J & K & L --> M[Pipeline VAAET<br/>por cada ROI]
    M --> N[Merge anotaciones<br/>en frame final]
```

## Layouts Visuales

```mermaid
block-beta
    columns 3
    
    block:single:1
        columns 1
        s1["1 Cámara"]
        s2["┌─────────┐"]
        s3["│  Full   │"]
        s4["│  Frame  │"]
        s5["└─────────┘"]
    end
    
    block:dual:1
        columns 1
        d1["2 Cámaras"]
        d2["┌────┬────┐"]
        d3["│ L  │ R  │"]
        d4["│    │    │"]
        d5["└────┴────┘"]
    end
    
    block:quad:1
        columns 1
        q1["4 Cámaras"]
        q2["┌────┬────┐"]
        q3["│ TL │ TR │"]
        q4["├────┼────┤"]
        q5["│ BL │ BR │"]
    end
```

## Notas

- La detección de layout se basa en análisis de bordes y áreas negras entre vistas
- Cada ROI se procesa independientemente con su propia instancia de tracking
- Las perspectivas pueden variar entre cámaras — las homografías se aplican por ROI
- Las métricas agregadas (velocidad promedio, conteos) combinan datos de todas las ROIs

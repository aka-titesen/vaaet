# Selección adaptativa de YOLO 11

La duración proviene del nombre estándar o, para nombres libres, de metadata OpenCV. Los umbrales se definen únicamente en `vaaet.settings.YOLO_MODEL_VARIANTS`.

```mermaid
flowchart TD
    A["Video"] --> B["Extract duration"]
    B --> C{"Duration"}
    C -->|"≤ 5 min"| X["yolo11x"]
    C -->|"≤ 30 min"| L["yolo11l"]
    C -->|"≤ 3 h"| M["yolo11m"]
    C -->|"≤ 12 h"| S["yolo11s"]
    C -->|"> 12 h"| N["yolo11n"]
    X & L & M & S & N --> D["Download on first use"]
    D --> I["Vehicle detection"]
```

No se versionan pesos `.pt` y no se prometen FPS concretos: dependen de GPU, resolución y carga del runtime.

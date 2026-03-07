<!-- context: VAAET/Docs/diagrams/colab-aws-architecture.md — Diagrama C4 de contenedores.
Referenciado por DDS.md, AGENTS.md, ADR-005, ADR-007. -->

# Arquitectura Colab + AWS RDS

Diagrama de contenedores mostrando la infraestructura de ejecución.

```mermaid
C4Context
    title VAAET - Arquitectura de Contenedores

    Person(user, "Usuario", "Ingeniero de tráfico / Investigador")

    System_Boundary(colab, "Google Colab") {
        Container(notebook, "vaaet.ipynb", "Jupyter Notebook", "Pipeline completo de análisis")
        Container(gpu, "GPU Runtime", "NVIDIA T4 / V100", "Inferencia YOLO 11")
        Container(storage, "Colab Storage", "Efímero", "Video input/output temporal")
    }

    System_Ext(rds, "AWS RDS", "PostgreSQL - Persistencia durable")
    System_Ext(yolo_hub, "Ultralytics Hub", "Descarga de modelos YOLO 11")

    Rel(user, notebook, "Ejecuta celdas, sube video", "Browser")
    Rel(notebook, gpu, "Inferencia YOLO", "CUDA")
    Rel(notebook, storage, "Lee video input, escribe video output")
    Rel(notebook, rds, "INSERT cada 60s", "psycopg2 / TCP:5432")
    Rel(notebook, yolo_hub, "Descarga .pt en primera ejecución", "HTTPS")
    Rel(storage, user, "Auto-download video procesado", "Colab API")
```

## Flujo de Datos

```mermaid
flowchart LR
    A[👤 Usuario] -->|Sube .mp4| B[Google Colab]
    B -->|Inferencia| C[GPU T4/V100]
    C -->|Detecciones| B
    B -->|INSERT por minuto| D[(AWS RDS<br/>PostgreSQL)]
    B -->|Video anotado| A
    E[Ultralytics Hub] -->|modelo .pt| B
```

## Notas de Seguridad

- Las credenciales de RDS se obtienen via **variables de entorno** (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) o **`getpass`** interactivo
- Nunca se imprimen credenciales en outputs de celdas
- La conexión a RDS usa el puerto estándar 5432 sin SSL (pendiente como mejora)
- El almacenamiento de Colab es efímero — los videos se pierden al cerrar la sesión

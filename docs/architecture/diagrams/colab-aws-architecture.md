<!-- context: VAAET/docs/diagrams/colab-aws-architecture.md — C4 container diagram.
Referenced by SAD.md, AGENTS.md, ADR-005, ADR-007. -->

# Colab + AWS RDS Architecture

Container diagram showing the runtime infrastructure.

```mermaid
C4Context
    title VAAET - Container Architecture

    Person(user, "User", "Traffic engineer / Researcher")

    System_Boundary(colab, "Google Colab") {
        Container(notebook, "Notebooks", "Jupyter Notebooks", "Module 1 and Module 2 pipelines")
        Container(gpu, "GPU Runtime", "NVIDIA T4 / V100", "YOLO 11 inference")
        Container(storage, "Colab Storage", "Ephemeral", "Temporary video input/output")
    }

    System_Ext(rds, "AWS RDS", "PostgreSQL - Durable persistence")
    System_Ext(yolo_hub, "Ultralytics Hub", "YOLO 11 model download")

    Rel(user, notebook, "Runs cells, uploads video", "Browser")
    Rel(notebook, gpu, "YOLO inference", "CUDA")
    Rel(notebook, storage, "Reads video input, writes video output")
    Rel(notebook, rds, "INSERT every 60s", "psycopg2 / TCP:5432")
    Rel(notebook, yolo_hub, "Downloads .pt on first run", "HTTPS")
    Rel(storage, user, "Auto-download processed video", "Colab API")
```

## Data Flow

```mermaid
flowchart LR
    A[User] -->|Uploads .mp4| B[Google Colab]
    B -->|Inference| C[GPU T4/V100]
    C -->|Detections| B
    B -->|INSERT per minute| D[(AWS RDS<br/>PostgreSQL)]
    B -->|Annotated video| A
    E[Ultralytics Hub] -->|.pt model| B
```

## Security Notes

- RDS credentials are obtained via **environment variables** (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) or interactive **`getpass`**
- Credentials are never printed in cell outputs
- RDS connection uses standard port 5432 without SSL (pending improvement)
- Colab storage is ephemeral — videos are lost when the session ends

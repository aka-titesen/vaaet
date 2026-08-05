# Arquitectura Colab y PostgreSQL

```mermaid
flowchart LR
    U["Usuario"] --> N["Three Colab notebooks"]
    G["GitHub repository"] --> N
    N --> GPU["Managed GPU runtime"]
    Y["Ultralytics"] -->|"YOLO weights at runtime"| N
    N <-->|"Complete model bundle"| D["Google Drive"]
    N -->|"Profile-specific SQLAlchemy access"| P[("PostgreSQL 14+")]
    N -->|"Download"| U
```

Las credenciales por perfil se leen directamente de Colab Secrets. `vaaet_raw`,
`vaaet_ml` y `vaaet_feedback` separan responsabilidades; Alembic y el administrador
nunca se ejecutan desde Colab. `/content` es efímero y la persistencia permanece
deshabilitada por defecto.

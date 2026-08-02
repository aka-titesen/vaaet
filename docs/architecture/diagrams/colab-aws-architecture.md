# Arquitectura Colab y PostgreSQL

```mermaid
flowchart LR
    U["Usuario"] --> N["Three Colab notebooks"]
    G["GitHub repository"] --> N
    N --> GPU["Managed GPU runtime"]
    Y["Ultralytics"] -->|"YOLO weights at runtime"| N
    N <-->|"Complete model bundle"| D["Google Drive"]
    N -->|"Opt-in SQLAlchemy writes"| P[("PostgreSQL")]
    N -->|"Download"| U
```

Las credenciales se leen de Colab Secrets con variables de entorno como fallback. `/content` es efímero; los outputs deben descargarse o copiarse. La persistencia está deshabilitada por defecto.

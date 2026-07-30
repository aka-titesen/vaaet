        <!-- context: VAAET/docs/diagrams/speed-calculation.md -- Detailed speed calculation flow.
        Referenced by DDS.md as the active physics-first speed path. -->

        # Physics-First Speed Calculation Flow

        Detail of the active `estimate_speed()` + `process_clip_telemetry()` pipeline in
        `src/perception/speed.py` and `src/perception/pipeline.py`.

        ```mermaid
        flowchart TD
            A[Track history<br/>recent centroid deque] --> B[Estimate global motion<br/>Optical Flow]
            B --> C[Compensate camera motion<br/>subtract global vector]
            C --> D[Compute per-frame displacement norms]
            D --> E[Apply noise floor<br/>clip abrupt anomalies]
            E --> F[Perspective correction<br/>zone-based factor by Y position]
            F --> G[Convert pixels to meters<br/>pixels_per_meter]
            G --> H[Physics speed<br/>distance / dt -> km/h]
            H --> I{Plausible speed<br/>for vehicle type?}
            I -->|No| J[Reject sample]
            I -->|Yes| K{Reliable track?
flow ratio, gap recovery,
recent anomaly checks}
            K -->|No| L[Exclude from minute summary]
            K -->|Yes| M[Track-level smoothing
short moving average]

            A --> N[Motion state analysis
near_zero vs stationary]
            N --> O[Hysteresis-based stationary confirmation]

            M --> P[Minute accumulator]
            O --> P
            J --> P
            L --> P

            P --> Q[Robust minute summary
trimmed / outlier-aware mean]
            Q --> R[Quality signals
rejected, recovered, stationary, near-zero]
            R --> S[Classifier-ready telemetry row]

            T[Optional dormant path<br/>MLP speed fusion] -.only if explicitly wired.-> M
        ```

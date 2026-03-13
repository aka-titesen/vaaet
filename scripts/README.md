# Scripts

This directory contains manual utilities that support the academic workflow
but are not part of the active notebook runtime.

- `convert_backup.py`: one-time local conversion from PostgreSQL `.backup`
  to CSV so Module 1 can run in Google Colab without `pg_restore`.
- `evaluate_real_clips.py`: offline comparison utility for reviewing
  baseline vs candidate results on exported telemetry CSVs.

Temporary notebook patch/debug artifacts do not belong here and should
not be committed.

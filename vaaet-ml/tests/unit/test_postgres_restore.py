from __future__ import annotations

from pathlib import Path

from vaaet_ml.data.postgres_restore import resolve_pg_restore_for_backup


def test_csv_input_does_not_prepare_pg_restore(tmp_path: Path, monkeypatch) -> None:
    backup = tmp_path / "traffic_data.backup"
    backup.touch()
    csv = tmp_path / "traffic_data_raw.csv"
    csv.touch()
    monkeypatch.setattr("vaaet_ml.data.postgres_restore.shutil.which", lambda _name: None)

    assert resolve_pg_restore_for_backup(backup, csv, in_colab=True) is None

from __future__ import annotations

import pytest

from vaaet.inference.bundle import authorize_bundle


def test_candidate_bundle_cannot_persist() -> None:
    manifest = {"training_lifecycle": {"deployment_stage": "candidate", "input_policy": "canonical-v2"}}

    with pytest.raises(RuntimeError, match="sólo offline"):
        authorize_bundle(
            manifest,
            allow_pilot=True,
            allow_experimental=True,
            persist_to_database=True,
        )

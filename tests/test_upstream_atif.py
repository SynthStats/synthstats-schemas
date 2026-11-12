"""The schema package exposes Harbor's ATIF contract to callers."""

import pytest
from pydantic import ValidationError

import synthstats_schemas as schemas


def test_upstream_trajectory_accepts_agent_and_extension_fields():
    payload = {
        "schema_version": "ATIF-v1.7",
        "session_id": "conformance",
        "agent": {"name": "producer", "version": "1"},
        "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
        "extra": {"task_id": "task", "producer_field": [1, 2]},
    }
    trajectory = schemas.Trajectory.model_validate(payload)
    assert trajectory.agent.name == "producer"
    assert trajectory.steps[0].step_id == 1
    assert trajectory.extra == {"task_id": "task", "producer_field": [1, 2]}


def test_upstream_trajectory_rejects_nonsequential_steps():
    with pytest.raises(ValidationError, match="sequential"):
        schemas.Trajectory.model_validate({
            "agent": {"name": "producer", "version": "1"},
            "steps": [{"step_id": 2, "source": "agent", "message": "done"}],
        })

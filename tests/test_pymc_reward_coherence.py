"""A PyMC reward and its component must describe the same diagnostics."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synthstats_schemas import RewardArtifact

FIXTURE = Path(__file__).parent / "fixtures/retained_rewards/pymc_elpd_reward.json"
COMPILATION = Path(__file__).parent / "fixtures/harbor_template_reward_artifact.json"


def test_retained_pymc_reward_is_valid():
    RewardArtifact.model_validate(json.loads(FIXTURE.read_text()))


@pytest.mark.parametrize(
    ("field", "value"),
    [("semantic_status", "critical"), ("pareto_k_warning", 1)],
)
def test_pymc_component_diagnostics_cannot_contradict_reward(field, value):
    artifact = json.loads(FIXTURE.read_text())
    artifact["reward_components"][0]["diagnostics"][field] = value
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(artifact)


def test_retained_compilation_reward_is_valid():
    RewardArtifact.model_validate(json.loads(COMPILATION.read_text()))


@pytest.mark.parametrize(
    ("field", "value"), [("total_layers", 7), ("passed_layers", False)]
)
def test_compilation_component_metrics_cannot_contradict_reward(field, value):
    artifact = json.loads(COMPILATION.read_text())
    artifact["reward_components"][0]["metrics"][field] = value
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(artifact)

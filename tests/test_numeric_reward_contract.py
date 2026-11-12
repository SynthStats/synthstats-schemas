"""The numeric scorer's reward contract is enforced at its schema boundary."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synthstats_schemas import RewardArtifact

FIXTURES = Path(__file__).parent / "fixtures"
CONTEXT = {"expected_contract": "synthstats.harbor.numeric_admission.v1"}


@pytest.mark.parametrize("kind", ["score", "refusal"])
def test_retained_numeric_records_validate(kind):
    artifact = json.loads((FIXTURES / f"pymc_numeric_{kind}_train.json").read_text())
    RewardArtifact.model_validate(artifact, context=CONTEXT)


def test_score_cannot_pass_more_layers_than_exist():
    artifact = json.loads((FIXTURES / "pymc_numeric_score_train.json").read_text())
    artifact["metrics"].update(passed_layers=100, total_layers=1)
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(artifact, context=CONTEXT)


def test_numeric_score_cannot_disagree_with_reward():
    artifact = json.loads((FIXTURES / "pymc_numeric_score_train.json").read_text())
    artifact["reward"] = -123.0
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(artifact, context=CONTEXT)


@pytest.mark.parametrize(
    ("kind", "path", "value"),
    [
        ("score", ("provenance", "admission_contract"), None),
        ("score", ("provenance", "profile"), "other"),
        ("score", ("metrics", "baseline_score"), True),
        ("score", ("metrics", "holdout_elpd"), None),
        ("score", ("reward_transform", "authoritative"), False),
        ("score", ("reward_transform", "authoritative"), 1),
        ("score", ("reward_transform", "baseline_score"), 999.0),
        ("score", ("scalarization", "weights"), {"holdout_elpd": 2.0}),
        ("score", ("reward_components", 0, "weight"), 2.0),
        ("score", ("reward_components", 0, "log_contribution"), 99.0),
        ("refusal", ("reward",), 1.0),
        ("refusal", ("reward_transform", "authoritative"), False),
        ("refusal", ("reward_transform", "authoritative"), 1),
        ("refusal", ("reward_components", 0, "weight"), 2.0),
    ],
)
def test_numeric_records_reject_contradictory_envelope(kind, path, value):
    artifact = json.loads((FIXTURES / f"pymc_numeric_{kind}_train.json").read_text())
    parent = artifact
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = value
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(artifact, context=CONTEXT)


def test_refusal_gate_copies_cannot_contradict_each_other():
    artifact = json.loads((FIXTURES / "pymc_numeric_refusal_train.json").read_text())
    gate = {
        "name": "min_predictive_scale", "applicable": None, "assessed": False,
        "passed": None, "status": "unassessed", "data_std": None,
        "min_pred_std": None, "threshold": None, "detail": "density_gate_status=unassessed",
    }
    for diagnostics in (artifact["diagnostics"], artifact["reward_components"][0]["diagnostics"]):
        diagnostics.update(
            density_gates=[gate.copy()], density_gate_applicable=None,
            density_gate_assessed=False, density_gate_passed=None,
            density_gate_status="unassessed", density_gate_threshold=1.0,
        )
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(artifact, context=CONTEXT)


def test_retained_numeric_compilation_refusal_validates():
    artifact = json.loads((FIXTURES / "compilation_refusal_train.json").read_text())
    RewardArtifact.model_validate(artifact, context=CONTEXT)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("provenance", "verifier"), "other"),
        (("reward_transform", "authoritative"), 1),
        (("reward_transform", "pass_log_reward"), 1.0),
        (("reward",), 1.0),
        (("reward_components", 0, "weight"), 2.0),
        (("reward_components", 0, "diagnostics", "failure_reason"), "different"),
        (("scalarization", "weights"), {"compilation_cascade": 2.0}),
    ],
)
def test_compilation_refusal_rejects_contradictory_envelope(path, value):
    artifact = json.loads((FIXTURES / "compilation_refusal_train.json").read_text())
    parent = artifact
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = value
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(artifact, context=CONTEXT)

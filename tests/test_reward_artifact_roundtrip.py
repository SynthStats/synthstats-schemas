"""Round-trip law: a validated artifact dumps to a payload its validator accepts.

Validate an artifact, dump the validated instance, and validate the dump again.
"""

import json
from pathlib import Path

import pytest

from synthstats_schemas import RewardArtifact

FIXTURES = Path(__file__).parent / "fixtures"


def _recorded(name):
    return json.loads((FIXTURES / name).read_text())


def _policy_row():
    row = _recorded("compilation_refusal_train.json")
    row["kind"] = "policy_refusal"
    for diagnostics in (row["diagnostics"], row["reward_components"][0]["diagnostics"]):
        diagnostics.pop("failed_layer", None)
    row["metrics"] = {"cascade_reward": 0.0, "passed_layers": 5, "total_layers": 6}
    return row


def _rows():
    return [
        pytest.param(
            _recorded("compilation_refusal_train.json"), id="compilation_refusal"
        ),
        pytest.param(
            _recorded("pymc_numeric_refusal_train.json"), id="numeric_refusal"
        ),
        pytest.param(_recorded("pymc_numeric_score_train.json"), id="numeric_score"),
        pytest.param(_policy_row(), id="policy_refusal"),
    ]


@pytest.mark.parametrize("payload", _rows())
def test_a_validated_artifact_dumps_to_a_payload_its_validator_accepts(payload):
    artifact = RewardArtifact.model_validate(payload)
    for dumped in (
        artifact.model_dump(),
        artifact.model_dump(mode="json"),
        json.loads(artifact.model_dump_json()),
    ):
        assert RewardArtifact.model_validate(dumped) == artifact

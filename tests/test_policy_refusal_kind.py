"""A policy refusal is its own typed row kind.

A policy refusal names no failed cascade layer: the policy refused after
the verifier passed passed_layers of total_layers. Its row kind carries
those counts and no failed_layer; the compilation refusal keeps
failed_layer, restricted to the cascade's layer names. The attribution
table keys the policy refusal by its kind and reason code.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synthstats_schemas import RewardArtifact
from synthstats_schemas.verdicts import Attribution, attribution_for

FIXTURES = Path(__file__).parent / "fixtures"
CONTEXT = {"expected_contract": "synthstats.harbor.numeric_admission.v1"}
POLICY_REFUSAL_KIND = "policy_refusal"


def _compilation_row():
    return json.loads((FIXTURES / "compilation_refusal_train.json").read_text())


def _policy_row():
    row = _compilation_row()
    row["kind"] = POLICY_REFUSAL_KIND
    for diagnostics in (row["diagnostics"], row["reward_components"][0]["diagnostics"]):
        diagnostics.pop("failed_layer", None)
    row["metrics"] = {"cascade_reward": 0.0, "passed_layers": 5, "total_layers": 6}
    return row


def test_a_policy_refusal_validates_as_its_own_kind():
    RewardArtifact.model_validate(_policy_row(), context=CONTEXT)


def test_the_compilation_refusal_refuses_a_non_cascade_failed_layer():
    row = _compilation_row()
    for diagnostics in (row["diagnostics"], row["reward_components"][0]["diagnostics"]):
        diagnostics["failed_layer"] = POLICY_REFUSAL_KIND
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(row, context=CONTEXT)


def test_the_policy_refusal_is_attributed_by_kind_and_reason():
    assert attribution_for(None, POLICY_REFUSAL_KIND, POLICY_REFUSAL_KIND) is (
        Attribution.CANDIDATE
    )
    with pytest.raises(ValueError):
        attribution_for(None, POLICY_REFUSAL_KIND, "an_unlisted_reason_code")


def test_a_retained_compilation_refusal_without_a_kind_still_validates():
    RewardArtifact.model_validate(_compilation_row(), context=CONTEXT)


def test_a_policy_refusal_names_the_verifier_that_ships():
    # The factory transport writes policy refusals under the shipped verifier,
    # ppl_synth_data.harbor.pymc_verifier.
    row = _policy_row()
    for record in (row, row["reward_components"][0]):
        record["producer"] = "ppl_synth_data.harbor.pymc_verifier"
    RewardArtifact.model_validate(row, context=CONTEXT)

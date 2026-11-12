"""The scored branch gates on its source.

Both refused branches route by reward_source; the scored branch must do the
same, so a scored validation.compilation row (the producer's chokepoint gives
it no numeric contract) validates under the general RewardArtifact envelope
even when a consumer passes the expected-contract context.
"""
import pytest

from synthstats_schemas import RewardArtifact

CONTEXT = {"expected_contract": "synthstats.harbor.numeric_admission.v1"}

SCORED_COMPILATION_ROW = {
    "schema_version": "synthstats.harbor.reward.v1",
    "producer": "ppl_synth_data.harbor.validator_template",
    "reward_source": "validation.compilation",
    "evidence_kind": "verifier",
    "truth_owner": "verifier",
    "provenance": {"runtime": "pymc", "verifier": "harbor_pymc_l0_l5", "profile": "strict"},
    "reward": 0.0,
    "training_log_reward": 0.0,
    "diagnostics": {"elpd_status": "scored", "failed_layer": None,
                    "disposition": "scored", "training_target_status": "score_transform"},
    "metrics": {"cascade_reward": 0.0, "passed_layers": 6, "total_layers": 6},
    "reward_components": [{
        "name": "compilation_cascade", "kind": "terminal_verifier", "value": 0.0,
        "source": "validation.compilation", "evidence_kind": "verifier",
        "truth_owner": "verifier", "producer": "ppl_synth_data.harbor.validator_template",
        "weight": 1.0, "log_contribution": 0.0, "confidence": 1.0,
        "metrics": {"cascade_reward": 0.0, "passed_layers": 6, "total_layers": 6},
        "diagnostics": {"elpd_status": "scored", "disposition": "scored"},
        "provenance": {"runtime": "pymc", "verifier": "harbor_pymc_l0_l5", "profile": "strict"},
    }],
    "reward_transform": {"kind": "binary_log_reward", "pass_log_reward": 0.0,
                         "lower_log_reward_bound": -20.0, "authoritative": True},
    "scalarization": {"method": "weighted_sum", "component_names": ["compilation_cascade"],
                      "weights": {"compilation_cascade": 1.0},
                      "lower_log_reward_bound": -20.0},
    "components": {"compilation_cascade": 0.0},
}


def test_a_scored_compilation_row_validates_with_the_contract_context():
    RewardArtifact.model_validate(SCORED_COMPILATION_ROW, context=CONTEXT)

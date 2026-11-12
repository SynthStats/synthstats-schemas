from synthstats_schemas.reward import RewardComponent


def test_qualified_checks_are_a_verifier_owned_reward_component():
    component = RewardComponent.model_validate(
        {
            "name": "qualified_checks",
            "kind": "terminal_verifier",
            "value": 0.5,
            "source": "validation.qualified_checks",
            "evidence_kind": "verifier",
            "truth_owner": "verifier",
            "producer": "ppl_synth_data.reward.scalarization",
            "weight": 0.0,
            "log_contribution": 0.0,
        }
    )

    assert component.value == 0.5
    assert component.weight == 0.0

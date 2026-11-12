"""The stock PyMC verifier's reward rows have their own admission contract.

The stock verifier (ppl_synth_data.harbor.pymc_verifier) builds the program once,
fits it with pm.sample and scores the held-out rows with pm.set_data and
pm.compute_log_likelihood. It has no L0-L6 cascade, so its rows cannot claim
synthstats.harbor.numeric_admission.v1 (verifier harbor_pymc_l0_l6, passed_layers,
failed_layer). They declare synthstats.harbor.stock_admission.v1 instead, and the
copies inside one row must agree exactly as they do for the numeric contract.
"""

import copy
import math

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from synthstats_schemas import RewardArtifact

STOCK = "synthstats.harbor.stock_admission.v1"
CONTEXT = {"expected_contract": STOCK}
PROVENANCE = {"runtime": "pymc", "verifier": "pymc_verifier", "admission_contract": STOCK}


def _scored():
    # Illustrative numbers: held-out ELPD -7.0 against baseline -40 and gold -10,
    # so temperature = |gold - baseline| + 1e-8 and log reward = 33 / 30.
    baseline, gold, elpd = -40.0, -10.0, -7.0
    temperature = abs(gold - baseline) + 1e-8
    log_reward = round((elpd - baseline) / temperature, 6)
    diagnostics = {"disposition": "scored", "reason_code": None, "error": None,
                   "hidden_n_obs": 10, "training_n_obs": 20}
    return {
        "schema_version": "synthstats.harbor.reward.v1",
        "producer": "ppl_synth_data.harbor.pymc_verifier",
        "provenance": dict(PROVENANCE),
        "diagnostics": diagnostics,
        "reward": round(math.exp(log_reward), 6), "training_log_reward": log_reward,
        "reward_source": "validation.pymc_elpd", "evidence_kind": "verifier",
        "truth_owner": "verifier",
        "metrics": {"holdout_elpd": elpd, "avg_holdout_logp": elpd / 10,
                    "train_elpd_loo": -15.0, "pareto_k_max": 0.2, "baseline_score": baseline,
                    "gold_score": gold, "temperature": temperature, "hidden_n_obs": 10},
        "reward_transform": {"kind": "baseline_normalized_log_reward", "baseline_score": baseline,
                             "gold_score": gold, "temperature": temperature,
                             "lower_log_reward_bound": -20.0, "authoritative": True},
        "components": {"holdout_elpd": elpd},
        "reward_components": [{
            "name": "holdout_elpd", "kind": "terminal_verifier", "value": elpd,
            "source": "validation.pymc_elpd", "evidence_kind": "verifier",
            "truth_owner": "verifier", "producer": "ppl_synth_data.harbor.pymc_verifier",
            "weight": 1.0, "log_contribution": log_reward,
            "metrics": {"baseline_score": baseline, "temperature": temperature,
                        "train_elpd_loo": -15.0, "pareto_k_max": 0.2, "hidden_n_obs": 10},
            "diagnostics": diagnostics, "provenance": dict(PROVENANCE),
        }],
        "scalarization": {"method": "weighted_sum", "component_names": ["holdout_elpd"],
                          "weights": {"holdout_elpd": 1.0}, "lower_log_reward_bound": -20.0,
                          "temperature": temperature,
                          "diagnostics": {"baseline_score": baseline, "gold_score": gold}},
    }


def _refused():
    diagnostics = {"disposition": "refused", "reason_code": "L2_model_build",
                   "error": "NameError: name 'alpha' is not defined", "hidden_n_obs": 10,
                   "training_n_obs": 20}
    return {
        "schema_version": "synthstats.harbor.reward.v1",
        "producer": "ppl_synth_data.harbor.pymc_verifier",
        "provenance": dict(PROVENANCE),
        "diagnostics": diagnostics,
        "reward": 0.0, "training_log_reward": -20.0,
        "reward_source": "validation.pymc_elpd", "evidence_kind": "verifier",
        "truth_owner": "verifier",
        "metrics": {"hidden_n_obs": 10},
        "reward_transform": {"kind": "baseline_normalized_log_reward",
                             "lower_log_reward_bound": -20.0,
                             "error_policy": "candidate_refusal_zero", "authoritative": True},
        "components": {"pymc_elpd_error": 0.0},
        "reward_components": [{
            "name": "pymc_elpd_error", "kind": "penalty", "value": 0.0,
            "source": "validation.pymc_elpd", "evidence_kind": "verifier",
            "truth_owner": "verifier", "producer": "ppl_synth_data.harbor.pymc_verifier",
            "weight": 1.0, "log_contribution": -20.0, "metrics": {},
            "diagnostics": diagnostics, "provenance": dict(PROVENANCE),
        }],
        "scalarization": {"method": "manual", "component_names": ["pymc_elpd_error"],
                          "weights": {"pymc_elpd_error": 1.0}, "lower_log_reward_bound": -20.0},
    }


def _set(record, path, value):
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return record


def test_the_contract_constant_is_exported():
    from synthstats_schemas.numeric_reward import STOCK_ADMISSION_CONTRACT

    assert STOCK_ADMISSION_CONTRACT == STOCK


@pytest.mark.parametrize("build", [_scored, _refused], ids=["scored", "refused"])
def test_a_consistent_stock_row_validates(build):
    RewardArtifact.model_validate(build(), context=CONTEXT)
    RewardArtifact.model_validate(build())  # the declaration alone selects the contract


@pytest.mark.parametrize(
    ("build", "path", "value"),
    [
        (_scored, ("provenance", "verifier"), "harbor_pymc_l0_l6"),
        (_scored, ("metrics", "holdout_elpd"), None),
        (_scored, ("metrics", "hidden_n_obs"), None),
        (_scored, ("metrics", "temperature"), 1.0),
        (_scored, ("reward_transform", "authoritative"), False),
        (_scored, ("reward_transform", "baseline_score"), 999.0),
        (_scored, ("scalarization", "weights"), {"holdout_elpd": 2.0}),
        (_scored, ("reward_components", 0, "weight"), 2.0),
        (_scored, ("reward_components", 0, "log_contribution"), 99.0),
        (_scored, ("reward_components", 0, "value"), -6.0),
        (_scored, ("reward_components", 0, "metrics", "hidden_n_obs"), 3),
        (_refused, ("reward",), 1.0),
        (_refused, ("training_log_reward",), -1.0),
        (_refused, ("reward_transform", "authoritative"), False),
        (_refused, ("reward_components", 0, "weight"), 2.0),
        (_refused, ("reward_components", 0, "value"), 1.0),
    ],
)
def test_a_contradictory_stock_row_is_refused(build, path, value):
    record = _set(copy.deepcopy(build()), path, value)
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(record)


def test_a_row_without_the_declaration_fails_the_stock_expectation():
    record = _scored()
    record["provenance"].pop("admission_contract")
    record["reward_components"][0]["provenance"].pop("admission_contract")
    RewardArtifact.model_validate(record)  # the general envelope still admits it
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(record, context=CONTEXT)


def test_a_numeric_contract_row_fails_the_stock_expectation():
    record = _scored()
    record["provenance"] = {"runtime": "pymc", "verifier": "harbor_pymc_l0_l6",
                            "profile": "strict",
                            "admission_contract": "synthstats.harbor.numeric_admission.v1"}
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(record, context=CONTEXT)


def test_declared_contract_rejects_missing_disposition():
    record = _scored()
    record["diagnostics"].pop("disposition")
    record["reward_transform"] = {"authoritative": False}
    record["metrics"] = {}
    record["reward"] = -123.0
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(record, context=CONTEXT)


@pytest.mark.parametrize("corruption", ["negative_reward", "wrong_log_reward"])
def test_scored_reward_follows_declared_transform(corruption):
    record = _scored()
    if corruption == "negative_reward":
        record["reward"] = -123.0
    else:
        record["training_log_reward"] = 99.0
        record["reward_components"][0]["log_contribution"] = 99.0
        record["reward"] = math.exp(99.0)
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(record, context=CONTEXT)


def test_stock_average_matches_heldout_total_and_count():
    record = _scored()
    record["metrics"]["avg_holdout_logp"] = 123.0
    with pytest.raises(ValidationError):
        RewardArtifact.model_validate(record, context=CONTEXT)


def test_scored_reward_accepts_producer_rounding_of_unrounded_score():
    record = _scored()
    # Producer calculation: ELPD -7.000049, baseline -40, temperature 30.00000001.
    # The emitted ELPD is -7.0, log reward 1.099998, and reward 3.004161.
    record["metrics"]["avg_holdout_logp"] = -0.7000049
    record["training_log_reward"] = 1.099998
    record["reward_components"][0]["log_contribution"] = 1.099998
    record["reward"] = 3.004161
    RewardArtifact.model_validate(record, context=CONTEXT)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_producer_rounded_stock_scores_validate(elpd):
    record = _scored()
    log_reward = max((elpd + 40.0) / 30.00000001, -20.0)
    rounded_elpd = round(elpd, 4)
    record["metrics"].update(holdout_elpd=rounded_elpd, avg_holdout_logp=elpd / 10)
    record["components"]["holdout_elpd"] = rounded_elpd
    record["reward_components"][0].update(value=rounded_elpd, log_contribution=round(log_reward, 6))
    record["training_log_reward"] = round(log_reward, 6)
    record["reward"] = round(math.exp(min(log_reward, 20.0)), 6)
    RewardArtifact.model_validate(record, context=CONTEXT)

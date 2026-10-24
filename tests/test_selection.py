"""Contract tests for the candidate-selection schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthstats_schemas import (
    MIN_HOLDOUT_FOR_SELECTION,
    CandidateKey,
    FitnessRecord,
    GateOutcome,
    SelectionMode,
    SelectionRecord,
    SourcePlanCard,
)


def _passing_gate(name: str) -> GateOutcome:
    return GateOutcome(name=name, passed=True, detail="ok")


def _fitness(holdout_n: int) -> FitnessRecord:
    return FitnessRecord(
        metric="avg_holdout_logp",
        draws=500,
        holdout_n=holdout_n,
        n_splits=4,
        point_estimate=-1.20,
        bootstrap_ci_low=-1.55,
        bootstrap_ci_high=-0.95,
        runner_up_margin=0.18,
    )


def _record(**overrides):
    base = dict(
        runtime="pymc",
        candidate_key=CandidateKey(
            task_id="task-001", family="glm-5.2", candidate_idx=3
        ),
        selection_mode=SelectionMode.selected,
        eligible=True,
        cheap_gate=_passing_gate("crash_gate"),
        integrity_gates=(
            _passing_gate("l3_strict_profile"),
            _passing_gate("l4_fabricated"),
        ),
        density_gates=(_passing_gate("min_predictive_variance"),),
        fitness=_fitness(MIN_HOLDOUT_FOR_SELECTION),
        source_plan_signature="zero_inflated_count/overdispersed",
        winner_reason="highest robust eligible score",
    )
    base.update(overrides)
    return SelectionRecord(**base)


def test_valid_selected_record_round_trips():
    rec = _record()
    again = SelectionRecord.model_validate_json(rec.model_dump_json())
    assert again == rec
    assert again.selection_mode is SelectionMode.selected
    assert again.candidate_key.candidate_idx == 3


def test_eligible_must_equal_all_gates_pass_failing_gate():
    # A failed integrity gate but eligible=True must be rejected.
    with pytest.raises(ValidationError, match="eligible must equal"):
        _record(
            integrity_gates=(
                GateOutcome(name="l4_fabricated", passed=False, detail="boom"),
            ),
        )


def test_eligible_false_while_all_gates_pass_is_rejected():
    with pytest.raises(ValidationError, match="eligible must equal"):
        _record(eligible=False)


def test_selected_below_holdout_threshold_is_rejected():
    with pytest.raises(ValidationError, match="holdout_n >="):
        _record(fitness=_fitness(MIN_HOLDOUT_FOR_SELECTION - 1))


def test_gate_only_allows_no_fitness_when_eligible():
    rec = _record(selection_mode=SelectionMode.gate_only, fitness=None)
    assert rec.fitness is None
    assert rec.eligible is True


def test_rejected_requires_ineligible_and_no_fitness():
    failing = dict(
        cheap_gate=_passing_gate("crash_gate"),
        integrity_gates=(GateOutcome(name="l3", passed=False, detail="data in prior"),),
        density_gates=(_passing_gate("min_predictive_variance"),),
        eligible=False,
        selection_mode=SelectionMode.rejected,
        fitness=None,
    )
    rec = _record(**failing)
    assert rec.selection_mode is SelectionMode.rejected
    # rejected carrying a fitness record is forbidden
    with pytest.raises(ValidationError, match="must not carry a fitness record"):
        _record(**{**failing, "fitness": _fitness(MIN_HOLDOUT_FOR_SELECTION)})


def test_selected_without_fitness_is_rejected():
    with pytest.raises(ValidationError, match="must carry a fitness record"):
        _record(selection_mode=SelectionMode.selected, fitness=None)


def test_fitness_ci_must_bracket_point_estimate():
    with pytest.raises(ValidationError, match="within"):
        FitnessRecord(
            metric="avg_holdout_logp",
            draws=500,
            holdout_n=10,
            point_estimate=0.0,
            bootstrap_ci_low=-1.0,
            bootstrap_ci_high=-0.5,
        )


def test_record_is_frozen_and_forbids_extra():
    rec = _record()
    with pytest.raises(ValidationError):
        rec.eligible = False  # frozen
    with pytest.raises(ValidationError):
        SelectionRecord.model_validate({**rec.model_dump(), "unexpected": 1})


def test_source_plan_card_coerces_json_arrays_to_tuples():
    card = SourcePlanCard.model_validate(
        {
            "runtime": "lazyppl",
            "cell_key": "dp_crp/complex",
            "domain": "manufacturing",
            "structural_signature": "dp_clustered_emission_seen_support",
            "required_structure": ["dp_clustering"],
            "allowed_degrade_ladder": ["finite_mixture_k3", "finite_mixture_k2"],
            "forbidden_fallbacks": ["ordinary_regression"],
            "observable_cues": [
                "several recurring regimes",
                "unknown number of clusters",
            ],
            "validation_recipe": "finite_realized_clusters_logsumexp",
            "n_floor": 40,
        }
    )
    assert isinstance(card.required_structure, tuple)
    assert card.n_floor == 40
    assert card.runtime.value == "lazyppl"


def test_source_plan_card_rejects_zero_n_floor():
    with pytest.raises(ValidationError):
        SourcePlanCard(
            runtime="pymc",
            cell_key="count/simple",
            domain="ecology",
            structural_signature="poisson_loglinear",
            validation_recipe="holdout_logp",
            n_floor=0,
        )

"""The total attribution table: every producer failure triple has a row.

A triple the table does not list raises: there is no default branch, so a
producer change that adds a reason must add its table row first and cannot
silently flip candidate versus infrastructure or mask a row.
"""
from __future__ import annotations

import pytest

from synthstats_schemas.verdicts import (
    ATTRIBUTION_TABLE,
    Attribution,
    TimeoutOrigin,
    VerdictStatus,
    attribution_for,
)


def test_a_structural_refusal_is_the_candidates_own_failure() -> None:
    assert attribution_for("L2_model_build", None, "L2_model_build") is Attribution.CANDIDATE
    assert attribution_for("L0_parse", None, "L0_parse") is Attribution.CANDIDATE


def test_invalid_task_score_metadata_is_infrastructure_during_model_build() -> None:
    assert (
        attribution_for("L2_model_build", None, "task_score_metadata_invalid")
        is Attribution.INFRASTRUCTURE
    )


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("candidate", Attribution.CANDIDATE),
        ("runner_deadline", Attribution.INFRASTRUCTURE),
        ("runner_kill", Attribution.INFRASTRUCTURE),
        ("unresolved", Attribution.UNCHECKED),
        (None, Attribution.UNCHECKED),
    ],
)
def test_timeout_attribution_requires_trusted_origin(origin, expected) -> None:
    assert attribution_for(
        "L2_model_build", None, "verifier_timeout", timeout_origin=origin
    ) is expected


def test_timeout_rejects_unknown_origin() -> None:
    with pytest.raises(ValueError):
        attribution_for("L2_model_build", None, "verifier_timeout", timeout_origin="other")


def test_timeout_without_origin_is_unchecked_and_enum_is_accepted() -> None:
    triple = ("L2_model_build", None, "verifier_timeout")
    assert attribution_for(*triple) is Attribution.UNCHECKED
    assert (
        attribution_for(*triple, timeout_origin=TimeoutOrigin.RUNNER_DEADLINE)
        is Attribution.INFRASTRUCTURE
    )


def test_a_claimed_cause_floors_and_only_trusted_signals_mask() -> None:
    # A cause counts as infrastructure only when the
    # trusted side establishes it (a host or worker signal outside the
    # sandbox), never a reason code the submitted program can write. The
    # named claimed triples floor; the supervisor's timer kill, the worker
    # deaths and invalid task metadata mask; a timeout needs origin evidence.
    for key in (
        ("L2_model_build", None, "verifier_resource_failure"),
        ("L2_model_build", None, "task_data_invalid"),
        ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_resource_failure"),
        ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "task_data_invalid"),
    ):
        assert ATTRIBUTION_TABLE[key] is Attribution.CANDIDATE, key
    for key in (
        ("L2_model_build", None, "verifier_timeout"),
        ("L1_ast_safety", None, "verifier_timeout"),
        ("L0_parse", None, "verifier_timeout"),
        ("L3_harbor_pymc_strict_profile", None, "verifier_timeout"),
        ("L4_exploit_heuristics", None, "verifier_timeout"),
        ("L5_data_perturbation", None, "verifier_timeout"),
        ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_timeout"),
    ):
        assert ATTRIBUTION_TABLE[key] is Attribution.UNCHECKED, key
    assert (
        ATTRIBUTION_TABLE[
            ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "task_score_metadata_invalid")
        ]
        is Attribution.INFRASTRUCTURE
    )
    assert (
        attribution_for(
            "L6_holdout_elpd", VerdictStatus.TIMEOUT.value, "verifier_timeout",
            timeout_origin=TimeoutOrigin.RUNNER_DEADLINE,
        )
        is Attribution.INFRASTRUCTURE
    )
    assert (
        attribution_for(
            "L6_holdout_elpd",
            VerdictStatus.SAMPLING_FAILED.value,
            "worker_died_before_handshake",
        )
        is Attribution.INFRASTRUCTURE
    )


def test_the_parameter_free_likelihood_refusal_is_a_candidate_zero() -> None:
    # A target likelihood with no free parameter
    # is the program's own refusal, a floor, never a masked null.
    assert (
        attribution_for(
            "L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "parameter_free_likelihood"
        )
        is Attribution.CANDIDATE
    )


def test_the_programs_own_statistics_failures_are_candidate() -> None:
    assert (
        attribution_for(
            "L6_holdout_elpd", VerdictStatus.NONFINITE.value, "heldout_score_nonfinite"
        )
        is Attribution.CANDIDATE
    )
    assert (
        attribution_for(
            "L6_holdout_elpd",
            VerdictStatus.SAMPLING_FAILED.value,
            "no_free_variables",
        )
        is Attribution.CANDIDATE
    )


def test_the_ambiguous_families_are_explicit_unchecked_rows() -> None:
    for key in (
        ("L6_holdout_elpd", VerdictStatus.HOLDOUT_FAILED.value, "failure_origin_unresolved"),
        ("L2_model_build", None, "failure_origin_unresolved"),
        ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "failure_origin_unresolved"),
    ):
        assert ATTRIBUTION_TABLE[key] is Attribution.UNCHECKED, key


def test_a_triple_the_table_does_not_list_raises() -> None:
    with pytest.raises(ValueError, match="no attribution row"):
        attribution_for("L2_model_build", None, "a_brand_new_reason")
    with pytest.raises(ValueError, match="no attribution row"):
        attribution_for("L6_holdout_elpd", VerdictStatus.SCORED.value, None)


def test_the_completed_rows_classify_by_the_emission_semantics() -> None:
    # These reason codes classify as candidate failures.
    for key in (
        ("L2_model_build", None, "task_data_invalid"),
        ("L2_model_build", None, "invalid_reward_reference"),
        ("L2_model_build", None, "unsupported_verifier_profile"),
        ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "task_data_invalid"),
        ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "invalid_reward_reference"),
        ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_resource_failure"),
        ("L6_holdout_elpd", VerdictStatus.ISOLATION_FAILED.value, "verifier_contract_failure"),
        ("L3_harbor_pymc_strict_profile", None, "verifier_resource_failure"),
        ("L2_model_build", None, "verifier_execution_unresolved"),
        ("L2_model_build", None, "verifier_predictive_call_error"),
        ("L5_data_perturbation", None, "l5_point_logps_unassessed"),
    ):
        assert ATTRIBUTION_TABLE[key] is Attribution.CANDIDATE, key
    # The program's own nonfinite likelihood and degenerate predictive
    # scale, consistent with the scored-nonfinite and density rows above.
    for key in (
        ("L6_holdout_elpd", VerdictStatus.NONFINITE.value, "heldout_loglik_nonfinite"),
        ("L6_holdout_elpd", VerdictStatus.DENSITY_GATE_FAILED.value, "min_predictive_scale"),
        ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_l5_error"),
        ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "reference_transform_nonfinite"),
    ):
        assert ATTRIBUTION_TABLE[key] is Attribution.CANDIDATE, key
    # The honest unknown that names no cause is the one UNCHECKED family
    # (failure_origin_unresolved stays UNCHECKED, masked, logged
    # and counted as a defect to resolve).
    for key in (
        ("L4_exploit_heuristics", None, "failure_origin_unresolved"),
        ("L0_parse", None, "failure_origin_unresolved"),
    ):
        assert ATTRIBUTION_TABLE[key] is Attribution.UNCHECKED, key

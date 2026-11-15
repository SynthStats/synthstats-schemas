"""The total attribution table: every producer failure triple has a row.

A triple the table does not list raises: there is no default branch, so a
producer change that adds a reason must add its table row first and cannot
silently flip candidate versus infrastructure or mask a row.
"""
from __future__ import annotations

import pytest

from synthstats_schemas.numeric_reward import CompilationComponentDiagnostics
from synthstats_schemas.verdicts import (
    ATTRIBUTION_TABLE,
    Attribution,
    CascadeLayer,
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



# The reason codes whose side depends on who established the cause. A call
# without trusted origin evidence never masks any of them. The first four have
# rows that answer the program's failure when no origin is given; the last four
# are raised by the host-side verifier only, so their rows are UNCHECKED.
ORIGIN_DECIDED_CODES = (
    "verifier_resource_failure",
    "verifier_execution_unresolved",
    "task_data_invalid",
    "invalid_reward_reference",
    "compiler_unavailable",
    "task_data_mismatch",
    "invalid_reward_declaration",
    "reducer_environment_mismatch",
)
HOST_ONLY_CODES = (
    "compiler_unavailable",
    "task_data_mismatch",
    "invalid_reward_declaration",
    "reducer_environment_mismatch",
)

L0 = "L0_parse"
L1 = "L1_ast_safety"
L2 = "L2_model_build"
L3 = "L3_harbor_pymc_strict_profile"
L6 = "L6_holdout_elpd"
L6B = "L6b_predictive_reducer"
OURS = {"fault_origin": "trusted_runner"}
DEADLINE = {"timeout_origin": "runner_deadline"}
KILLED = {"timeout_origin": "runner_kill"}
CANDIDATE = Attribution.CANDIDATE
INFRASTRUCTURE = Attribution.INFRASTRUCTURE
UNCHECKED = Attribution.UNCHECKED

# Faults that are the program's own: (id, failed_layer, status, reason_code).
PROGRAM_FAULTS = [
    ("source_absent", L0, None, "submission_missing"),
    ("source_refused", L1, None, "submission_refused"),
    ("compile_error", L2, None, "compile_failed"),
    ("compile_cpu_limit", L2, None, "compile_timeout"),
    ("compile_heap_limit", L2, None, "compile_resources"),
    ("run_cpu_limit", L3, None, "timeout"),
    ("run_memory_limit", L3, None, "memory_limit"),
    ("packet_over_size_limit", L3, None, "packet_too_large"),
    ("run_exception", L3, None, "packet_rejected"),
    ("reducer_invalid_packet", L6, "density_gate_failed", "predictive_output_invalid"),
    ("reducer_nonfinite_samples", L6, "density_gate_failed", "predictive_samples_nonfinite"),
    ("prediction_off_support", L6, None, "off_support"),
    ("prediction_nonfinite", L6, None, "nonfinite"),
    ("prediction_wrong_shape", L6, None, "wrong_shape"),
    ("prediction_non_numeric", L6, None, "non_numeric"),
]

# Faults the verifier host observes itself, each with the trusted origin
# argument its call site passes: (id, failed_layer, status, reason_code, origin).
HOST_FAULTS = [
    ("setup_fault", L3, "unjudged", "verifier_resource_failure", OURS),
    ("source_copy_failed", L0, "unjudged", "verifier_resource_failure", OURS),
    ("compiler_killed", L2, "unjudged", "verifier_resource_failure", OURS),
    ("program_killed", L3, "unjudged", "verifier_resource_failure", OURS),
    ("run_status_unclassified", L3, "unjudged", "verifier_resource_failure", OURS),
    ("reducer_fault", L6B, "unjudged", "verifier_resource_failure", OURS),
    ("compile_wall_ceiling", L2, "unjudged", "verifier_timeout", DEADLINE),
    ("run_wall_ceiling", L3, "unjudged", "verifier_timeout", DEADLINE),
    ("run_killed_after_ceiling", L3, "unjudged", "verifier_timeout", KILLED),
    ("reducer_wall_ceiling", L6B, "unjudged", "verifier_timeout", DEADLINE),
    ("execution_unresolved", L3, "unjudged", "verifier_execution_unresolved", OURS),
    ("reducer_execution_unresolved", L6B, "unjudged", "verifier_execution_unresolved", OURS),
    ("compiler_exit_without_diagnostic", L2, "unjudged", "compiler_unavailable", OURS),
    ("replay_task_data_differs", L2, "unjudged", "task_data_mismatch", OURS),
    ("reward_declaration_invalid", L6, "unjudged", "invalid_reward_declaration", OURS),
    ("reducer_environment_differs", L6B, "unjudged", "reducer_environment_mismatch", OURS),
]
HOST_FAULT_IDS = [row[0] for row in HOST_FAULTS]


def _evidence(origin, value):
    """The same origin argument with another value."""
    return {name: value for name in origin}


@pytest.mark.parametrize(
    ("layer", "status", "code"),
    [row[1:] for row in PROGRAM_FAULTS],
    ids=[row[0] for row in PROGRAM_FAULTS],
)
def test_a_program_fault_is_the_programs(layer, status, code) -> None:
    assert attribution_for(layer, status, code) is CANDIDATE


@pytest.mark.parametrize(
    ("layer", "status", "code", "origin"),
    [row[1:] for row in HOST_FAULTS],
    ids=HOST_FAULT_IDS,
)
def test_a_host_fault_is_ours_with_its_origin_and_unchecked_without_it(
    layer, status, code, origin
) -> None:
    assert attribution_for(layer, status, code, **origin) is INFRASTRUCTURE
    assert attribution_for(layer, status, code) is UNCHECKED


@pytest.mark.parametrize(
    ("layer", "status", "code", "origin"),
    [row[1:] for row in HOST_FAULTS],
    ids=HOST_FAULT_IDS,
)
def test_a_forgotten_origin_never_charges_a_host_fault_to_the_program(
    layer, status, code, origin
) -> None:
    assert attribution_for(layer, status, code) is not CANDIDATE
    assert attribution_for(layer, status, code, **_evidence(origin, "unresolved")) is not CANDIDATE


@pytest.mark.parametrize(
    ("layer", "status", "code", "origin"),
    [row[1:] for row in HOST_FAULTS],
    ids=HOST_FAULT_IDS,
)
def test_a_host_fault_row_never_masks_without_trusted_origin(
    layer, status, code, origin
) -> None:
    other_kind = (
        {"fault_origin": "trusted_runner"}
        if "timeout_origin" in origin
        else {"timeout_origin": "runner_deadline"}
    )
    for evidence in (
        {},
        _evidence(origin, "unresolved"),
        _evidence(origin, "candidate"),
        other_kind,
    ):
        assert attribution_for(layer, status, code, **evidence) is not INFRASTRUCTURE, evidence


@pytest.mark.parametrize(
    ("layer", "status", "code", "origin"),
    [row[1:] for row in HOST_FAULTS],
    ids=HOST_FAULT_IDS,
)
def test_a_host_fault_is_the_programs_only_on_candidate_evidence(
    layer, status, code, origin
) -> None:
    assert attribution_for(layer, status, code, **_evidence(origin, "candidate")) is CANDIDATE


def test_the_reducer_layer_holds_only_the_reducers_own_faults() -> None:
    reducer_rows = {key for key in ATTRIBUTION_TABLE if key[0] == L6B}
    assert reducer_rows == {
        (L6B, "unjudged", "verifier_resource_failure"),
        (L6B, "unjudged", "verifier_timeout"),
        (L6B, "unjudged", "reducer_environment_mismatch"),
        (L6B, "unjudged", "verifier_execution_unresolved"),
    }
    assert all(ATTRIBUTION_TABLE[key] is UNCHECKED for key in reducer_rows)
    assert all(row[1] != L6B for row in PROGRAM_FAULTS)


def test_the_reducer_layer_is_a_cascade_layer_that_a_record_may_name() -> None:
    assert CascadeLayer(L6B).value == L6B
    diagnostics = CompilationComponentDiagnostics.model_validate(
        {
            "failed_layer": L6B,
            "failure_reason": "the reducer stopped",
            "semantic_status": "critical",
            "disposition": "refused",
            "reason_code": "verifier_resource_failure",
        }
    )
    assert diagnostics.failed_layer is CascadeLayer(L6B)


@pytest.mark.parametrize(
    "triple",
    [
        (L2, None, "compiler_unavailable"),
        (L2, None, "task_data_mismatch"),
        (L3, None, "verifier_execution_unresolved"),
        (L6, "unjudged", "reducer_environment_mismatch"),
    ],
    ids=lambda triple: "-".join(str(part) for part in triple),
)
def test_a_triple_without_a_row_raises_whatever_the_origin(triple) -> None:
    for evidence in ({}, OURS, DEADLINE):
        with pytest.raises(ValueError, match="no attribution row"):
            attribution_for(*triple, **evidence)


@pytest.mark.parametrize("code", ORIGIN_DECIDED_CODES)
def test_a_fault_the_trusted_side_established_is_never_the_programs(code) -> None:
    triples = [key for key in ATTRIBUTION_TABLE if key[2] == code]
    assert triples, code
    for key in triples:
        assert attribution_for(*key, fault_origin="trusted_runner") is INFRASTRUCTURE, key


@pytest.mark.parametrize("code", ORIGIN_DECIDED_CODES)
def test_an_origin_decided_code_never_masks_without_trusted_origin(code) -> None:
    triples = [key for key in ATTRIBUTION_TABLE if key[2] == code]
    assert triples, code
    for key in triples:
        for origin in (None, "unresolved"):
            assert attribution_for(*key, fault_origin=origin) is not INFRASTRUCTURE, (key, origin)
        assert attribution_for(*key, fault_origin="candidate") is CANDIDATE, key


@pytest.mark.parametrize("code", HOST_ONLY_CODES)
def test_a_host_only_code_without_origin_is_unchecked(code) -> None:
    triples = [key for key in ATTRIBUTION_TABLE if key[2] == code]
    assert triples, code
    for key in triples:
        assert attribution_for(*key) is UNCHECKED, key


@pytest.mark.parametrize(
    "triple",
    [
        (L2, None, "task_data_invalid"),
        (L6, "unjudged", "task_data_invalid"),
        (L2, None, "invalid_reward_reference"),
        (L6, "unjudged", "invalid_reward_reference"),
    ],
    ids=lambda triple: "-".join(str(part) for part in triple),
)
def test_the_valuations_unjudged_codes_are_ours_only_with_trusted_origin(triple) -> None:
    assert attribution_for(*triple) is CANDIDATE
    assert attribution_for(*triple, fault_origin="unresolved") is CANDIDATE
    assert attribution_for(*triple, fault_origin="candidate") is CANDIDATE
    assert attribution_for(*triple, fault_origin="trusted_runner") is INFRASTRUCTURE


@pytest.mark.parametrize("origin", [None, "unresolved", "candidate", "trusted_runner"])
def test_a_candidates_own_fault_ignores_origin_evidence(origin) -> None:
    for layer, status, code in (
        (L2, None, "compile_failed"),
        (L3, None, "packet_rejected"),
        (L6, "density_gate_failed", "predictive_output_invalid"),
    ):
        assert attribution_for(layer, status, code, fault_origin=origin) is CANDIDATE


def test_origin_evidence_for_one_kind_of_fault_never_masks_another() -> None:
    assert (
        attribution_for(L2, None, "verifier_timeout", fault_origin="trusted_runner") is UNCHECKED
    )
    assert (
        attribution_for(L2, None, "verifier_resource_failure", timeout_origin="runner_deadline")
        is CANDIDATE
    )


def test_an_unknown_fault_origin_raises_whatever_the_reason() -> None:
    with pytest.raises(ValueError):
        attribution_for(L2, None, "compile_failed", fault_origin="other")
    with pytest.raises(ValueError):
        attribution_for(L2, None, "verifier_timeout", fault_origin="other")


def test_fault_origin_accepts_the_enum_and_rejects_unknown_values() -> None:
    from synthstats_schemas.verdicts import FaultOrigin

    assert (
        attribution_for(
            L3, "unjudged", "verifier_resource_failure", fault_origin=FaultOrigin.TRUSTED_RUNNER
        )
        is INFRASTRUCTURE
    )
    with pytest.raises(ValueError):
        attribution_for(L3, "unjudged", "verifier_resource_failure", fault_origin="other")


def test_an_unlisted_triple_raises_whatever_the_origin() -> None:
    with pytest.raises(ValueError, match="no attribution row"):
        attribution_for(L3, None, "a_brand_new_reason", fault_origin="trusted_runner")

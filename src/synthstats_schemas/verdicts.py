"""The producer's verdict vocabulary and the total attribution table.

One shared vocabulary for consumers and producers. A
triple the producer can emit and the table does not list raises here: there is
no default branch, so a producer change cannot silently flip candidate versus
infrastructure or send a row to a silent mask.

The triple is (failed_layer, status, reason_code): the layer the verifier
failed at, the recorded outcome status, and the producer's reason code. The
table is total over the triples the producer can emit. UNCHECKED is an explicit
table value only where the evidence cannot tell a candidate failure from an
infrastructure failure; such a row is treated as infrastructure (masked: its
reward is null and never the floor reward that a program's own failure earns),
and its count is logged, because each UNCHECKED triple is a defect to resolve.
The producer is the package that writes the verdict records.

The lookup table maps each supported triple to an attribution.

Two kinds of failure are decided by who established the cause, not by the
triple alone: a timeout (``timeout_origin``) and a fault of the verifier host
(``fault_origin``). The verifier host and its runner are the trusted side: they
run outside the sandbox of the submitted program. Each argument carries
evidence from the trusted side, and a producer must pass it for every new
record of these two kinds. A call without it returns UNCHECKED: the failure is
neither charged to the program nor declared infrastructure on the strength of
a claim.
"""
from __future__ import annotations

from enum import Enum


class VerdictStatus(str, Enum):
    """The recorded outcome status (diagnostics.elpd_status) of a verdict."""

    SCORED = "scored"
    BINDING_REFUSED = "binding_refused"
    DENSITY_GATE_FAILED = "density_gate_failed"
    NONFINITE = "nonfinite"
    SAMPLING_FAILED = "sampling_failed"
    HOLDOUT_FAILED = "holdout_failed"
    TIMEOUT = "timeout"
    UNJUDGED = "unjudged"
    ISOLATION_FAILED = "isolation_failed"


class Attribution(str, Enum):
    """Whose failure a verdict is: the candidate's, the infrastructure's, or undecidable.

    A candidate failure (the program is wrong as Python, PyMC or statistics) is
    the policy's own failure: its floor reward counts in the group. An
    infrastructure failure is masked (null, never the floor). UNCHECKED marks
    the triples where the evidence cannot distinguish the two; such a row is
    treated as infrastructure and its count is logged.

    The values are the classification strings stored in projection rows.
    """

    CANDIDATE = "program_defect"
    INFRASTRUCTURE = "infrastructure"
    UNCHECKED = "UNCHECKED"


class TimeoutOrigin(str, Enum):
    """Origin established by the trusted runner, not by submitted code."""

    CANDIDATE = "candidate"
    RUNNER_DEADLINE = "runner_deadline"
    RUNNER_KILL = "runner_kill"
    UNRESOLVED = "unresolved"


class FaultOrigin(str, Enum):
    """Origin of a fault of the verifier host, established by the trusted side.

    The verifier host runs outside the sandbox that executes the submitted
    program. ``TRUSTED_RUNNER`` means the trusted side established that the
    fault arose in the verifier host or its runner, so the fault is not the
    program's. ``CANDIDATE`` means the evidence points to the program.
    ``UNRESOLVED`` means there is no evidence.
    """

    CANDIDATE = "candidate"
    TRUSTED_RUNNER = "trusted_runner"
    UNRESOLVED = "unresolved"


class CascadeLayer(str, Enum):
    """The producer's cascade layer names (the verifier's cascade layers).

    One enum for every surface that names a cascade layer: a refused row's
    failed_layer takes only these names, so a reader
    or a table with no default branch cannot mistake a sentinel for a layer.
    """

    L0_PARSE = "L0_parse"
    L1_AST_SAFETY = "L1_ast_safety"
    L2_MODEL_BUILD = "L2_model_build"
    L3_HARBOR_PYMC_STRICT_PROFILE = "L3_harbor_pymc_strict_profile"
    L4_EXPLOIT_HEURISTICS = "L4_exploit_heuristics"
    L5_DATA_PERTURBATION = "L5_data_perturbation"
    L5B_PREDICTOR_CONNECTIVITY = "L5b_predictor_connectivity"
    L6_HOLDOUT_ELPD = "L6_holdout_elpd"
    L6B_PREDICTIVE_REDUCER = "L6b_predictive_reducer"


# A policy refusal is its own row kind: it names no
# failed layer, so its triple carries its kind in the status slot.
POLICY_REFUSAL_KIND = "policy_refusal"

# The producer's structural layer codes: the layers where a refusal is the
# program's own.
_STRUCTURAL_LAYERS = (
    CascadeLayer.L0_PARSE.value,
    CascadeLayer.L1_AST_SAFETY.value,
    CascadeLayer.L2_MODEL_BUILD.value,
    CascadeLayer.L3_HARBOR_PYMC_STRICT_PROFILE.value,
    CascadeLayer.L4_EXPLOIT_HEURISTICS.value,
    CascadeLayer.L5_DATA_PERTURBATION.value,
)

# The reason codes whose side depends on who established the cause. A submitted
# program can write the first two into its own record, so the rows that such a
# record names floor as the program's failure. The next two come from the checks
# of the task data and the reward reference made before scoring; their rows also
# answer the program's failure unless the trusted side says otherwise. The last
# four are raised by the verifier host. For all eight, ``fault_origin`` carries
# the trusted side's evidence.
_ORIGIN_DECIDED_REASON_CODES = frozenset(
    {
        "verifier_resource_failure",
        "verifier_execution_unresolved",
        "task_data_invalid",
        "invalid_reward_reference",
        "compiler_unavailable",
        "task_data_mismatch",
        "invalid_reward_declaration",
        "reducer_environment_mismatch",
    }
)

# The total table: (failed_layer, status, reason_code) -> Attribution. It is
# total over the producer's FAILURE triples. A scored verdict carries no fault
# to attribute, so its caller handles it before any lookup; attribution_for
# raises for a scored triple like any other triple the table does not list.
#
# The calls follow the classification rule: a cause counts as
# INFRASTRUCTURE only when the trusted side establishes it (a host or worker
# signal outside the sandbox), never a reason code or text that the submitted
# program can write. A claimed code therefore floors as CANDIDATE: a claim
# never buys a mask. failure_origin_unresolved,
# the honest unknown that names no cause, is UNCHECKED (masked, logged and
# counted as a defect to resolve). The trusted signals are the supervisor's
# timer kill (the timeout status), the worker deaths, and the verifier's own
# observed-value transplant probe. A timeout reason alone does not establish
# its origin; attribution_for requires the trusted runner's origin evidence.
ATTRIBUTION_TABLE: dict[tuple[str | None, str | None, str | None], Attribution] = {
    # A policy refusal: no layer failed (the verifier
    # passed passed_layers of total_layers), so its triple is (None, kind,
    # reason_code) -- the kind in the status slot. The refusal is the
    # policy's own act, so it is floored as a candidate failure.
    (None, POLICY_REFUSAL_KIND, POLICY_REFUSAL_KIND): Attribution.CANDIDATE,
    # Structural refusals at the verifier layers: the program is wrong.
    **{
        (layer, None, layer): Attribution.CANDIDATE for layer in _STRUCTURAL_LAYERS
    },
    # The claimed-cause family (the codes the recorded fixtures carry as
    # "submission-authored"): a claim never buys a mask, so each floors as the
    # candidate's own failure.
    ("L2_model_build", None, "failure_origin_unresolved"): Attribution.UNCHECKED,
    ("L2_model_build", None, "verifier_resource_failure"): Attribution.CANDIDATE,
    ("L2_model_build", None, "verifier_timeout"): Attribution.UNCHECKED,
    ("L2_model_build", None, "verifier_contract_failure"): Attribution.CANDIDATE,
    ("L2_model_build", None, "verifier_l5_error"): Attribution.CANDIDATE,
    ("L2_model_build", None, "reference_transform_nonfinite"): Attribution.CANDIDATE,
    ("L2_model_build", None, "task_score_metadata_invalid"): Attribution.INFRASTRUCTURE,
    ("L1_ast_safety", None, "failure_origin_unresolved"): Attribution.UNCHECKED,
    ("L1_ast_safety", None, "verifier_resource_failure"): Attribution.CANDIDATE,
    ("L1_ast_safety", None, "verifier_timeout"): Attribution.UNCHECKED,
    ("L1_ast_safety", None, "verifier_contract_failure"): Attribution.CANDIDATE,
    ("L0_parse", None, "failure_origin_unresolved"): Attribution.UNCHECKED,
    ("L0_parse", None, "verifier_timeout"): Attribution.UNCHECKED,
    ("L5_data_perturbation", None, "failure_origin_unresolved"): Attribution.UNCHECKED,
    # The producer's named rule refusals (the guard family): stock_admission's
    # distribution-call guard and the admission checks refuse a program with a
    # named program-defect reason at the layer whose construct call ran the
    # check (the producer's own refusal-reason codes, carried as reason_code
    # == refusal_code on its refusal record). A claim never buys a mask: each
    # floors as the candidate's own failure. L5's target_not_observed row
    # above folds into this family with the same value; the producer emits
    # predictor_connectivity_failed at L5b.
    **{
        (layer, None, code): Attribution.CANDIDATE
        for layer in (
            "L2_model_build",
            "L3_harbor_pymc_strict_profile",
            "L5_data_perturbation",
        )
        for code in (
            "l5_point_logps_unassessed", "predictor_connectivity_failed",
            "predictive_output_invalid", "predictive_samples_nonfinite",
            "heldout_loglik_nonfinite", "heldout_score_nonfinite",
            "total_size_scaled_likelihood", "target_not_observed",
            "target_observed_more_than_once", "auxiliary_depends_on_target",
            "target_not_row_bound", "row_latent_in_heldout_likelihood",
            "y_self_conditioned", "y_statistic_in_likelihood",
            "prior_predictive_sampling_failed", "prior_depends_on_target",
            "model_evaluation_failed", "prediction_order_dependent",
            "parameter_free_likelihood", "unknown_distribution_keyword",
        )
    },
    ("L5b_predictor_connectivity", None, "predictor_connectivity_failed"): Attribution.CANDIDATE,
    # Held-out and density verdicts: the program's own statistics and its
    # binding faults are candidate.
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "target_not_observed"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "L2_model_build"): Attribution.CANDIDATE,
    # The refusal of a target likelihood with no free parameter is
    # the program's own fault, a floor.
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "parameter_free_likelihood"): Attribution.CANDIDATE,
    ("acquired_input_admission", VerdictStatus.BINDING_REFUSED.value, "empty_acquisition"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.NONFINITE.value, "heldout_score_nonfinite"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "failure_origin_unresolved"): Attribution.UNCHECKED,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_timeout"): Attribution.UNCHECKED,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_contract_failure"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "unsupported_verifier_profile"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "task_score_metadata_invalid"): Attribution.INFRASTRUCTURE,
    ("L6_holdout_elpd", VerdictStatus.DENSITY_GATE_FAILED.value, "density_gate_failed"): Attribution.CANDIDATE,
    # Sampling failures: the worker deaths are trusted signals and
    # infrastructure; the program that cannot be sampled from is candidate.
    ("L6_holdout_elpd", VerdictStatus.SAMPLING_FAILED.value, "worker_died_before_handshake"): Attribution.INFRASTRUCTURE,
    ("L6_holdout_elpd", VerdictStatus.SAMPLING_FAILED.value, "worker_closed_result_channel"): Attribution.INFRASTRUCTURE,
    ("L6_holdout_elpd", VerdictStatus.SAMPLING_FAILED.value, "no_free_variables"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.SAMPLING_FAILED.value, "bad_initial_energy"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.SAMPLING_FAILED.value, "failure_origin_unresolved"): Attribution.UNCHECKED,
    # Holdout failures: the observed-value transplant case is the verifier's
    # own probe (a trusted signal); the TypeErrors whose traceback was never
    # persisted stay explicitly UNCHECKED (the masked family).
    ("L6_holdout_elpd", VerdictStatus.HOLDOUT_FAILED.value, "observed_value_transplant_ineffective"): Attribution.INFRASTRUCTURE,
    ("L6_holdout_elpd", VerdictStatus.HOLDOUT_FAILED.value, "failure_origin_unresolved"): Attribution.UNCHECKED,
    ("L6_holdout_elpd", VerdictStatus.TIMEOUT.value, "verifier_timeout"): Attribution.UNCHECKED,
    ("L0_parse", None, "verifier_resource_failure"): Attribution.CANDIDATE,
    ("L2_model_build", None, "unsupported_verifier_profile"): Attribution.CANDIDATE,
    ("L2_model_build", None, "task_data_invalid"): Attribution.CANDIDATE,
    ("L2_model_build", None, "invalid_reward_reference"): Attribution.CANDIDATE,
    ("L2_model_build", None, "verifier_execution_unresolved"): Attribution.CANDIDATE,
    ("L2_model_build", None, "verifier_predictive_call_error"): Attribution.CANDIDATE,
    ("L3_harbor_pymc_strict_profile", None, "verifier_resource_failure"): Attribution.CANDIDATE,
    ("L3_harbor_pymc_strict_profile", None, "verifier_timeout"): Attribution.UNCHECKED,
    ("L3_harbor_pymc_strict_profile", None, "failure_origin_unresolved"): Attribution.UNCHECKED,
    ("L4_exploit_heuristics", None, "verifier_resource_failure"): Attribution.CANDIDATE,
    ("L4_exploit_heuristics", None, "verifier_timeout"): Attribution.UNCHECKED,
    ("L4_exploit_heuristics", None, "failure_origin_unresolved"): Attribution.UNCHECKED,
    ("L5_data_perturbation", None, "verifier_resource_failure"): Attribution.CANDIDATE,
    ("L5_data_perturbation", None, "verifier_timeout"): Attribution.UNCHECKED,
    ("L6_holdout_elpd", VerdictStatus.NONFINITE.value, "heldout_loglik_nonfinite"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.DENSITY_GATE_FAILED.value, "min_predictive_scale"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.ISOLATION_FAILED.value, "verifier_contract_failure"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "task_data_invalid"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "invalid_reward_reference"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_resource_failure"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_l5_error"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "reference_transform_nonfinite"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_execution_unresolved"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "verifier_predictive_call_error"): Attribution.CANDIDATE,
    # The producer-triples enumeration: the
    # producer's remaining emission sites. Each binding/refusal code is the
    # program's own fault (a floor); the semantic harness's machinery rows are
    # UNCHECKED (masked and counted, a defect to resolve).
    ("L5b_predictor_connectivity", None, "L5b_predictor_connectivity"): Attribution.CANDIDATE,
    ("L5b_predictor_connectivity", None, "predictor_connectivity_unassessed"): Attribution.CANDIDATE,
    ("L5b_predictor_connectivity", None, "verifier_execution_failed"): Attribution.UNCHECKED,
    ("L5b_predictor_connectivity", None, "verifier_setup_failure"): Attribution.UNCHECKED,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "target_observed_more_than_once"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "auxiliary_depends_on_target"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "target_not_row_bound"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "row_latent_in_heldout_likelihood"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "y_self_conditioned"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "y_statistic_in_likelihood"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "covariate_statistic_in_likelihood"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "predictor_connectivity_failed"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "prediction_order_dependent"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "total_size_scaled_likelihood"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "target_column_permutation_invariant"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "prior_depends_on_target"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "prior_predictive_sampling_failed"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.BINDING_REFUSED.value, "model_evaluation_failed"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.DENSITY_GATE_FAILED.value, "predictive_output_invalid"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.DENSITY_GATE_FAILED.value, "predictive_samples_nonfinite"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.DENSITY_GATE_FAILED.value, "predictive_spread_nonfinite"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", VerdictStatus.DENSITY_GATE_FAILED.value, "predictive_spread_unassessed"): Attribution.CANDIDATE,
    # Failures that the verifier host observes in the submitted program, from
    # outside the sandbox that runs it. A source that is missing or refused, a
    # compile error or limit, a run limit and a rejected packet are the
    # program's own failures.
    ("L0_parse", None, "submission_missing"): Attribution.CANDIDATE,
    ("L1_ast_safety", None, "submission_refused"): Attribution.CANDIDATE,
    ("L2_model_build", None, "compile_failed"): Attribution.CANDIDATE,
    ("L2_model_build", None, "compile_timeout"): Attribution.CANDIDATE,
    ("L2_model_build", None, "compile_resources"): Attribution.CANDIDATE,
    ("L3_harbor_pymc_strict_profile", None, "timeout"): Attribution.CANDIDATE,
    ("L3_harbor_pymc_strict_profile", None, "memory_limit"): Attribution.CANDIDATE,
    ("L3_harbor_pymc_strict_profile", None, "packet_too_large"): Attribution.CANDIDATE,
    ("L3_harbor_pymc_strict_profile", None, "packet_rejected"): Attribution.CANDIDATE,
    # Predictions that the scoring of the held-out rows refuses: a value
    # outside the response's support, a non-finite value, a wrong shape or a
    # non-numeric value. Each is the program's own failure.
    ("L6_holdout_elpd", None, "off_support"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", None, "nonfinite"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", None, "wrong_shape"): Attribution.CANDIDATE,
    ("L6_holdout_elpd", None, "non_numeric"): Attribution.CANDIDATE,
    # Faults of the verifier host itself, recorded as unjudged under triples of
    # their own, apart from the rows of the same reasons that a program's own
    # record can name (which answer the program's failure). Without origin
    # evidence the side is unknown, so each row is UNCHECKED (masked, logged
    # and counted as a defect to resolve). ``fault_origin`` (or
    # ``timeout_origin`` for a timeout) supplies the evidence, and only the
    # trusted side's evidence makes the fault ours.
    ("L0_parse", VerdictStatus.UNJUDGED.value, "verifier_resource_failure"): Attribution.UNCHECKED,
    ("L2_model_build", VerdictStatus.UNJUDGED.value, "verifier_resource_failure"): Attribution.UNCHECKED,
    ("L3_harbor_pymc_strict_profile", VerdictStatus.UNJUDGED.value, "verifier_resource_failure"): Attribution.UNCHECKED,
    ("L2_model_build", VerdictStatus.UNJUDGED.value, "verifier_timeout"): Attribution.UNCHECKED,
    ("L3_harbor_pymc_strict_profile", VerdictStatus.UNJUDGED.value, "verifier_timeout"): Attribution.UNCHECKED,
    ("L3_harbor_pymc_strict_profile", VerdictStatus.UNJUDGED.value, "verifier_execution_unresolved"): Attribution.UNCHECKED,
    ("L2_model_build", VerdictStatus.UNJUDGED.value, "compiler_unavailable"): Attribution.UNCHECKED,
    ("L2_model_build", VerdictStatus.UNJUDGED.value, "task_data_mismatch"): Attribution.UNCHECKED,
    ("L6_holdout_elpd", VerdictStatus.UNJUDGED.value, "invalid_reward_declaration"): Attribution.UNCHECKED,
    # The trusted reducer, which turns the program's packet into the predictive
    # score after the program has ended.
    ("L6b_predictive_reducer", VerdictStatus.UNJUDGED.value, "verifier_resource_failure"): Attribution.UNCHECKED,
    ("L6b_predictive_reducer", VerdictStatus.UNJUDGED.value, "verifier_timeout"): Attribution.UNCHECKED,
    ("L6b_predictive_reducer", VerdictStatus.UNJUDGED.value, "reducer_environment_mismatch"): Attribution.UNCHECKED,
    ("L6b_predictive_reducer", VerdictStatus.UNJUDGED.value, "verifier_execution_unresolved"): Attribution.UNCHECKED,
}


def attribution_for(
    failed_layer: str | None,
    status: str | None,
    reason_code: str | None,
    *,
    timeout_origin: TimeoutOrigin | str | None = None,
    fault_origin: FaultOrigin | str | None = None,
) -> Attribution:
    """The attribution of one producer triple.

    A triple the table does not list raises: there is no default branch. A
    producer change that adds a reason must add its table row first. Timeout
    origin must come from the trusted runner, not the candidate's output.

    A producer must pass ``timeout_origin`` for every new record whose reason is
    ``verifier_timeout``. ``candidate`` makes the timeout the program's failure;
    ``runner_deadline`` and ``runner_kill`` make it an infrastructure failure.
    A record without an origin reads ``UNCHECKED``: the row is masked like an
    infrastructure failure, and its count is logged as a defect to resolve.

    A producer must pass ``fault_origin`` for every new record whose reason is one of
    ``verifier_resource_failure``, ``verifier_execution_unresolved``,
    ``task_data_invalid``, ``invalid_reward_reference``,
    ``compiler_unavailable``, ``task_data_mismatch``,
    ``invalid_reward_declaration`` and ``reducer_environment_mismatch``.
    ``trusted_runner`` makes the fault an infrastructure failure and
    ``candidate`` makes it the program's. ``unresolved`` or no origin returns
    the table row. The rows for faults of the verifier host answer
    ``UNCHECKED`` (never the program's). The rows of the same reasons that a
    submitted program can write into its own record answer the program's
    failure. Other reasons do not use ``fault_origin`` to decide, so evidence
    about one kind of fault never masks another. The value is checked on every
    call: an origin that is not a ``FaultOrigin`` value raises ``ValueError``,
    whatever the reason.
    """
    key = (failed_layer, status, reason_code)
    try:
        attribution = ATTRIBUTION_TABLE[key]
    except KeyError:
        raise ValueError(
            f"no attribution row for the producer triple {key!r}; "
            "a producer change that adds a reason must add its table row"
        ) from None
    fault = (
        FaultOrigin(fault_origin)
        if fault_origin is not None
        else FaultOrigin.UNRESOLVED
    )
    if reason_code in _ORIGIN_DECIDED_REASON_CODES:
        if fault is FaultOrigin.TRUSTED_RUNNER:
            return Attribution.INFRASTRUCTURE
        if fault is FaultOrigin.CANDIDATE:
            return Attribution.CANDIDATE
        return attribution
    if reason_code != "verifier_timeout":
        return attribution
    origin = (
        TimeoutOrigin(timeout_origin)
        if timeout_origin is not None
        else TimeoutOrigin.UNRESOLVED
    )
    if origin is TimeoutOrigin.CANDIDATE:
        return Attribution.CANDIDATE
    if origin in (TimeoutOrigin.RUNNER_DEADLINE, TimeoutOrigin.RUNNER_KILL):
        return Attribution.INFRASTRUCTURE
    return Attribution.UNCHECKED

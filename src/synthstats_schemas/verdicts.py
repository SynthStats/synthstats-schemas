"""The producer's verdict vocabulary and the total attribution table.

One shared vocabulary for consumers and producers. A
triple the producer can emit and the table does not list raises here: there is
no default branch, so a producer change cannot silently flip candidate versus
infrastructure or send a row to a silent mask.

The triple is (failed_layer, status, reason_code): the layer the verifier
failed at, the recorded outcome status, and the producer's reason code. The
table is total over the triples the producer can emit. UNCHECKED is an explicit
table value only where the evidence cannot tell a candidate failure from an
infrastructure failure; such a row is treated as infrastructure (null and
masked, never the floor), and its count is logged,
because each UNCHECKED triple is a defect to resolve.

The lookup table maps each supported triple to an attribution.
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
    ("L5_data_perturbation", None, "l5_point_logps_unassessed"): Attribution.CANDIDATE,
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
    ("L5_data_perturbation", None, "target_observed_more_than_once"): Attribution.CANDIDATE,
    ("L5_data_perturbation", None, "target_not_row_bound"): Attribution.CANDIDATE,
    ("L5_data_perturbation", None, "auxiliary_depends_on_target"): Attribution.CANDIDATE,
    ("L5b_predictor_connectivity", None, "L5b_predictor_connectivity"): Attribution.CANDIDATE,
    ("L5b_predictor_connectivity", None, "predictor_connectivity_failed"): Attribution.CANDIDATE,
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
}


def attribution_for(
    failed_layer: str | None,
    status: str | None,
    reason_code: str | None,
    *,
    timeout_origin: TimeoutOrigin | str | None = None,
) -> Attribution:
    """The attribution of one producer triple.

    A triple the table does not list raises: there is no default branch. A
    producer change that adds a reason must add its table row first. Timeout
    origin must come from the trusted runner, not the candidate's output.

    A producer passes ``timeout_origin`` for every new record whose reason is
    ``verifier_timeout``. ``candidate`` makes the timeout the program's failure;
    ``runner_deadline`` and ``runner_kill`` make it an infrastructure failure.
    A record without an origin reads ``UNCHECKED``: the row is masked like an
    infrastructure failure, and its count is logged as a defect to resolve.
    """
    key = (failed_layer, status, reason_code)
    try:
        attribution = ATTRIBUTION_TABLE[key]
    except KeyError:
        raise ValueError(
            f"no attribution row for the producer triple {key!r}; "
            "a producer change that adds a reason must add its table row"
        ) from None
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

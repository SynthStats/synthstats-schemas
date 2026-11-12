"""Check attribution of program defects and infrastructure failures."""
from synthstats_schemas.verdicts import (
    ATTRIBUTION_TABLE,
    Attribution,
    attribution_for,
)

PRODUCER_NAMED_REASONS = (
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
CONSTRUCT_LAYERS = (
    "L2_model_build",
    "L3_harbor_pymc_strict_profile",
    "L5_data_perturbation",
)


def test_named_check_triple_resolves():
    """The named check's own triple (the row for this exact producer triple)."""
    assert (
        attribution_for("L2_model_build", None, "unknown_distribution_keyword")
        is Attribution.CANDIDATE
    )


def test_every_producer_named_reason_floors_as_candidate():
    for layer in CONSTRUCT_LAYERS:
        for code in PRODUCER_NAMED_REASONS:
            assert (
                attribution_for(layer, None, code) is Attribution.CANDIDATE
            ), (layer, code)


def test_l5b_connectivity_failure_floors_as_candidate():
    assert (
        attribution_for("L5b_predictor_connectivity", None, "predictor_connectivity_failed")
        is Attribution.CANDIDATE
    )


def test_negative_controls_keep_their_attribution():
    # The honest unknown stays UNCHECKED (masked), never floored.
    assert (
        attribution_for("L2_model_build", None, "failure_origin_unresolved")
        is Attribution.UNCHECKED
    )
    assert (
        attribution_for("L5_data_perturbation", None, "failure_origin_unresolved")
        is Attribution.UNCHECKED
    )
    # The trusted worker-death signals stay INFRASTRUCTURE, never floored.
    assert (
        attribution_for("L6_holdout_elpd", "sampling_failed", "worker_died_before_handshake")
        is Attribution.INFRASTRUCTURE
    )
    assert (
        attribution_for("L6_holdout_elpd", "sampling_failed", "worker_closed_result_channel")
        is Attribution.INFRASTRUCTURE
    )
    # A triple the producer cannot emit still raises (no silent floor, no mask).
    try:
        attribution_for("L2_model_build", None, "not_a_producer_code")
    except ValueError:
        pass
    else:
        raise AssertionError("an unlisted triple must raise")


def test_table_is_total_over_the_producer_family():
    for layer in CONSTRUCT_LAYERS:
        for code in PRODUCER_NAMED_REASONS:
            assert (layer, None, code) in ATTRIBUTION_TABLE


if __name__ == "__main__":
    test_named_check_triple_resolves()
    test_every_producer_named_reason_floors_as_candidate()
    test_l5b_connectivity_failure_floors_as_candidate()
    test_negative_controls_keep_their_attribution()
    test_table_is_total_over_the_producer_family()
    print("VERDICT-GUARD-TRIPLES-OK")

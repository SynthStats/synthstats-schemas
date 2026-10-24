import math

import pytest
from pydantic import ValidationError

from synthstats_schemas import (
    ATIFExportManifest,
    ATIFRecord,
    EvidenceCard,
    EvidenceKind,
    GenerationRecord,
    LiveRichRewardManifest,
    MixedATIFAssemblyManifest,
    POSTTRAINING_ACCEPTANCE_BUNDLE_SCHEMA_VERSION,
    REWARD_SOURCE_CATALOG,
    RECORD_SCHEMA_VERSION,
    ReleaseManifestAdapter,
    PostTrainingAcceptanceBundle,
    RewardComponent,
    RewardComponentKind,
    RewardArtifact,
    RewardRubric,
    RewardScalarization,
    RewardScalarizationMethod,
    RewardSourceNotRegisteredError,
    RichRewardATIFExportManifest,
    RubricCriterion,
    SubTBTrainingTrajectory,
    TBTrainingTrajectory,
    TaskManifest,
    TraceRecord,
    TraceStep,
    TrainingReceipt,
    TrainingTrajectoryAdapter,
    TrialRecord,
    TruthOwner,
    ensure_reward_source_registered,
    get_capability_validation_requirements,
    get_reward_source,
    list_reward_sources,
)


def _artifact(**overrides):
    payload = {
        "reward_source": "validation.pymc_semantic_holdout_logp",
        "evidence_kind": "verifier",
        "truth_owner": "verifier",
        "producer": "ppl_synth_data.validation.pymc_semantics",
        "training_log_reward": -12.5,
        "reward_transform": {
            "name": "baseline_normalized_log_reward",
            "lower_log_reward_bound": -20.0,
        },
        "metrics": {"elpd_loo": -123.0, "compile_ok": True},
        "diagnostics": {"pareto_k_warning": False},
        "provenance": {"task_id": "surgical"},
        "reward": 0.25,
        "components": {"raw": 0.25},
    }
    payload.update(overrides)
    return RewardArtifact.model_validate(payload)


def test_reward_catalog_is_static_and_explicit() -> None:
    names = list_reward_sources()
    assert "validation.compilation" in names
    assert "validation.formatting" in names
    assert "validation.pymc_semantic_holdout_logp" in names
    assert "validation.expected_information_gain" in names
    assert "validation.predictive_metric" in names
    assert "heuristic.expected_information_gain_proxy" in names
    assert "heuristic.composite_reward" in names
    assert "heuristic.runtime_import" in names
    assert "heuristic.judge_exception" in names
    assert "validation.elpd" not in names
    assert "unknown" not in names
    assert not hasattr(REWARD_SOURCE_CATALOG, "__setitem__")


def test_llm_critic_reward_source_is_not_registered() -> None:
    """heuristic.llm_critic is not a registered reward source."""
    assert "heuristic.llm_critic" not in list_reward_sources()
    with pytest.raises(RewardSourceNotRegisteredError, match="not registered"):
        get_reward_source("heuristic.llm_critic")


def test_reward_source_lookup_is_loud() -> None:
    source = get_reward_source("validation.pymc_semantic_holdout_logp")
    assert source.evidence_kind is EvidenceKind.verifier
    assert source.truth_owner is TruthOwner.verifier

    with pytest.raises(RewardSourceNotRegisteredError):
        ensure_reward_source_registered("validation.elpd")


def test_capability_validation_requirements_are_public_and_copied() -> None:
    requirements = get_capability_validation_requirements("tinker_provider_tb")

    assert requirements["require_dataset_hash"] is True
    assert requirements["require_logz_delta"] is True
    assert requirements["require_source_provenance"] is True

    requirements["require_logz_delta"] = False

    fresh_requirements = get_capability_validation_requirements("tinker_provider_tb")
    assert fresh_requirements["require_logz_delta"] is True


def test_harbor_validator_template_is_compilation_not_semantic_logp() -> None:
    producer = "ppl_synth_data.harbor.validator_template"

    semantic_logp = get_reward_source("validation.pymc_semantic_holdout_logp")
    compilation = get_reward_source("validation.compilation")

    assert producer not in semantic_logp.allowed_producers
    assert producer in compilation.allowed_producers


@pytest.mark.parametrize(
    "bad_source",
    [
        "validation.elpd",
        "validation.likelihood",
        "validation.harbor_pymc_strict_profile",
        "validation.semantic_score",
        "validation.ok",
        "judge_error",
        "verifier",
        "verifier_result",
        "final_metrics",
        "default",
        "missing_artifacts",
        "unsupported_runtime",
        "no_judge",
        "llm_critic",
        "lazyppl_elpd_error",
        "unknown",
    ],
)
def test_reward_artifact_rejects_unregistered_sources(bad_source: str) -> None:
    with pytest.raises(ValidationError, match="not registered"):
        _artifact(reward_source=bad_source)


def test_reward_artifact_requires_training_log_reward() -> None:
    with pytest.raises(ValidationError, match="training_log_reward"):
        RewardArtifact(
            reward_source="validation.pymc_semantic_holdout_logp",
            evidence_kind="verifier",
            truth_owner="verifier",
            producer="ppl_synth_data.validation.pymc_semantics",
            reward_transform={"name": "baseline_normalized_log_reward"},
        )


def test_contracts_reject_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RewardArtifact.model_validate(
            {
                **_artifact().model_dump(mode="json"),
                "unexpected_field": {"reward": 1.0},
            }
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ATIFRecord.model_validate(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "trace-1",
                "steps": [{"source": "agent", "message": "program submitted"}],
                "final_metrics": {"reward": 1.0},
            }
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TBTrainingTrajectory.model_validate(
            {
                "trajectory_id": "trajectory-1",
                "token_ids": [1],
                "logprobs": [-0.1],
                "loss_mask": [True],
                "reward_artifact": _artifact().model_dump(mode="json"),
                "stored_policy_logits": [0.0],
            }
        )


def test_contracts_use_strict_scalar_validation() -> None:
    with pytest.raises(ValidationError) as error:
        _artifact(training_log_reward="-12.5")
    assert [item["loc"] for item in error.value.errors()] == [("training_log_reward",)]

    with pytest.raises(ValidationError) as error:
        _artifact(reward="0.25")
    assert [item["loc"] for item in error.value.errors()] == [("reward",)]

    with pytest.raises(ValidationError) as error:
        _semantic_component(value="-3.0")
    assert [item["loc"] for item in error.value.errors()] == [("value",)]


@pytest.mark.parametrize("bad_value", [math.inf, -math.inf, math.nan])
def test_reward_artifact_rejects_non_finite_training_log_reward(
    bad_value: float,
) -> None:
    with pytest.raises(ValidationError, match="training_log_reward"):
        _artifact(training_log_reward=bad_value)


@pytest.mark.parametrize("bad_value", [math.inf, -math.inf, math.nan])
def test_reward_artifact_rejects_non_finite_metric_values(bad_value: float) -> None:
    with pytest.raises(ValidationError, match="metric"):
        _artifact(metrics={"elpd_loo": bad_value})


def test_reward_artifact_rejects_unfloored_training_log_reward() -> None:
    with pytest.raises(ValidationError, match="greater than or equal"):
        _artifact(training_log_reward=-20.000001)


def test_reward_artifact_rejects_provenance_mismatch() -> None:
    with pytest.raises(ValidationError, match="evidence_kind mismatch"):
        _artifact(evidence_kind="heuristic")
    with pytest.raises(ValidationError, match="truth_owner mismatch"):
        _artifact(truth_owner="heuristic")
    with pytest.raises(ValidationError, match="producer"):
        _artifact(producer="tests")


def test_reward_artifact_round_trips() -> None:
    reward = _artifact()
    assert RewardArtifact.model_validate_json(reward.model_dump_json()) == reward


def _semantic_component(**overrides) -> RewardComponent:
    payload = {
        "name": "semantic_score",
        "kind": "terminal_verifier",
        "value": -3.0,
        "source": "validation.pymc_semantic_holdout_logp",
        "evidence_kind": "verifier",
        "truth_owner": "verifier",
        "producer": "ppl_synth_data.validation.pymc_semantics",
        "weight": 1.0,
        "log_contribution": -0.5,
        "confidence": 1.0,
        "metrics": {"avg_holdout_logp": -3.0},
    }
    payload.update(overrides)
    return RewardComponent.model_validate(payload)


def test_reward_component_validates_source_and_producer() -> None:
    component = _semantic_component()
    assert component.source == "validation.pymc_semantic_holdout_logp"

    with pytest.raises(ValidationError, match="evidence_kind mismatch"):
        _semantic_component(evidence_kind="heuristic")

    with pytest.raises(ValidationError, match="producer"):
        _semantic_component(producer="tests")


def test_reward_artifact_carries_scalarized_component_evidence() -> None:
    component = _semantic_component()
    reward = _artifact(
        reward_source="validation.pymc_semantic_holdout_logp",
        producer="ppl_synth_data.validation.pymc_semantics",
        training_log_reward=-0.5,
        reward_components=(component,),
        scalarization=RewardScalarization(
            method=RewardScalarizationMethod.weighted_sum,
            component_names=("semantic_score",),
            weights={"semantic_score": 1.0},
            lower_log_reward_bound=-20.0,
            temperature=1.0,
        ),
    )

    restored = RewardArtifact.model_validate_json(reward.model_dump_json())

    assert restored.reward_components[0].name == "semantic_score"
    assert restored.scalarization is not None
    assert restored.scalarization.component_names == ("semantic_score",)


def test_reward_artifact_accepts_json_array_fields_explicitly() -> None:
    component = _semantic_component().model_dump(mode="json")
    reward = RewardArtifact.model_validate(
        {
            **_artifact(
                reward_source="validation.pymc_semantic_holdout_logp",
                producer="ppl_synth_data.validation.pymc_semantics",
                training_log_reward=-0.5,
            ).model_dump(mode="json"),
            "reward_components": [component],
            "scalarization": {
                "method": "weighted_sum",
                "component_names": ["semantic_score"],
                "weights": {"semantic_score": 1.0},
            },
        }
    )

    assert reward.reward_components[0].name == "semantic_score"
    assert reward.scalarization is not None
    assert reward.scalarization.component_names == ("semantic_score",)


def test_reward_artifact_rejects_scalarization_component_mismatch() -> None:
    with pytest.raises(ValidationError, match="scalarization component_names"):
        _artifact(
            reward_components=(_semantic_component(),),
            scalarization=RewardScalarization(
                method=RewardScalarizationMethod.weighted_sum,
                component_names=("missing",),
            ),
        )


def test_reward_artifact_carries_evolving_rubric_metadata() -> None:
    reward = _artifact(
        reward_source="heuristic.composite_reward",
        evidence_kind="heuristic",
        truth_owner="heuristic",
        producer="ppl_synth_data.reward.scalarization",
        reward_components=(
            RewardComponent(
                name="model_critique",
                kind=RewardComponentKind.process_rubric,
                value=0.75,
                source="heuristic.composite_reward",
                evidence_kind=EvidenceKind.heuristic,
                truth_owner=TruthOwner.heuristic,
                producer="ppl_synth_data.reward.scalarization",
                confidence=0.6,
            ),
        ),
        scalarization=RewardScalarization(
            method=RewardScalarizationMethod.weighted_sum,
            component_names=("model_critique",),
            weights={"model_critique": 0.1},
        ),
        rubric=RewardRubric(
            rubric_id="boxlm_model_critique",
            version="2026-05-04.1",
            title="BoxLM-style model criticism",
            criteria=(
                RubricCriterion(
                    criterion_id="posterior_predictive_mismatch",
                    description="Penalize unexplained posterior predictive mismatch.",
                    weight=1.0,
                ),
            ),
            parent_version="2026-05-03.1",
            evolution_note="Added posterior-predictive criterion after contrastive failures.",
        ),
    )

    assert reward.rubric is not None
    assert reward.rubric.parent_version == "2026-05-03.1"


def test_process_information_gain_proxy_is_not_verifier_owned_eig() -> None:
    verifier_eig = get_reward_source("validation.expected_information_gain")
    proxy = get_reward_source("heuristic.expected_information_gain_proxy")

    assert verifier_eig.evidence_kind is EvidenceKind.verifier
    assert verifier_eig.truth_owner is TruthOwner.verifier
    assert proxy.evidence_kind is EvidenceKind.heuristic
    assert proxy.truth_owner is TruthOwner.heuristic
    assert (
        "ppl_synth_data.dataset.rich_reward_export" in proxy.allowed_producers
    )

    component = RewardComponent(
        name="process_information_gain_proxy",
        kind=RewardComponentKind.process_information_gain,
        value=0.6,
        source="heuristic.expected_information_gain_proxy",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        producer="ppl_synth_data.dataset.rich_reward_export",
        weight=0.05,
        log_contribution=0.005,
    )
    assert component.source == "heuristic.expected_information_gain_proxy"


def test_reward_artifact_rejects_rubric_without_rubric_component() -> None:
    with pytest.raises(ValidationError, match="rubric requires"):
        _artifact(
            reward_components=(_semantic_component(),),
            rubric=RewardRubric(
                rubric_id="unused",
                version="1",
                title="Unused",
                criteria=(RubricCriterion(criterion_id="c", description="Criterion."),),
            ),
        )


def test_task_manifest_validates_reward_sources() -> None:
    manifest = TaskManifest(
        task_id="surgical",
        name="Surgical survival modeling",
        version="2026-04-30",
        reward_sources=["validation.pymc_semantic_holdout_logp"],
    )
    assert manifest.reward_sources == ["validation.pymc_semantic_holdout_logp"]

    with pytest.raises(ValidationError, match="not registered"):
        TaskManifest(
            task_id="bad",
            name="Bad",
            version="2026-04-30",
            reward_sources=["validation.elpd"],
        )


def test_task_manifest_rejects_duplicate_reward_sources() -> None:
    with pytest.raises(ValidationError, match="reward_sources must be unique"):
        TaskManifest(
            task_id="duplicate-sources",
            name="Duplicate reward sources",
            version="2026-04-30",
            reward_sources=["validation.pymc_semantic_holdout_logp", "validation.pymc_semantic_holdout_logp"],
        )


def test_task_and_card_schema_versions_are_literal() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        TaskManifest(
            schema_version="ppl.platform.task.v1",
            task_id="bad-version",
            name="Bad",
            version="2026-04-30",
        )

    with pytest.raises(ValidationError, match="schema_version"):
        EvidenceCard(
            schema_version="ppl.platform.card.v1",
            evidence_id="evidence-1",
            task_id="surgical",
            reward_artifact=_artifact(),
        )


def test_atif_and_evidence_card_round_trip() -> None:
    reward = _artifact()
    atif = ATIFRecord(
        schema_version="ATIF-v1.6",
        session_id="trace-1",
        steps=[TraceStep(source="agent", message="program submitted")],
        reward_artifact=reward,
    )
    card = EvidenceCard(
        evidence_id="evidence-1",
        task_id="surgical",
        reward_artifact=reward,
        summary="Verifier-owned reward.",
    )
    assert ATIFRecord.model_validate_json(atif.model_dump_json()) == atif
    assert EvidenceCard.model_validate_json(card.model_dump_json()) == card


def test_trace_schema_versions_are_literal() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        ATIFRecord(
            schema_version="atif.v1.6",
            session_id="trace-1",
            steps=[TraceStep(source="agent", message="program submitted")],
        )

    for record_cls, payload in [
        (
            GenerationRecord,
            {
                "record_id": "generation-1",
                "raw_output": "program",
            },
        ),
        (
            TraceRecord,
            {
                "trace_id": "trace-1",
            },
        ),
        (
            TrialRecord,
            {
                "trial_id": "trial-1",
            },
        ),
    ]:
        assert (
            record_cls.model_validate(payload).schema_version == RECORD_SCHEMA_VERSION
        )
        with pytest.raises(ValidationError, match="schema_version"):
            record_cls.model_validate(
                {**payload, "schema_version": "ppl.platform.record.v1"}
            )


def test_tb_trajectory_allows_missing_eos() -> None:
    reward = _artifact()
    trajectory = TBTrainingTrajectory(
        trajectory_id="trajectory-1",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        loss_mask=[True, True, True],
        reward_artifact=reward,
    )
    assert trajectory.eos_logprobs is None
    assert (
        TrainingTrajectoryAdapter.validate_python(trajectory.model_dump()) == trajectory
    )


def test_tb_trajectory_rejects_misaligned_optional_eos() -> None:
    with pytest.raises(ValidationError, match="eos_logprobs length"):
        TBTrainingTrajectory(
            trajectory_id="trajectory-1",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            loss_mask=[True, True, True],
            eos_logprobs=[-1.0],
            reward_artifact=_artifact(),
        )


def test_subtb_trajectory_requires_eos() -> None:
    with pytest.raises(ValidationError, match="eos_logprobs"):
        SubTBTrainingTrajectory(
            trajectory_id="trajectory-1",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            loss_mask=[True, True, True],
            reward_artifact=_artifact(),
        )


def test_training_trajectory_adapter_rejects_subtb_without_eos() -> None:
    with pytest.raises(ValidationError, match="eos_logprobs"):
        TrainingTrajectoryAdapter.validate_python(
            {
                "objective": "subtb",
                "trajectory_id": "trajectory-1",
                "token_ids": [1, 2, 3],
                "logprobs": [-0.1, -0.2, -0.3],
                "loss_mask": [True, True, True],
                "reward_artifact": _artifact().model_dump(mode="json"),
            }
        )


def test_subtb_trajectory_rejects_misaligned_eos() -> None:
    with pytest.raises(ValidationError, match="eos_logprobs length"):
        SubTBTrainingTrajectory(
            trajectory_id="trajectory-1",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            loss_mask=[True, True, True],
            eos_logprobs=[-1.0],
            reward_artifact=_artifact(),
        )


def test_training_trajectory_rejects_bad_token_arrays() -> None:
    with pytest.raises(ValidationError, match="loss_mask must contain"):
        TBTrainingTrajectory(
            trajectory_id="trajectory-1",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            loss_mask=[False, False, False],
            reward_artifact=_artifact(),
        )

    with pytest.raises(ValidationError, match="token_ids must be non-negative"):
        TBTrainingTrajectory(
            trajectory_id="trajectory-1",
            token_ids=[1, -2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            loss_mask=[True, True, True],
            reward_artifact=_artifact(),
        )


_HASH = "a" * 64
_HASH_B = "b" * 64


def _atif_trial_artifact(**overrides):
    payload = {
        "atif_id": "pymc__run__completion-000001",
        "trial_dir": "/tmp/atif/pymc__run__completion-000001",
        "runtime": "pymc",
        "split": "train",
        "local_record_id": "completion-000001",
        "prompt_id": "prompt-000001",
        "env_id": "env-a",
        "reward_source": "validation.pymc_semantic_holdout_logp",
        "training_log_reward": -0.5,
        "data_ref": "stage_C_data/data/env-a.parquet",
        "data_ref_sha256": _HASH,
        "prompt_sha256": _HASH,
        "completion_sha256": _HASH,
        "parsed_program_sha256": _HASH,
        "validation_sha256": _HASH,
        "env_spec_sha256": _HASH,
        "stage_g_row_sha256": _HASH,
        "trajectory_sha256": _HASH,
        "reward_sha256": _HASH,
    }
    payload.update(overrides)
    return payload


def _atif_export_manifest(**overrides):
    artifact = _atif_trial_artifact()
    payload = {
        "status": "success",
        "schema_version": "ppl_synth_data.atif_export.v1",
        "digest_algorithm": "sha256",
        "run_dir": "/tmp/run",
        "out_dir": "/tmp/atif",
        "metric_name": "avg_holdout_logp",
        "baseline": "mean",
        "baseline_value": -10.0,
        "reward_temperature": 1.0,
        "runtime": "pymc",
        "split": "train",
        "reward_mode": "semantic_metric",
        "reward_source": "validation.pymc_semantic_holdout_logp",
        "policy_logprobs_present": False,
        "training_input_mode": "message_only_requires_policy_rescore",
        "direct_offline_collect_training_supported": False,
        "identity_mode": "runtime_run_local_record_id",
        "count_input": 1,
        "count_exported": 1,
        "count_unique_atif_ids": 1,
        "split_counts": {"train": 1},
        "atif_ids": ["pymc__run__completion-000001"],
        "count_dropped": 0,
        "trial_dirs": ["/tmp/atif/pymc__run__completion-000001"],
        "trial_artifacts": {"pymc__run__completion-000001": artifact},
        "dropped": [],
    }
    payload.update(overrides)
    return payload


def _mixed_trial_artifact(**overrides):
    payload = {
        **_atif_trial_artifact(),
        "trial_dir": "/tmp/mixed-atif/pymc__run__completion-000001",
        "source_run_id": "source-a",
        "source_run_dir": "/tmp/source-run-a",
        "source_atif_dir": "/tmp/atif-a",
        "source_trajectory_sha256": _HASH_B,
        "source_reward_sha256": _HASH_B,
    }
    payload.update(overrides)
    return payload


def _mixed_atif_manifest(**overrides):
    artifact = _mixed_trial_artifact()
    payload = {
        "status": "success",
        "schema_version": "ppl_synth_data.mixed_atif_assembly.v1",
        "digest_algorithm": "sha256",
        "timestamp_utc": "2026-06-17T08:00:00+00:00",
        "out_dir": "/tmp/mixed-atif",
        "source_run_dirs": {"source-a": "/tmp/source-run-a"},
        "source_atif_dirs": {"source-a": "/tmp/atif-a"},
        "source_manifest_sha256": {"source-a": _HASH},
        "trial_source_run_ids": {"pymc__run__completion-000001": "source-a"},
        "source_counts": {"source-a": 1},
        "runtime_counts": {"pymc": 1},
        "reward_source_counts": {"validation.pymc_semantic_holdout_logp": 1},
        "split_counts": {"train": 1},
        "count_exported": 1,
        "row_count": 1,
        "count_unique_atif_ids": 1,
        "atif_ids": ["pymc__run__completion-000001"],
        "trial_dirs": ["/tmp/mixed-atif/pymc__run__completion-000001"],
        "trial_artifacts": {"pymc__run__completion-000001": artifact},
    }
    payload.update(overrides)
    return payload


def test_atif_export_manifest_round_trips_and_uses_release_adapter() -> None:
    manifest = ATIFExportManifest.model_validate(_atif_export_manifest())
    restored = ReleaseManifestAdapter.validate_json(manifest.model_dump_json())

    assert restored == manifest
    assert manifest.count_exported == 1
    assert (
        manifest.trial_artifacts["pymc__run__completion-000001"].env_spec_sha256
        == _HASH
    )


def test_atif_export_manifest_rejects_manifest_drift() -> None:
    with pytest.raises(ValidationError, match="count_exported"):
        ATIFExportManifest.model_validate(_atif_export_manifest(count_exported=2))

    with pytest.raises(ValidationError, match="data_ref"):
        ATIFExportManifest.model_validate(
            _atif_export_manifest(
                trial_artifacts={
                    "pymc__run__completion-000001": _atif_trial_artifact(
                        data_ref="../secret.parquet",
                    )
                }
            )
        )

    with pytest.raises(ValidationError, match="env_spec_sha256"):
        payload = _atif_export_manifest()
        payload["trial_artifacts"]["pymc__run__completion-000001"].pop(
            "env_spec_sha256"
        )
        ATIFExportManifest.model_validate(payload)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ATIFExportManifest.model_validate(
            {
                **_atif_export_manifest(),
                "unexpected_field": True,
            }
        )

    with pytest.raises(ValidationError, match="reward_source"):
        ATIFExportManifest.model_validate(
            _atif_export_manifest(reward_source="validation.elpd")
        )


def test_mixed_atif_manifest_requires_source_maps_and_lineage() -> None:
    manifest = MixedATIFAssemblyManifest.model_validate(_mixed_atif_manifest())
    restored = ReleaseManifestAdapter.validate_json(manifest.model_dump_json())

    assert restored == manifest
    assert manifest.source_run_dirs == {"source-a": "/tmp/source-run-a"}
    assert manifest.trial_source_run_ids == {"pymc__run__completion-000001": "source-a"}


def test_mixed_atif_manifest_rejects_source_map_drift() -> None:
    with pytest.raises(ValidationError, match="source_run_dirs"):
        MixedATIFAssemblyManifest.model_validate(
            _mixed_atif_manifest(source_run_dirs={})
        )

    with pytest.raises(ValidationError, match="unknown source_run_id"):
        MixedATIFAssemblyManifest.model_validate(
            _mixed_atif_manifest(
                trial_source_run_ids={"pymc__run__completion-000001": "missing-source"}
            )
        )

    with pytest.raises(ValidationError, match="artifact source_run_dir"):
        MixedATIFAssemblyManifest.model_validate(
            _mixed_atif_manifest(
                trial_artifacts={
                    "pymc__run__completion-000001": _mixed_trial_artifact(
                        source_run_dir="/tmp/other",
                    )
                }
            )
        )

    with pytest.raises(ValidationError, match="trial_dir"):
        payload = _mixed_atif_manifest()
        payload["trial_artifacts"]["pymc__run__completion-000001"].pop("trial_dir")
        MixedATIFAssemblyManifest.model_validate(payload)

    with pytest.raises(ValidationError, match="trial_dirs must match"):
        MixedATIFAssemblyManifest.model_validate(
            _mixed_atif_manifest(trial_dirs=["/tmp/mixed-atif/other"])
        )

    with pytest.raises(ValidationError, match="artifact trial_dir basename"):
        MixedATIFAssemblyManifest.model_validate(
            _mixed_atif_manifest(
                trial_dirs=["/tmp/mixed-atif/other"],
                trial_artifacts={
                    "pymc__run__completion-000001": _mixed_trial_artifact(
                        trial_dir="/tmp/mixed-atif/other",
                    )
                },
            )
        )

    with pytest.raises(ValidationError, match="reward_source_counts"):
        MixedATIFAssemblyManifest.model_validate(
            _mixed_atif_manifest(reward_source_counts={"validation.compilation": 1})
        )


def test_release_manifests_reject_bad_hash_and_non_finite_rewards() -> None:
    with pytest.raises(ValidationError, match="data_ref_sha256"):
        ATIFExportManifest.model_validate(
            _atif_export_manifest(
                trial_artifacts={
                    "pymc__run__completion-000001": _atif_trial_artifact(
                        data_ref_sha256="not-a-hash",
                    )
                }
            )
        )

    with pytest.raises(ValidationError, match="training_log_reward"):
        ATIFExportManifest.model_validate(
            _atif_export_manifest(
                trial_artifacts={
                    "pymc__run__completion-000001": _atif_trial_artifact(
                        training_log_reward=math.inf,
                    )
                }
            )
        )


def test_rich_reward_manifest_contracts_are_strict() -> None:
    rich = RichRewardATIFExportManifest.model_validate(
        {
            "status": "success",
            "schema_version": "ppl_synth_data.rich_reward_atif_export.v1",
            "source_atif_dir": "/tmp/atif",
            "out_dir": "/tmp/rich",
            "rubric_scores_path": "/tmp/rubric.json",
            "reward_source": "heuristic.composite_reward",
            "base_reward_sources": ["validation.pymc_semantic_holdout_logp"],
            "component_reward_sources": ["heuristic.composite_reward"],
            "rubric_id": "boxlm_model_critique",
            "rubric_version": "2026-06-17",
            "heuristic_weight": 0.1,
            "process_weight": 0.2,
            "count_input": 1,
            "count_exported": 1,
            "count_dropped": 0,
            "atif_ids": ["lazyppl__run__completion-000001"],
            "dropped": [],
            "training_input_mode": "message_only_requires_policy_rescore",
        }
    )
    assert ReleaseManifestAdapter.validate_json(rich.model_dump_json()) == rich

    with pytest.raises(ValidationError, match="count_input"):
        RichRewardATIFExportManifest.model_validate(
            {**rich.model_dump(mode="json"), "count_dropped": 1}
        )

    live = LiveRichRewardManifest.model_validate(
        {
            "status": "success",
            "schema_version": "ppl_synth_data.live_rich_reward.v1",
            "trial_dir": "/tmp/trial",
            "atif_id": "lazyppl__run__completion-000001",
            "base_trajectory_path": "/tmp/trial/trajectory.json",
            "reward_source": "heuristic.composite_reward",
            "component_reward_sources": ["heuristic.composite_reward"],
            "component_names": ["rubric"],
            "scalarization": {"method": "weighted_sum"},
            "rubric": {"rubric_id": "boxlm_model_critique"},
            "written_reward_paths": ["/tmp/trial/reward.json"],
            "written_trajectory_paths": ["/tmp/trial/trajectory.json"],
        }
    )
    assert ReleaseManifestAdapter.validate_json(live.model_dump_json()) == live

    with pytest.raises(ValidationError, match="written_reward_paths"):
        LiveRichRewardManifest.model_validate(
            {**live.model_dump(mode="json"), "written_reward_paths": []}
        )


def _training_receipt(**overrides):
    payload = {
        "schema_version": "synthstats.training_receipt.v1",
        "runner": {"type": "offline_rescore"},
        "dataset": {
            "path": "/tmp/atif",
            "row_count": 2,
            "train_rows": 2,
            "unique_atif_ids_consumed": ["row-a", "row-b"],
            "unique_atif_ids_consumed_count": 2,
            "reward_stats": {
                "training_log_reward_min": -0.5,
                "training_log_reward_max": 1.0,
                "training_log_reward_mean": 0.25,
                "training_log_reward_std": 0.75,
            },
            "consumed_reward_evidence": {
                "reward_source_counts": {"validation.pymc_semantic_holdout_logp": 2},
                "truth_owner_counts": {"verifier": 2},
            },
        },
        "objective": {
            "loss_type": "tb",
            "loss_name": "gfn_tb",
            "initial_logZ": 0.0,
            "final_logZ": 0.2,
            "tb_max_residual": 500.0,
            "subtb_lambda": 0.9,
            "entropy_coef": 0.0,
        },
        "policy": {
            "model_name": "Qwen/Qwen3.5-0.8B",
            "dtype": "bfloat16",
        },
        "optimization": {
            "steps": 2,
            "batch_size": 1,
            "policy_lr": 0.00001,
            "logZ_lr": 0.01,
            "max_grad_norm": 1.0,
        },
        "metrics": {
            "objective_gfn_tb_executed": 1.0,
            "training_log_reward_consumed": 2.0,
            "consumed_reward_sources": ["validation.pymc_semantic_holdout_logp"],
        },
        "artifacts": {
            "policy_weight_change_report": "policy_weight_change_report.json",
            "policy_weight_change_report_sha256": _HASH,
            "final_checkpoint": "checkpoints/final",
            "checkpoints": ["checkpoints/final"],
        },
        "weight_change": {
            "schema_version": "synthstats.policy_weight_change_report.v1",
            "proof_type": "hf_lora_delta",
            "changed_tensor_count": 48,
            "weight_delta_l2": 0.02,
            "tensor_count": 48,
        },
        "provenance": {
            "source_identity_kind": "source_tree_sha256",
            "source_identity": _HASH,
            "source_tree_sha256": _HASH,
            "source_tree_file_count": 10,
            "source_tree_total_bytes": 1024,
        },
    }
    payload.update(overrides)
    return payload


def test_training_receipt_contract_accepts_real_proof_shape() -> None:
    receipt = TrainingReceipt.model_validate(_training_receipt())

    assert receipt.schema_version == "synthstats.training_receipt.v1"
    assert receipt.dataset.train_rows == 2
    assert receipt.weight_change.changed_tensor_count == 48


def test_training_receipt_contract_rejects_permissive_or_fake_shapes() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainingReceipt.model_validate(
            {**_training_receipt(), "unexpected_field": True}
        )

    bad_metrics = _training_receipt()
    bad_metrics["metrics"]["loss"] = math.inf
    with pytest.raises(ValidationError, match="metrics"):
        TrainingReceipt.model_validate(bad_metrics)

    with pytest.raises(ValidationError, match="unique_atif_ids_consumed_count"):
        TrainingReceipt.model_validate(
            _training_receipt(
                dataset={
                    **_training_receipt()["dataset"],
                    "row_count": 1,
                }
            )
        )

    with pytest.raises(ValidationError, match="weight_delta_l2"):
        TrainingReceipt.model_validate(
            _training_receipt(
                weight_change={
                    "schema_version": "synthstats.policy_weight_change_report.v1",
                    "changed_tensor_count": 1,
                    "weight_delta_l2": 0.0,
                }
            )
        )


def test_tinker_training_receipt_requires_provider_evidence() -> None:
    tinker = _training_receipt(
        runner={"type": "tinker"},
        policy={
            "model_name": "Qwen/Qwen3.5-4B",
            "backend": "tinker",
            "dtype": "bfloat16",
            "lora_config": {"rank": 8},
        },
        artifacts={
            **_training_receipt()["artifacts"],
            "final_tinker_remote_state": {
                "type": "tinker_checkpoint",
                "path": "tinker://session:train:0/weights/final",
                "provider_session_id": "session",
            },
        },
        provider={
            "name": "tinker",
            "capability_probe": {
                "selected_model_name": "Qwen/Qwen3.5-4B",
                "selected_model_supported": True,
            },
        },
    )
    assert TrainingReceipt.model_validate(tinker).provider is not None

    with pytest.raises(ValidationError, match="provider"):
        TrainingReceipt.model_validate({**tinker, "provider": None})


def _acceptance_bundle(**overrides):
    payload = {
        "schema_version": POSTTRAINING_ACCEPTANCE_BUNDLE_SCHEMA_VERSION,
        "claim": {
            "level": "current-proof-scale",
            "description": "unit test bundle",
        },
        "required_capabilities": ["offline_current_policy_tb_rescore"],
        "files": [
            {
                "name": "proof_archive",
                "path": "proof.tar.gz",
                "sha256": _HASH,
                "sha256_sidecar": "proof.tar.gz.sha256",
            }
        ],
        "training_receipts": [
            {
                "name": "offline_tb",
                "path": "runs/offline_tb/training_receipt.json",
                "sha256": _HASH,
                "validation": "offline_tb_validation.json",
                "validation_sha256": _HASH,
                "weight_report": "runs/offline_tb/policy_weight_change_report.json",
                "weight_report_sha256": _HASH,
                "capabilities": ["offline_current_policy_tb_rescore"],
                "min_canonical_reward_sources": 1,
                "min_unique_rows": 2,
                "require_truth_owner": "verifier",
                "require_finite_training_reward_stats": True,
                "validation_config": {
                    "require_objective": "gfn_tb",
                    "min_train_rows": 2,
                    "min_unique_rows": 2,
                    "require_dataset_hash": True,
                    "require_final_checkpoint": True,
                    "require_logz_delta": True,
                    "require_policy_delta": True,
                    "require_training_log_reward_consumed": True,
                    "require_training_trajectory_adapter": True,
                },
            }
        ],
        "secret_scans": [
            {"name": "secret_scan", "path": "secret_scan.json", "sha256": _HASH},
            {
                "name": "empty_scan",
                "kind": "empty_text",
                "path": "secret_scan.txt",
                "sha256": _HASH,
            },
        ],
        "cleanup_receipts": [
            {
                "name": "sky_cleanup",
                "kind": "sky_status_clean",
                "path": "sky_status.txt",
                "sha256": _HASH,
                "run_id": "unit-test-run",
                "scope": {"provider": "skypilot", "command": "sky status --refresh"},
            },
            {
                "name": "volumes_cleanup",
                "kind": "json_empty_collection",
                "field": "Volumes",
                "path": "volumes.json",
                "sha256": _HASH,
                "run_id": "unit-test-run",
                "scope": {"provider": "aws", "resource": "ebs"},
            },
            {
                "name": "tinker_cleanup",
                "kind": "tinker_checkpoints_deleted",
                "path": "tinker_cleanup.json",
                "sha256": _HASH,
                "run_id": "unit-test-run",
                "scope": {"provider": "tinker", "resource": "checkpoints"},
            },
        ],
        "prohibited_marker_scans": [
            {"name": "logs", "paths": ["logs/run.log"]},
        ],
        "secret_text_scans": [
            {"name": "logs", "paths": ["logs/run.log"]},
        ],
    }
    payload.update(overrides)
    return payload


def test_posttraining_acceptance_bundle_contract_is_closed() -> None:
    bundle = PostTrainingAcceptanceBundle.model_validate(_acceptance_bundle())

    assert bundle.required_capabilities == ["offline_current_policy_tb_rescore"]
    assert bundle.cleanup_receipts[2].kind == "tinker_checkpoints_deleted"

    file_capability_bundle = _acceptance_bundle(
        required_capabilities=["release_grade_mixed_atif"],
        training_receipts=[],
        files=[
            {
                "name": "release_manifest",
                "kind": "release_manifest",
                "path": "manifest.json",
                "sha256": _HASH,
                "capabilities": ["release_grade_mixed_atif"],
                "min_count_exported": 120,
                "min_runtime_counts": {"pymc": 1, "lazyppl": 1},
                "min_split_counts": {"train": 1, "validation": 1, "test": 1},
                "min_reward_sources": 2,
                "require_zero_dropped": True,
                "require_nonzero_reward_variance": True,
                "require_data_ref_digests": True,
                "require_env_spec_digests": True,
                "require_trial_artifacts_match_count": True,
            }
        ],
    )
    assert PostTrainingAcceptanceBundle.model_validate(file_capability_bundle).files[
        0
    ].capabilities == ["release_grade_mixed_atif"]

    with pytest.raises(ValidationError, match="required_capabilities"):
        PostTrainingAcceptanceBundle.model_validate(
            _acceptance_bundle(required_capabilities=["tinker_provider_tb"])
        )

    weak_capability = _acceptance_bundle()
    weak_capability["training_receipts"][0]["validation_config"].pop(
        "require_logz_delta"
    )
    with pytest.raises(ValidationError, match="requires validation_config"):
        PostTrainingAcceptanceBundle.model_validate(weak_capability)

    producer_bundle = _acceptance_bundle(
        required_capabilities=["producer_fresh_provider_generation"],
        training_receipts=[],
        evidence_receipts=[
            {
                "name": "fresh_openai_generation",
                "kind": "fresh_provider_generation",
                "path": "fresh_generation_summary.json",
                "sha256": _HASH,
                "validation": "atif_validation.json",
                "validation_sha256": _HASH,
                "capabilities": ["producer_fresh_provider_generation"],
                "min_count_trials": 1,
                "min_stage_f_validated": 1,
                "require_provider_response_id": True,
                "require_provider_usage": True,
                "require_request_id_coverage": True,
                "require_zero_provider_failures": True,
                "expected_provider_backend": "openai",
                "expected_completion_model": "gpt-5.5-pro",
                "semantic_validation_settings": {
                    "pymc_draws": 100,
                    "pymc_tune": 100,
                    "pymc_timeout_sec": 180,
                },
            }
        ],
    )
    producer_receipt = PostTrainingAcceptanceBundle.model_validate(
        producer_bundle
    ).evidence_receipts[0]
    assert producer_receipt.capabilities == ["producer_fresh_provider_generation"]
    assert producer_receipt.semantic_validation_settings is not None
    assert producer_receipt.semantic_validation_settings.pymc_draws == 100

    invalid_semantic_budget = producer_bundle.copy()
    invalid_semantic_budget["evidence_receipts"] = [
        dict(producer_bundle["evidence_receipts"][0])
    ]
    invalid_semantic_budget["evidence_receipts"][0]["semantic_validation_settings"] = {
        "pymc_draws": 0,
        "pymc_tune": 100,
        "pymc_timeout_sec": 180,
    }
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        PostTrainingAcceptanceBundle.model_validate(invalid_semantic_budget)

    weak_provider_bundle = producer_bundle.copy()
    weak_provider_bundle["evidence_receipts"] = [
        dict(producer_bundle["evidence_receipts"][0])
    ]
    weak_provider_bundle["evidence_receipts"][0].pop("require_provider_usage")
    with pytest.raises(ValidationError, match="require_provider_usage"):
        PostTrainingAcceptanceBundle.model_validate(weak_provider_bundle)

    direct_subtb_bundle = _acceptance_bundle(
        required_capabilities=["direct_fixed_logprob_subtb_replay"],
        training_receipts=[],
        evidence_receipts=[
            {
                "name": "direct_subtb_replay",
                "kind": "direct_fixed_logprob_subtb_replay",
                "path": "direct_subtb_replay_summary.json",
                "sha256": _HASH,
                "validation": "derived_packet_contract_summary.json",
                "validation_sha256": _HASH,
                "derived_manifest": "derived_manifest.json",
                "derived_manifest_sha256": _HASH,
                "steps": "steps.txt",
                "steps_sha256": _HASH,
                "required_steps": [
                    "derive_live_subtb_atif_packet",
                    "validate_derived_packet_contracts",
                    "direct_fixed_logprob_subtb_replay",
                    "summarize_direct_subtb_replay",
                ],
                "capabilities": ["direct_fixed_logprob_subtb_replay"],
                "min_row_count": 2,
                "min_response_token_count": 1,
                "min_checkpoint_count": 1,
                "require_eos_logprobs": True,
                "require_terminal_eos": True,
                "require_objective_marker": True,
                "require_fixed_logprob_boundary": True,
                "expected_reward_source": "validation.compilation",
                "expected_truth_owner": "verifier",
                "expected_source_rollout_tensor_artifact_sha256": _HASH,
            }
        ],
    )
    assert (
        PostTrainingAcceptanceBundle.model_validate(direct_subtb_bundle)
        .evidence_receipts[0]
        .kind
        == "direct_fixed_logprob_subtb_replay"
    )

    weak_direct_subtb = direct_subtb_bundle.copy()
    weak_direct_subtb["evidence_receipts"] = [
        dict(direct_subtb_bundle["evidence_receipts"][0])
    ]
    weak_direct_subtb["evidence_receipts"][0]["require_fixed_logprob_boundary"] = False
    with pytest.raises(ValidationError, match="require_fixed_logprob_boundary"):
        PostTrainingAcceptanceBundle.model_validate(weak_direct_subtb)

    missing_scan_hash = _acceptance_bundle()
    missing_scan_hash["secret_scans"][0].pop("sha256")
    with pytest.raises(ValidationError, match="sha256"):
        PostTrainingAcceptanceBundle.model_validate(missing_scan_hash)

    missing_cleanup_scope = _acceptance_bundle()
    missing_cleanup_scope["cleanup_receipts"][0].pop("scope")
    with pytest.raises(ValidationError, match="scope"):
        PostTrainingAcceptanceBundle.model_validate(missing_cleanup_scope)

    with pytest.raises(ValidationError, match="literal_error"):
        PostTrainingAcceptanceBundle.model_validate(
            _acceptance_bundle(required_capabilities=["policy_mock"])
        )

    invalid_config = _acceptance_bundle()
    invalid_config["training_receipts"][0]["validation_config"]["unexpected_field"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PostTrainingAcceptanceBundle.model_validate(invalid_config)

    with pytest.raises(ValidationError, match="kind"):
        PostTrainingAcceptanceBundle.model_validate(
            _acceptance_bundle(
                cleanup_receipts=[
                    {
                        "name": "cleanup",
                        "kind": "local_docker_ok",
                        "path": "cleanup.json",
                        "sha256": _HASH,
                        "run_id": "unit-test-run",
                        "scope": {"provider": "local"},
                    }
                ]
            )
        )

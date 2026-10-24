"""Training receipt and post-training evidence-bundle contracts."""

from __future__ import annotations

import math
from typing import Annotated, Any, Final, Literal

from pydantic import Field, field_validator, model_validator

from synthstats_schemas.base import ContractModel, require_json_serializable

TRAINING_RECEIPT_SCHEMA_VERSION: Final[Literal["synthstats.training_receipt.v1"]] = (
    "synthstats.training_receipt.v1"
)
POLICY_WEIGHT_CHANGE_REPORT_SCHEMA_VERSION: Final[
    Literal["synthstats.policy_weight_change_report.v1"]
] = "synthstats.policy_weight_change_report.v1"
POSTTRAINING_ACCEPTANCE_BUNDLE_SCHEMA_VERSION: Final[
    Literal["synthstats.posttraining_acceptance_bundle.v1"]
] = "synthstats.posttraining_acceptance_bundle.v1"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]

RunnerType = Literal["sft", "offline_rescore", "skyrl", "tinker", "grpo_tinker"]
ObjectiveLossType = Literal["sft", "tb", "subtb", "drgrpo"]
BundleCapability = Literal[
    "sft_warmstart",
    "offline_current_policy_tb_rescore",
    "offline_current_policy_subtb_rescore",
    "live_harbor_skyrl_daytona_tb",
    "live_harbor_skyrl_daytona_subtb",
    "tinker_provider_sft",
    "tinker_provider_tb",
    "tinker_provider_subtb",
    "release_grade_mixed_atif",
    "producer_data_reward_pipeline",
    "producer_fresh_provider_generation",
    "direct_fixed_logprob_subtb_replay",
    "contract_strictness",
]
EvidenceReceiptKind = Literal[
    "fresh_provider_generation",
    "direct_fixed_logprob_subtb_replay",
]
CleanupReceiptKind = Literal[
    "sky_status_clean",
    "json_empty_collection",
    "ec2_all_terminated",
    "daytona_zero",
    "tinker_checkpoints_deleted",
]
SecretScanKind = Literal["json_ok_zero_findings", "empty_text"]
BundleFileKind = Literal["release_manifest"]


def _validate_jsonish(value: Any, *, field_name: str) -> Any:
    require_json_serializable(value, field_name=field_name)
    _reject_non_finite(value, field_name=field_name)
    return value


def _reject_non_finite(value: Any, *, field_name: str) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain non-finite floats")
        return
    if isinstance(value, int | str):
        return
    if isinstance(value, list):
        for item in value:
            _reject_non_finite(item, field_name=field_name)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field_name} object keys must be non-empty strings")
            _reject_non_finite(item, field_name=field_name)
        return
    raise ValueError(f"{field_name} must be JSON-compatible")


class SourceProvenance(ContractModel):
    git_available: bool | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    git_error: str | None = None
    source_identity_kind: Literal[
        "git_commit",
        "source_tree_sha256",
        "source_bundle_sha256",
    ]
    source_identity: NonEmptyStr
    source_tree_sha256: Sha256 | None = None
    source_bundle_sha256: Sha256 | None = None
    source_tree_file_count: NonNegativeInt | None = None
    source_tree_total_bytes: NonNegativeInt | None = None
    source_tree_hash_scope: str | None = None
    source_tree_root: str | None = None

    @model_validator(mode="after")
    def identity_matches_declared_kind(self) -> "SourceProvenance":
        expected = getattr(self, self.source_identity_kind)
        if expected is None or self.source_identity != expected:
            raise ValueError("source_identity must match source_identity_kind")
        return self


class TrainingReceiptRunner(ContractModel):
    type: RunnerType
    configured_type: str | None = None


class TrainingReceiptDataset(ContractModel):
    path: str | None = None
    resolved_path: str | None = None
    row_count: NonNegativeInt
    train_rows: NonNegativeInt
    sha256: Sha256 | None = None
    file_manifest_sha256: Sha256 | None = None
    file_count: NonNegativeInt | None = None
    total_bytes: NonNegativeInt | None = None
    max_examples: NonNegativeInt | None = None
    max_target_tokens: PositiveInt | None = None
    unique_atif_ids_consumed: list[NonEmptyStr] | None = None
    unique_atif_ids_consumed_count: NonNegativeInt | None = None
    reward_stats: dict[str, Any] | None = None
    available_reward_evidence: dict[str, Any] | None = None
    consumed_reward_evidence: dict[str, Any] | None = None
    reward_evidence: dict[str, Any] | None = None
    consumed_reward_sources: list[NonEmptyStr] | None = None
    atif_manifest_count_exported: NonNegativeInt | None = None

    @field_validator(
        "reward_stats",
        "available_reward_evidence",
        "consumed_reward_evidence",
        "reward_evidence",
    )
    @classmethod
    def dynamic_dicts_are_jsonish(
        cls, value: dict[str, Any] | None, info: object
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return _validate_jsonish(
            value,
            field_name=getattr(info, "field_name", "dataset_dynamic_dict"),
        )

    @model_validator(mode="after")
    def counts_are_coherent(self) -> "TrainingReceiptDataset":
        if self.unique_atif_ids_consumed is not None:
            observed = len(set(self.unique_atif_ids_consumed))
            if self.unique_atif_ids_consumed_count is None:
                raise ValueError(
                    "unique_atif_ids_consumed_count is required with "
                    "unique_atif_ids_consumed"
                )
            if observed != self.unique_atif_ids_consumed_count:
                raise ValueError(
                    "unique_atif_ids_consumed_count must match unique consumed ids"
                )
            if observed > self.row_count:
                raise ValueError(
                    "unique_atif_ids_consumed_count must not exceed row_count"
                )
        return self


class TrainingReceiptObjective(ContractModel):
    loss_type: ObjectiveLossType
    loss_name: NonEmptyStr
    advantage_estimator: str | None = None
    advantages: list[float] | None = None
    initial_logZ: float | None = None
    final_logZ: float | None = None
    tb_max_residual: float | None = None
    subtb_lambda: float | None = None
    entropy_coef: float | None = None

    @field_validator(
        "initial_logZ",
        "final_logZ",
        "tb_max_residual",
        "subtb_lambda",
        "entropy_coef",
    )
    @classmethod
    def optional_floats_are_finite(
        cls, value: float | None, info: object
    ) -> float | None:
        if value is not None and not math.isfinite(value):
            field_name = getattr(info, "field_name", "objective_float")
            raise ValueError(f"{field_name} must be finite")
        return value

    @field_validator("advantages")
    @classmethod
    def advantages_are_finite(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        for advantage in value:
            if not math.isfinite(advantage):
                raise ValueError("advantages must be finite")
        return value

    @model_validator(mode="after")
    def loss_type_matches_loss_name(self) -> "TrainingReceiptObjective":
        expected = {
            "sft": {"sft", "sft_mle"},
            "tb": {"tb", "gfn_tb"},
            "subtb": {"subtb", "gfn_subtb"},
            "drgrpo": {"drgrpo"},
        }[self.loss_type]
        if self.loss_name not in expected:
            raise ValueError("loss_name must match loss_type")
        if self.loss_type in {"tb", "subtb"} and (
            self.initial_logZ is None or self.final_logZ is None
        ):
            raise ValueError("TB/SubTB receipts require initial_logZ and final_logZ")
        return self


class TrainingReceiptPolicy(ContractModel):
    model_name: NonEmptyStr
    backend: str | None = None
    checkpoint_path: str | None = None
    dtype: str | None = None
    lora_config: dict[str, Any] | None = None

    @field_validator("lora_config")
    @classmethod
    def lora_config_is_jsonish(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return _validate_jsonish(value, field_name="lora_config")


class TrainingReceiptOptimization(ContractModel):
    steps: PositiveInt
    batch_size: PositiveInt | None = None
    train_batch_size: PositiveInt | None = None
    policy_mini_batch_size: PositiveInt | None = None
    policy_lr: NonNegativeFloat | None = None
    logZ_lr: NonNegativeFloat | None = None
    max_grad_norm: NonNegativeFloat | None = None

    @field_validator("policy_lr", "logZ_lr", "max_grad_norm")
    @classmethod
    def optimization_floats_are_finite(
        cls, value: float | None, info: object
    ) -> float | None:
        if value is not None and not math.isfinite(value):
            field_name = getattr(info, "field_name", "optimization_float")
            raise ValueError(f"{field_name} must be finite")
        return value


class TrainingReceiptArtifacts(ContractModel):
    training_receipt: str | None = None
    dataset_manifest: str | None = None
    checkpoints: list[NonEmptyStr]
    final_checkpoint: NonEmptyStr
    policy_weight_change_report: str | None = None
    policy_weight_change_report_sha256: Sha256 | None = None
    weight_change_report: str | None = None
    weight_change_report_sha256: Sha256 | None = None
    checkpoint_state: str | None = None
    replay_buffer_state: str | None = None
    replay_buffer_state_sha256: Sha256 | None = None
    tinker_capability_probe: str | None = None
    tinker_capability_probe_sha256: Sha256 | None = None
    final_sampler_sample_proof: str | None = None
    final_sampler_sample_proof_sha256: Sha256 | None = None
    final_tinker_remote_state: dict[str, Any] | None = None
    initial_tinker_sampler_weights: dict[str, Any] | None = None
    final_tinker_sampler_weights: dict[str, Any] | None = None

    @field_validator(
        "final_tinker_remote_state",
        "initial_tinker_sampler_weights",
        "final_tinker_sampler_weights",
    )
    @classmethod
    def provider_states_are_jsonish(
        cls, value: dict[str, Any] | None, info: object
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return _validate_jsonish(value, field_name=getattr(info, "field_name", "state"))

    @model_validator(mode="after")
    def final_checkpoint_is_declared(self) -> "TrainingReceiptArtifacts":
        if self.final_checkpoint not in self.checkpoints:
            raise ValueError("final_checkpoint must be listed in checkpoints")
        return self


class TrainingReceiptWeightChange(ContractModel):
    schema_version: Literal["synthstats.policy_weight_change_report.v1"] | None = None
    proof_type: str | None = None
    changed_tensor_count: PositiveInt
    weight_delta_l2: Annotated[float, Field(gt=0.0)]
    tensor_count: PositiveInt | None = None
    changed_file_count: NonNegativeInt | None = None
    structural_change_count: NonNegativeInt | None = None
    nonfinite_tensor_count: NonNegativeInt | None = None
    exact_weight_delta: bool | None = None
    checksum_delta_abs: NonNegativeFloat | None = None
    initial_loss: float | None = None
    final_loss: float | None = None
    dataset_path: str | None = None
    num_examples: PositiveInt | None = None
    trainable_param_count: PositiveInt | None = None

    @field_validator(
        "weight_delta_l2",
        "checksum_delta_abs",
        "initial_loss",
        "final_loss",
    )
    @classmethod
    def weight_floats_are_finite(
        cls, value: float | None, info: object
    ) -> float | None:
        if value is not None and not math.isfinite(value):
            field_name = getattr(info, "field_name", "weight_float")
            raise ValueError(f"{field_name} must be finite")
        return value


class TrainingReceiptProvider(ContractModel):
    name: NonEmptyStr
    capability_probe: dict[str, Any] | None = None
    lora_config: dict[str, Any] | None = None
    final_sampler_sample_proof: dict[str, Any] | None = None

    @field_validator("capability_probe", "lora_config", "final_sampler_sample_proof")
    @classmethod
    def provider_payloads_are_jsonish(
        cls, value: dict[str, Any] | None, info: object
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return _validate_jsonish(
            value,
            field_name=getattr(info, "field_name", "provider_payload"),
        )


class TrainingReceipt(ContractModel):
    schema_version: Literal["synthstats.training_receipt.v1"] = (
        TRAINING_RECEIPT_SCHEMA_VERSION
    )
    runner: TrainingReceiptRunner
    dataset: TrainingReceiptDataset
    objective: TrainingReceiptObjective
    policy: TrainingReceiptPolicy
    optimization: TrainingReceiptOptimization
    metrics: dict[str, Any]
    artifacts: TrainingReceiptArtifacts
    weight_change: TrainingReceiptWeightChange
    provenance: SourceProvenance | None = None
    environment: dict[str, Any] | None = None
    generator: dict[str, Any] | None = None
    checkpoint_state: dict[str, Any] | None = None
    provider: TrainingReceiptProvider | None = None

    @field_validator("metrics", "environment", "generator", "checkpoint_state")
    @classmethod
    def dynamic_payloads_are_jsonish(cls, value: Any, info: object) -> Any:
        if value is None:
            return None
        return _validate_jsonish(
            value, field_name=getattr(info, "field_name", "payload")
        )

    @model_validator(mode="after")
    def runner_specific_evidence_is_present(self) -> "TrainingReceipt":
        runner_type = self.runner.type
        if (
            runner_type in {"sft", "offline_rescore", "tinker", "grpo_tinker"}
            and not self.dataset.path
        ):
            raise ValueError(f"{runner_type} receipts require dataset.path")
        if runner_type == "sft" and self.dataset.sha256 is None:
            raise ValueError("SFT receipts require dataset.sha256")
        if runner_type == "skyrl":
            if self.generator is None:
                raise ValueError("SkyRL receipts require generator evidence")
            if self.environment is None:
                raise ValueError("SkyRL receipts require environment evidence")
        if runner_type == "tinker":
            if self.provider is None or self.provider.name != "tinker":
                raise ValueError("Tinker receipts require provider.name='tinker'")
            if self.artifacts.final_tinker_remote_state is None:
                raise ValueError("Tinker receipts require final_tinker_remote_state")
        if runner_type == "grpo_tinker":
            if self.provider is None or self.provider.name != "tinker":
                raise ValueError("GRPO Tinker receipts require provider.name='tinker'")
            if self.artifacts.initial_tinker_sampler_weights is None:
                raise ValueError(
                    "GRPO Tinker receipts require initial_tinker_sampler_weights"
                )
            if self.artifacts.final_tinker_sampler_weights is None:
                raise ValueError(
                    "GRPO Tinker receipts require final_tinker_sampler_weights"
                )
        return self


class BundleClaim(ContractModel):
    level: NonEmptyStr
    description: NonEmptyStr | None = None


class BundleFile(ContractModel):
    name: NonEmptyStr
    path: NonEmptyStr
    sha256: Sha256
    sha256_sidecar: str | None = None
    max_hash_bytes: PositiveInt | None = None
    kind: BundleFileKind | None = None
    capabilities: list[BundleCapability] = Field(default_factory=list)
    validation: NonEmptyStr | None = None
    validation_sha256: Sha256 | None = None
    min_count_exported: PositiveInt | None = None
    min_count_unique_atif_ids: PositiveInt | None = None
    min_runtime_counts: dict[NonEmptyStr, PositiveInt] = Field(default_factory=dict)
    min_split_counts: dict[NonEmptyStr, PositiveInt] = Field(default_factory=dict)
    min_reward_sources: PositiveInt | None = None
    require_zero_dropped: bool = False
    require_nonzero_reward_variance: bool = False
    require_data_ref_digests: bool = False
    require_env_spec_digests: bool = False
    require_trial_artifacts_match_count: bool = False

    @model_validator(mode="after")
    def sidecar_hashes_cover_optional_validation(self) -> "BundleFile":
        if self.validation is not None and self.validation_sha256 is None:
            raise ValueError("validation_sha256 is required with validation")
        return self


class TrainingReceiptValidationConfig(ContractModel):
    require_objective: str | None = None
    min_train_rows: PositiveInt | None = None
    min_unique_rows: PositiveInt | None = None
    require_unique_rows_equal_train_rows: bool = False
    require_reward_variance: bool = False
    require_nonzero_grad: bool = False
    require_policy_delta: bool = False
    require_logz_delta: bool = False
    require_eos_coverage: bool = False
    require_terminal_eos: bool = False
    require_entropy: bool = False
    require_target_eos: bool = False
    min_reward_sources: PositiveInt | None = None
    min_reward_components: PositiveInt | None = None
    min_reward_component_sources: PositiveInt | None = None
    require_scalarization: bool = False
    require_non_docker_sandbox_provider: bool = False
    require_training_log_reward_consumed: bool = False
    require_rollout_logprobs: bool = False
    require_harbor_artifacts: bool = False
    require_dataset_hash: bool = False
    require_final_checkpoint: bool = False
    require_replay_buffer: bool = False
    require_training_trajectory_adapter: bool = False
    require_tinker_provider_proof: bool = False
    require_source_provenance: bool = False
    forbidden_env_classes: list[NonEmptyStr] = Field(default_factory=list)
    epsilon: Annotated[float, Field(gt=0.0)] = 1e-12

    @field_validator("epsilon")
    @classmethod
    def epsilon_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("epsilon must be finite")
        return value


_CAPABILITY_VALIDATION_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "sft_warmstart": {
        "require_objective": "sft",
        "require_dataset_hash": True,
        "require_final_checkpoint": True,
        "require_policy_delta": True,
        "require_target_eos": True,
    },
    "offline_current_policy_tb_rescore": {
        "require_objective": "gfn_tb",
        "require_dataset_hash": True,
        "require_final_checkpoint": True,
        "require_policy_delta": True,
        "require_logz_delta": True,
        "require_training_log_reward_consumed": True,
        "require_training_trajectory_adapter": True,
    },
    "offline_current_policy_subtb_rescore": {
        "require_objective": "gfn_subtb",
        "require_dataset_hash": True,
        "require_final_checkpoint": True,
        "require_policy_delta": True,
        "require_eos_coverage": True,
        "require_terminal_eos": True,
        "require_rollout_logprobs": True,
        "require_training_log_reward_consumed": True,
        "require_training_trajectory_adapter": True,
    },
    "live_harbor_skyrl_daytona_tb": {
        "require_objective": "gfn_tb",
        "require_final_checkpoint": True,
        "require_harbor_artifacts": True,
        "require_logz_delta": True,
        "require_non_docker_sandbox_provider": True,
        "require_nonzero_grad": True,
        "require_policy_delta": True,
        "require_rollout_logprobs": True,
        "require_source_provenance": True,
        "require_training_log_reward_consumed": True,
    },
    "live_harbor_skyrl_daytona_subtb": {
        "require_objective": "gfn_subtb",
        "require_eos_coverage": True,
        "require_final_checkpoint": True,
        "require_harbor_artifacts": True,
        "require_non_docker_sandbox_provider": True,
        "require_nonzero_grad": True,
        "require_policy_delta": True,
        "require_rollout_logprobs": True,
        "require_source_provenance": True,
        "require_terminal_eos": True,
        "require_training_log_reward_consumed": True,
    },
    "tinker_provider_sft": {
        "require_objective": "sft",
        "require_dataset_hash": True,
        "require_final_checkpoint": True,
        "require_policy_delta": True,
        "require_source_provenance": True,
        "require_tinker_provider_proof": True,
    },
    "tinker_provider_tb": {
        "require_objective": "gfn_tb",
        "require_dataset_hash": True,
        "require_final_checkpoint": True,
        "require_logz_delta": True,
        "require_policy_delta": True,
        "require_source_provenance": True,
        "require_tinker_provider_proof": True,
        "require_training_log_reward_consumed": True,
    },
    "tinker_provider_subtb": {
        "require_objective": "gfn_subtb",
        "require_dataset_hash": True,
        "require_eos_coverage": True,
        "require_final_checkpoint": True,
        "require_policy_delta": True,
        "require_source_provenance": True,
        "require_terminal_eos": True,
        "require_tinker_provider_proof": True,
        "require_training_log_reward_consumed": True,
    },
}


def get_capability_validation_requirements(
    capability: BundleCapability,
) -> dict[str, Any]:
    return dict(_CAPABILITY_VALIDATION_REQUIREMENTS.get(capability, {}))


def _config_satisfies_requirement(
    config: TrainingReceiptValidationConfig,
    field_name: str,
    expected: Any,
) -> bool:
    observed = getattr(config, field_name)
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, str):
        return observed == expected
    if isinstance(expected, int):
        return (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed >= expected
        )
    return observed == expected


class BundleTrainingReceiptEntry(ContractModel):
    name: NonEmptyStr
    path: NonEmptyStr
    sha256: Sha256
    capabilities: list[BundleCapability] = Field(default_factory=list)
    validation: str | None = None
    validation_sha256: Sha256 | None = None
    validation_config: TrainingReceiptValidationConfig = Field(
        default_factory=TrainingReceiptValidationConfig
    )
    weight_report: str | None = None
    weight_report_sha256: Sha256 | None = None
    checkpoint_state: str | None = None
    checkpoint_state_sha256: Sha256 | None = None
    min_canonical_reward_sources: PositiveInt | None = None
    min_unique_rows: PositiveInt | None = None
    require_truth_owner: Literal["verifier", "heuristic", "auditor"] | None = None
    require_finite_training_reward_stats: bool = False

    @model_validator(mode="after")
    def sidecar_hashes_cover_optional_paths(self) -> "BundleTrainingReceiptEntry":
        if self.validation is not None and self.validation_sha256 is None:
            raise ValueError("validation_sha256 is required with validation")
        if self.weight_report is not None and self.weight_report_sha256 is None:
            raise ValueError("weight_report_sha256 is required with weight_report")
        if self.checkpoint_state is not None and self.checkpoint_state_sha256 is None:
            raise ValueError(
                "checkpoint_state_sha256 is required with checkpoint_state"
            )
        return self

    @model_validator(mode="after")
    def capabilities_require_validation_flags(self) -> "BundleTrainingReceiptEntry":
        missing: list[str] = []
        for capability in self.capabilities:
            requirements = get_capability_validation_requirements(capability)
            for field_name, expected in requirements.items():
                if not _config_satisfies_requirement(
                    self.validation_config,
                    field_name,
                    expected,
                ):
                    missing.append(
                        f"{capability} requires validation_config.{field_name}="
                        f"{expected!r}"
                    )
        if self.validation_config.require_policy_delta and self.weight_report is None:
            missing.append("policy-delta capabilities require weight_report")
        if (
            self.validation_config.require_tinker_provider_proof
            and self.checkpoint_state is None
        ):
            missing.append("Tinker provider capabilities require checkpoint_state")
        if missing:
            raise ValueError("; ".join(missing))
        return self


class SemanticValidationSettings(ContractModel):
    pymc_draws: PositiveInt
    pymc_tune: PositiveInt
    pymc_timeout_sec: PositiveInt


class BundleEvidenceReceiptEntry(ContractModel):
    name: NonEmptyStr
    kind: EvidenceReceiptKind
    path: NonEmptyStr
    sha256: Sha256
    capabilities: list[BundleCapability] = Field(default_factory=list)
    validation: NonEmptyStr | None = None
    validation_sha256: Sha256 | None = None
    min_count_trials: PositiveInt | None = None
    min_stage_f_validated: PositiveInt | None = None
    require_provider_response_id: bool = False
    require_provider_usage: bool = False
    require_request_id_coverage: bool = False
    require_zero_provider_failures: bool = False
    expected_provider_backend: NonEmptyStr | None = None
    expected_completion_model: NonEmptyStr | None = None
    derived_manifest: NonEmptyStr | None = None
    derived_manifest_sha256: Sha256 | None = None
    steps: NonEmptyStr | None = None
    steps_sha256: Sha256 | None = None
    required_steps: list[NonEmptyStr] = Field(default_factory=list)
    min_row_count: PositiveInt | None = None
    min_response_token_count: PositiveInt | None = None
    min_checkpoint_count: PositiveInt | None = None
    require_eos_logprobs: bool = False
    require_terminal_eos: bool = False
    require_objective_marker: bool = False
    require_fixed_logprob_boundary: bool = False
    expected_reward_source: NonEmptyStr | None = None
    expected_truth_owner: NonEmptyStr | None = None
    expected_source_rollout_tensor_artifact_sha256: Sha256 | None = None
    semantic_validation_settings: SemanticValidationSettings | None = None

    @model_validator(mode="after")
    def evidence_kind_is_fail_closed(self) -> "BundleEvidenceReceiptEntry":
        if self.kind == "fresh_provider_generation":
            missing: list[str] = []
            if self.validation is None:
                missing.append("validation")
            if self.validation is not None and self.validation_sha256 is None:
                missing.append("validation_sha256")
            if self.min_count_trials is None:
                missing.append("min_count_trials")
            if self.min_stage_f_validated is None:
                missing.append("min_stage_f_validated")
            if not self.require_provider_response_id:
                missing.append("require_provider_response_id")
            if not self.require_provider_usage:
                missing.append("require_provider_usage")
            if not self.require_request_id_coverage:
                missing.append("require_request_id_coverage")
            if not self.require_zero_provider_failures:
                missing.append("require_zero_provider_failures")
            if self.expected_provider_backend is None:
                missing.append("expected_provider_backend")
            if self.expected_completion_model is None:
                missing.append("expected_completion_model")
            if missing:
                raise ValueError(
                    "fresh_provider_generation evidence requires: " + ", ".join(missing)
                )
        if self.kind == "direct_fixed_logprob_subtb_replay":
            missing = []
            if self.validation is None:
                missing.append("validation")
            if self.validation is not None and self.validation_sha256 is None:
                missing.append("validation_sha256")
            if self.derived_manifest is None:
                missing.append("derived_manifest")
            if (
                self.derived_manifest is not None
                and self.derived_manifest_sha256 is None
            ):
                missing.append("derived_manifest_sha256")
            if self.steps is None:
                missing.append("steps")
            if self.steps is not None and self.steps_sha256 is None:
                missing.append("steps_sha256")
            if not self.required_steps:
                missing.append("required_steps")
            if self.min_row_count is None:
                missing.append("min_row_count")
            if self.min_response_token_count is None:
                missing.append("min_response_token_count")
            if self.min_checkpoint_count is None:
                missing.append("min_checkpoint_count")
            if not self.require_eos_logprobs:
                missing.append("require_eos_logprobs")
            if not self.require_terminal_eos:
                missing.append("require_terminal_eos")
            if not self.require_objective_marker:
                missing.append("require_objective_marker")
            if not self.require_fixed_logprob_boundary:
                missing.append("require_fixed_logprob_boundary")
            if self.expected_reward_source is None:
                missing.append("expected_reward_source")
            if self.expected_truth_owner is None:
                missing.append("expected_truth_owner")
            if self.expected_source_rollout_tensor_artifact_sha256 is None:
                missing.append("expected_source_rollout_tensor_artifact_sha256")
            if missing:
                raise ValueError(
                    "direct_fixed_logprob_subtb_replay evidence requires: "
                    + ", ".join(missing)
                )
        return self


class SecretScanReceiptEntry(ContractModel):
    name: NonEmptyStr
    path: NonEmptyStr
    sha256: Sha256
    kind: SecretScanKind = "json_ok_zero_findings"


class CleanupReceiptEntry(ContractModel):
    name: NonEmptyStr
    path: NonEmptyStr
    sha256: Sha256
    kind: CleanupReceiptKind
    field: NonEmptyStr | None = None
    run_id: NonEmptyStr
    scope: dict[str, Any]

    @field_validator("scope")
    @classmethod
    def scope_is_nonempty_jsonish(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("cleanup scope must not be empty")
        return _validate_jsonish(value, field_name="scope")


class TextScanEntry(ContractModel):
    name: NonEmptyStr
    path: NonEmptyStr | None = None
    paths: list[NonEmptyStr] = Field(default_factory=list)
    allow_skipped_files: bool = False

    @model_validator(mode="after")
    def has_at_least_one_path(self) -> "TextScanEntry":
        if self.path is None and not self.paths:
            raise ValueError("text scan entries require path or paths")
        return self


class PostTrainingAcceptanceBundle(ContractModel):
    schema_version: Literal["synthstats.posttraining_acceptance_bundle.v1"] = (
        POSTTRAINING_ACCEPTANCE_BUNDLE_SCHEMA_VERSION
    )
    claim: BundleClaim
    required_capabilities: list[BundleCapability]
    files: list[BundleFile]
    training_receipts: list[BundleTrainingReceiptEntry]
    evidence_receipts: list[BundleEvidenceReceiptEntry] = Field(default_factory=list)
    secret_scans: list[SecretScanReceiptEntry] = Field(default_factory=list)
    cleanup_receipts: list[CleanupReceiptEntry] = Field(default_factory=list)
    prohibited_marker_scans: list[TextScanEntry] = Field(default_factory=list)
    secret_text_scans: list[TextScanEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def required_capabilities_are_covered(self) -> "PostTrainingAcceptanceBundle":
        covered = {
            capability for file in self.files for capability in file.capabilities
        }
        covered.update(
            capability
            for receipt in self.training_receipts
            for capability in receipt.capabilities
        )
        covered.update(
            capability
            for receipt in self.evidence_receipts
            for capability in receipt.capabilities
        )
        missing = sorted(set(self.required_capabilities) - covered)
        if missing:
            raise ValueError(f"required_capabilities are not covered: {missing}")
        return self

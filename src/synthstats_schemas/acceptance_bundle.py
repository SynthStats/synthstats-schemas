"""Post-training acceptance bundle contracts (synthstats.posttraining_acceptance_bundle.v1).

These schemas gate post-training releases: capability requirements, evidence
receipts, secret scans, cleanup receipts, and the top-level bundle that ties
them together. They were split out of receipt.py, which keeps only the
synthstats.training_receipt.v1 contract.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal

from pydantic import (
    Field,
    FiniteFloat,
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
    model_validator,
)

from synthstats_schemas.base import ContractModel, NonEmptyStr, Sha256
from synthstats_schemas.receipt import (
    JsonObject,
    TrainableRewardSourceName,
)
from synthstats_schemas.reward import TruthOwner, ensure_reward_source_trainable

POSTTRAINING_ACCEPTANCE_BUNDLE_SCHEMA_VERSION: Final[
    Literal["synthstats.posttraining_acceptance_bundle.v1"]
] = "synthstats.posttraining_acceptance_bundle.v1"

TrainingReceiptCapability = Literal[
    "sft_warmstart",
    "offline_current_policy_tb_rescore",
    "offline_current_policy_subtb_rescore",
    "live_harbor_skyrl_daytona_tb",
    "live_harbor_skyrl_daytona_subtb",
    "tinker_provider_sft",
    "tinker_provider_tb",
    "tinker_provider_subtb",
]
BundleCapability = Literal[
    TrainingReceiptCapability,
    "bedrock_heuristic_review",
    "independent_review",
    "release_grade_mixed_atif",
    "producer_data_reward_pipeline",
    "producer_fresh_provider_generation",
    "direct_fixed_logprob_subtb_replay",
    "contract_strictness",
]
EvidenceReceiptKind = Literal[
    "fresh_provider_generation",
    "bedrock_heuristic_review",
    "oracle_review",
    "independent_review",
    "direct_fixed_logprob_subtb_replay",
]
ReviewVerdict = Literal["GO", "NO-GO"]
CleanupReceiptKind = Literal[
    "sky_status_clean",
    "json_empty_collection",
    "ec2_all_terminated",
    "daytona_zero",
    "tinker_checkpoints_deleted",
]
SecretScanKind = Literal["json_ok_zero_findings", "empty_text"]
BundleFileKind = Literal["release_manifest"]


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
    epsilon: Annotated[FiniteFloat, Field(gt=0.0)] = 1e-12


_CAPABILITY_VALIDATION_REQUIREMENTS: dict[TrainingReceiptCapability, dict[str, Any]] = {
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
    capability: TrainingReceiptCapability,
) -> dict[str, Any]:
    try:
        requirements = _CAPABILITY_VALIDATION_REQUIREMENTS[capability]
    except KeyError:
        raise ValueError(
            f"no validation requirements are declared for capability "
            f"{capability!r}: a bundle entry declaring it must be refused, not "
            "silently admitted with no requirements; tier the capability in "
            "_CAPABILITY_VALIDATION_REQUIREMENTS or retire it"
        ) from None
    return dict(requirements)


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
    capabilities: list[TrainingReceiptCapability] = Field(default_factory=list)
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


class _BundleEvidenceReceiptEntryBase(ContractModel):
    name: NonEmptyStr
    path: NonEmptyStr
    sha256: Sha256
    capabilities: list[BundleCapability] = Field(default_factory=list)
    semantic_validation_settings: SemanticValidationSettings | None = None


class FreshProviderGenerationEvidence(_BundleEvidenceReceiptEntryBase):
    kind: Literal["fresh_provider_generation"]
    validation: NonEmptyStr
    validation_sha256: Sha256
    min_count_trials: PositiveInt
    min_stage_f_validated: PositiveInt
    require_provider_response_id: Literal[True]
    require_provider_usage: Literal[True]
    require_request_id_coverage: Literal[True]
    require_zero_provider_failures: Literal[True]
    expected_provider_backend: NonEmptyStr
    expected_completion_model: NonEmptyStr


class BedrockHeuristicReviewEvidence(_BundleEvidenceReceiptEntryBase):
    kind: Literal["bedrock_heuristic_review"]
    model_id: NonEmptyStr
    route: NonEmptyStr
    region: NonEmptyStr
    request_id: NonEmptyStr
    prompt_sha256: Sha256
    output_sha256: Sha256
    heuristic_only: Literal[True]
    classification: NonEmptyStr
    review_verdict: Literal["GO"]
    unresolved_p0_count: Literal[0]
    unresolved_p1_count: Literal[0]


class OracleReviewEvidence(_BundleEvidenceReceiptEntryBase):
    kind: Literal["oracle_review"]
    model_id: NonEmptyStr
    route: NonEmptyStr
    response_id: NonEmptyStr | None = None
    output_sha256: Sha256
    heuristic_only: Literal[True]
    classification: NonEmptyStr
    review_verdict: Literal["GO"]
    unresolved_p0_count: Literal[0]
    unresolved_p1_count: Literal[0]


class IndependentReviewEvidence(_BundleEvidenceReceiptEntryBase):
    kind: Literal["independent_review"]
    model_id: NonEmptyStr
    route: NonEmptyStr
    response_id: NonEmptyStr | None = None
    output_sha256: Sha256
    heuristic_only: Literal[True]
    classification: NonEmptyStr
    review_verdict: Literal["GO"]
    unresolved_p0_count: Literal[0]
    unresolved_p1_count: Literal[0]


class DirectFixedLogprobSubTBReplayEvidence(_BundleEvidenceReceiptEntryBase):
    kind: Literal["direct_fixed_logprob_subtb_replay"]
    validation: NonEmptyStr
    validation_sha256: Sha256
    derived_manifest: NonEmptyStr
    derived_manifest_sha256: Sha256
    steps: NonEmptyStr
    steps_sha256: Sha256
    required_steps: list[NonEmptyStr] = Field(min_length=1)
    min_row_count: PositiveInt
    min_response_token_count: PositiveInt
    min_checkpoint_count: PositiveInt
    require_eos_logprobs: Literal[True]
    require_terminal_eos: Literal[True]
    require_objective_marker: Literal[True]
    require_fixed_logprob_boundary: Literal[True]
    expected_reward_source: TrainableRewardSourceName
    expected_truth_owner: Annotated[TruthOwner, Field(strict=False)]
    expected_source_rollout_tensor_artifact_sha256: Sha256

    @model_validator(mode="after")
    def truth_owner_matches_catalog(self) -> "DirectFixedLogprobSubTBReplayEvidence":
        expected = ensure_reward_source_trainable(self.expected_reward_source)
        if self.expected_truth_owner is not expected.truth_owner:
            raise ValueError(
                "expected_truth_owner must match expected_reward_source catalog owner"
            )
        return self


_BundleEvidenceReceiptUnion = Annotated[
    FreshProviderGenerationEvidence
    | BedrockHeuristicReviewEvidence
    | OracleReviewEvidence
    | IndependentReviewEvidence
    | DirectFixedLogprobSubTBReplayEvidence,
    Field(discriminator="kind"),
]

BundleEvidenceReceiptEntry = TypeAdapter(_BundleEvidenceReceiptUnion)


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
    scope: JsonObject = Field(min_length=1)


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
    # A bundle names at least one capability it needs covered: an empty
    # declaration would make the coverage gate pass whatever the bundle holds.
    required_capabilities: list[BundleCapability] = Field(min_length=1)
    files: list[BundleFile]
    training_receipts: list[BundleTrainingReceiptEntry]
    evidence_receipts: list[_BundleEvidenceReceiptUnion] = Field(default_factory=list)
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

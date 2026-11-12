"""Training receipt contract (synthstats.training_receipt.v1).

The post-training acceptance bundle contract
(synthstats.posttraining_acceptance_bundle.v1) lives in
synthstats_schemas.acceptance_bundle and is re-exported from the package root.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
    with_config,
)
from typing_extensions import TypedDict

from synthstats_schemas.base import (
    ContractModel,
    NonEmptyStr,
    Sha256,
)
from synthstats_schemas.canonical import canonical_json_bytes
from synthstats_schemas.reward import (
    ensure_reward_source_registered,
    ensure_reward_source_trainable,
)

def _json_object(value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError("must be a JSON object")

    def check(item: Any) -> None:
        if item is None or isinstance(item, bool | int | str):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("must not contain non-finite floats")
            return
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("object keys must be non-empty strings")
                check(child)
            return
        raise ValueError("must be JSON-compatible")

    check(value)
    return value


JsonObject = Annotated[dict[str, Any], AfterValidator(_json_object)]

TRAINING_RECEIPT_SCHEMA_VERSION: Final[Literal["synthstats.training_receipt.v1"]] = (
    "synthstats.training_receipt.v1"
)
POLICY_WEIGHT_CHANGE_REPORT_SCHEMA_VERSION: Final[
    Literal["synthstats.policy_weight_change_report.v1"]
] = "synthstats.policy_weight_change_report.v1"


def _registered_reward_source(name: str) -> str:
    ensure_reward_source_registered(name)
    return name


def _trainable_reward_source(name: str) -> str:
    ensure_reward_source_trainable(name)
    return name


RegisteredRewardSourceName = Annotated[
    NonEmptyStr, AfterValidator(_registered_reward_source)
]
TrainableRewardSourceName = Annotated[
    NonEmptyStr, AfterValidator(_trainable_reward_source)
]

_REWARD_CONSUMPTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "consumed_reward_evidence",
        "consumed_reward_sources",
        "reward_component_source_counts",
        "reward_source_counts",
    }
)

RunnerType = Literal["sft", "offline_rescore", "skyrl", "tinker"]
ObjectiveLossType = Literal["sft", "tb", "subtb"]

_RUNNER_OBJECTIVES: Final[dict[RunnerType, frozenset[ObjectiveLossType]]] = {
    "sft": frozenset({"sft"}),
    "offline_rescore": frozenset({"tb", "subtb"}),
    "skyrl": frozenset({"tb", "subtb"}),
    "tinker": frozenset({"sft", "tb", "subtb"}),
}


@with_config(ConfigDict(extra="forbid"))
class TrainingReceiptManifestEntry(TypedDict):
    sha256: Sha256
    size_bytes: NonNegativeInt


def _reject_alternate_reward_consumption(value: Any, *, field_name: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_alternate_reward_consumption(
                item,
                field_name=f"{field_name}[{index}]",
            )
        return
    if not isinstance(value, dict):
        return
    reserved = _REWARD_CONSUMPTION_KEYS.intersection(value)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(
            f"{field_name} contains reserved reward-consumption fields: {names}"
        )
    for key, item in value.items():
        _reject_alternate_reward_consumption(
            item,
            field_name=f"{field_name}.{key}",
        )


def _validate_canonical_relative_posix_path(value: str, *, field_name: str) -> str:
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL")
    if (
        value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or any(segment in {"", ".", ".."} for segment in value.split("/"))
    ):
        raise ValueError(f"{field_name} must be a canonical POSIX-relative path")
    return value


def _canonical_relative_posix_path(value: str) -> str:
    return _validate_canonical_relative_posix_path(
        value, field_name="path"
    )


def _configured_path(value: str) -> str:
    if "\x00" in value:
        raise ValueError("path must not contain NUL")
    if Path(value).is_absolute():
        return value
    return _validate_canonical_relative_posix_path(value, field_name="path")


def _absolute_path(value: str) -> str:
    if "\x00" in value or not Path(value).is_absolute():
        raise ValueError("path must be an absolute NUL-free path")
    return value


CanonicalPosixPath = Annotated[NonEmptyStr, AfterValidator(_canonical_relative_posix_path)]
ConfiguredPath = Annotated[NonEmptyStr, AfterValidator(_configured_path)]
AbsolutePath = Annotated[NonEmptyStr, AfterValidator(_absolute_path)]


def _manifest_sha256(manifest: dict[str, TrainingReceiptManifestEntry]) -> str:
    payload = canonical_json_bytes(manifest)
    return hashlib.sha256(payload).hexdigest()


class SFTSplitBinding(ContractModel):
    path: ConfiguredPath
    resolved_path: AbsolutePath
    sha256: Sha256
    row_count: NonNegativeInt


@with_config(ConfigDict(extra="forbid"))
class SFTSplitBindings(TypedDict):
    train: SFTSplitBinding
    validation: SFTSplitBinding
    test: SFTSplitBinding


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


class TrainingReceiptConsumedRewardEvidence(ContractModel):
    reward_source_counts: dict[TrainableRewardSourceName, PositiveInt]
    reward_component_source_counts: (
        dict[TrainableRewardSourceName, PositiveInt] | None
    ) = None


class TrainingReceiptDataset(ContractModel):
    path: str | None = None
    resolved_path: str | None = None
    row_count: NonNegativeInt
    train_rows: NonNegativeInt
    sha256: Sha256 | None = None
    sft_split_bindings: SFTSplitBindings | None = None
    sft_release_manifest_path: ConfiguredPath | None = None
    resolved_sft_release_manifest_path: AbsolutePath | None = None
    sft_release_manifest_sha256: Sha256 | None = None
    sft_release_manifest_schema_version: Literal["synthstats.sft_release.v1"] | None = (
        None
    )
    sft_release_manifest_row_count: PositiveInt | None = None
    sft_release_manifest_split_group_count: PositiveInt | None = None
    file_manifest_sha256: Sha256 | None = None
    file_count: NonNegativeInt | None = None
    total_bytes: NonNegativeInt | None = None
    max_examples: NonNegativeInt | None = None
    max_target_tokens: PositiveInt | None = None
    unique_atif_ids_consumed: list[NonEmptyStr] | None = None
    unique_atif_ids_consumed_count: NonNegativeInt | None = None
    reward_stats: JsonObject | None = None
    available_reward_evidence: JsonObject | None = None
    consumed_reward_evidence: TrainingReceiptConsumedRewardEvidence | None = None
    reward_evidence: JsonObject | None = None
    consumed_reward_sources: list[TrainableRewardSourceName] | None = None
    atif_manifest_count_exported: NonNegativeInt | None = None

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

        evidence = self.consumed_reward_evidence
        sources = self.consumed_reward_sources
        if evidence is None and sources is None:
            return self
        if evidence is None or sources is None:
            raise ValueError(
                "consumed_reward_evidence and consumed_reward_sources must be "
                "declared together"
            )
        if len(sources) != len(set(sources)):
            raise ValueError("consumed_reward_sources must not contain duplicates")
        if set(evidence.reward_source_counts) != set(sources):
            raise ValueError(
                "consumed_reward_evidence.reward_source_counts keys must exactly "
                "match consumed_reward_sources"
            )
        if sum(evidence.reward_source_counts.values()) != self.train_rows:
            raise ValueError(
                "consumed_reward_evidence.reward_source_counts values must sum "
                "to train_rows"
            )
        composite_count = evidence.reward_source_counts.get(
            "heuristic.composite_reward", 0
        )
        if composite_count:
            component_counts = evidence.reward_component_source_counts
            if not component_counts:
                raise ValueError(
                    "consumed_reward_evidence.reward_component_source_counts is "
                    "required for heuristic.composite_reward"
                )
            if sum(component_counts.values()) < composite_count:
                raise ValueError(
                    "consumed_reward_evidence.reward_component_source_counts must "
                    "cover heuristic.composite_reward rows"
                )
        return self


class TrainingReceiptObjective(ContractModel):
    loss_type: ObjectiveLossType
    loss_name: NonEmptyStr
    advantage_estimator: str | None = None
    advantages: list[FiniteFloat] | None = None
    initial_logZ: FiniteFloat | None = None
    final_logZ: FiniteFloat | None = None
    tb_max_residual: FiniteFloat | None = None
    subtb_lambda: FiniteFloat | None = None
    entropy_coef: FiniteFloat | None = None

    @model_validator(mode="after")
    def loss_type_matches_loss_name(self) -> "TrainingReceiptObjective":
        expected = {
            "sft": {"sft", "sft_mle"},
            "tb": {"tb", "gfn_tb"},
            "subtb": {"subtb", "gfn_subtb"},
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
    lora_config: JsonObject | None = None


class TrainingReceiptOptimization(ContractModel):
    steps: PositiveInt
    batch_size: PositiveInt | None = None
    train_batch_size: PositiveInt | None = None
    policy_mini_batch_size: PositiveInt | None = None
    policy_lr: NonNegativeFloat | None = None
    logZ_lr: NonNegativeFloat | None = None
    max_grad_norm: NonNegativeFloat | None = None


class TrainingReceiptArtifacts(ContractModel):
    training_receipt: str | None = None
    dataset_manifest: str | None = None
    checkpoints: list[CanonicalPosixPath]
    final_checkpoint: CanonicalPosixPath
    final_manifest: dict[CanonicalPosixPath, TrainingReceiptManifestEntry] = Field(
        min_length=1
    )
    final_manifest_sha256: Sha256
    policy_weight_change_report: str | None = None
    policy_weight_change_report_sha256: Sha256 | None = None
    checkpoint_state: str | None = None
    replay_buffer_state: str | None = None
    replay_buffer_state_sha256: Sha256 | None = None
    tinker_capability_probe: str | None = None
    tinker_capability_probe_sha256: Sha256 | None = None
    final_sampler_sample_proof: str | None = None
    final_sampler_sample_proof_sha256: Sha256 | None = None
    final_tinker_remote_state: JsonObject | None = None
    initial_tinker_sampler_weights: JsonObject | None = None
    final_tinker_sampler_weights: JsonObject | None = None

    @model_validator(mode="after")
    def final_checkpoint_is_declared(self) -> "TrainingReceiptArtifacts":
        if self.final_checkpoint not in self.checkpoints:
            raise ValueError("final_checkpoint must be listed in checkpoints")
        expected = _manifest_sha256(self.final_manifest)
        if self.final_manifest_sha256 != expected:
            raise ValueError(
                "final_manifest_sha256 must match compact sorted final_manifest JSON"
            )
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
    initial_loss: FiniteFloat | None = None
    final_loss: FiniteFloat | None = None
    dataset_path: str | None = None
    num_examples: PositiveInt | None = None
    trainable_param_count: PositiveInt | None = None


class TrainingReceiptProvider(ContractModel):
    name: NonEmptyStr
    capability_probe: JsonObject | None = None
    lora_config: JsonObject | None = None
    final_sampler_sample_proof: JsonObject | None = None


class TrainingReceipt(ContractModel):
    schema_version: Literal["synthstats.training_receipt.v1"] = (
        TRAINING_RECEIPT_SCHEMA_VERSION
    )
    runner: TrainingReceiptRunner
    dataset: TrainingReceiptDataset
    objective: TrainingReceiptObjective
    policy: TrainingReceiptPolicy
    optimization: TrainingReceiptOptimization
    metrics: JsonObject
    artifacts: TrainingReceiptArtifacts
    weight_change: TrainingReceiptWeightChange
    provenance: SourceProvenance | None = None
    environment: JsonObject | None = None
    generator: JsonObject | None = None
    checkpoint_state: JsonObject | None = None
    provider: TrainingReceiptProvider | None = None

    @model_validator(mode="before")
    @classmethod
    def reward_consumption_uses_only_canonical_dataset_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        for field_name, field_value in value.items():
            if field_name != "dataset":
                _reject_alternate_reward_consumption(
                    field_value,
                    field_name=field_name,
                )
                continue
            if not isinstance(field_value, dict):
                continue
            for dataset_field, dataset_value in field_value.items():
                if dataset_field in {
                    "consumed_reward_evidence",
                    "consumed_reward_sources",
                }:
                    continue
                _reject_alternate_reward_consumption(
                    dataset_value,
                    field_name=f"dataset.{dataset_field}",
                )
        return value

    @model_validator(mode="after")
    def runner_specific_evidence_is_present(self) -> "TrainingReceipt":
        runner_type = self.runner.type
        loss_type = self.objective.loss_type
        if loss_type not in _RUNNER_OBJECTIVES[runner_type]:
            raise ValueError(
                f"runner {runner_type!r} does not support objective {loss_type!r}"
            )
        sft_lineage_fields = (
            "sft_release_manifest_path",
            "resolved_sft_release_manifest_path",
            "sft_release_manifest_sha256",
            "sft_release_manifest_schema_version",
            "sft_release_manifest_row_count",
            "sft_release_manifest_split_group_count",
        )
        sft_lineage_values = {
            field_name: getattr(self.dataset, field_name)
            for field_name in sft_lineage_fields
        }
        if loss_type == "sft":
            split_bindings = self.dataset.sft_split_bindings
            if split_bindings is None:
                raise ValueError("SFT receipts require dataset.sft_split_bindings")

            train_binding = split_bindings["train"]
            train_identity_fields = ("path", "resolved_path", "sha256")
            missing_train_identity = sorted(
                field_name
                for field_name in train_identity_fields
                if getattr(self.dataset, field_name) is None
            )
            if missing_train_identity:
                raise ValueError(
                    "SFT receipts require dataset train identity fields: "
                    + ", ".join(missing_train_identity)
                )
            mismatched_train_identity = sorted(
                field_name
                for field_name in train_identity_fields
                if getattr(self.dataset, field_name)
                != getattr(train_binding, field_name)
            )
            if mismatched_train_identity:
                raise ValueError(
                    "SFT dataset train identity must match "
                    "dataset.sft_split_bindings.train: "
                    + ", ".join(mismatched_train_identity)
                )
            if not (
                self.dataset.row_count == self.dataset.train_rows >= 1
                and self.dataset.train_rows <= train_binding.row_count
            ):
                raise ValueError(
                    "SFT dataset row_count and train_rows must be equal, positive, "
                    "and no greater than sft_split_bindings.train.row_count"
                )
            if (
                self.dataset.max_examples is not None
                and self.dataset.row_count > self.dataset.max_examples
            ):
                raise ValueError(
                    "SFT dataset row_count must not exceed dataset.max_examples"
                )

            missing = sorted(
                field_name
                for field_name, value in sft_lineage_values.items()
                if value is None
            )
            if missing:
                raise ValueError(
                    "SFT receipts require release-manifest lineage: "
                    + ", ".join(missing)
                )
            release_row_count = self.dataset.sft_release_manifest_row_count
            split_group_count = self.dataset.sft_release_manifest_split_group_count
            if release_row_count is None or split_group_count is None:
                raise ValueError("SFT receipts require release-manifest counts")
            split_binding_row_count = sum(
                split_bindings[split].row_count
                for split in ("train", "validation", "test")
            )
            if split_binding_row_count > release_row_count:
                raise ValueError(
                    "SFT dataset.sft_split_bindings row counts must not exceed "
                    "release row count"
                )
            if not (
                release_row_count
                >= self.dataset.row_count
                >= self.dataset.train_rows
                >= 1
            ):
                raise ValueError(
                    "SFT release and dataset counts must satisfy "
                    "release rows >= dataset rows >= training rows >= 1"
                )
            if split_group_count > release_row_count:
                raise ValueError(
                    "SFT release split-group count must not exceed release row count"
                )
        else:
            if self.dataset.sft_split_bindings is not None:
                raise ValueError(
                    "non-SFT receipts must not declare dataset.sft_split_bindings"
                )
            present = sorted(
                field_name
                for field_name, value in sft_lineage_values.items()
                if value is not None
            )
            if present:
                raise ValueError(
                    "non-SFT receipts must not declare SFT release-manifest lineage: "
                    + ", ".join(present)
                )
        if self.objective.loss_type != "sft":
            if self.dataset.train_rows < 1:
                raise ValueError(
                    "reward-consuming receipts require positive train_rows"
                )
            if (
                self.dataset.consumed_reward_evidence is None
                or self.dataset.consumed_reward_sources is None
            ):
                raise ValueError(
                    "reward-consuming receipts require canonical dataset reward "
                    "consumption evidence"
                )
        if (
            runner_type in {"sft", "offline_rescore", "tinker"}
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
        return self


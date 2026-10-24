"""Release manifest and lineage contracts."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import Field, TypeAdapter, field_validator, model_validator

from synthstats_schemas.base import (
    TRAINING_LOG_REWARD_MIN,
    ContractModel,
    JsonValue,
)
from synthstats_schemas.reward import ensure_reward_source_registered

ATIF_EXPORT_MANIFEST_SCHEMA_VERSION: Final[Literal["ppl_synth_data.atif_export.v1"]] = (
    "ppl_synth_data.atif_export.v1"
)
MIXED_ATIF_MANIFEST_SCHEMA_VERSION: Final[
    Literal["ppl_synth_data.mixed_atif_assembly.v1"]
] = "ppl_synth_data.mixed_atif_assembly.v1"
RICH_REWARD_ATIF_MANIFEST_SCHEMA_VERSION: Final[
    Literal["ppl_synth_data.rich_reward_atif_export.v1"]
] = "ppl_synth_data.rich_reward_atif_export.v1"
LIVE_RICH_REWARD_MANIFEST_SCHEMA_VERSION: Final[
    Literal["ppl_synth_data.live_rich_reward.v1"]
] = "ppl_synth_data.live_rich_reward.v1"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, Field(min_length=1)]


def _validate_safe_id(value: str, *, field_name: str) -> str:
    path = Path(value)
    if path.name != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a single safe path segment")
    return value


def _validate_relative_data_ref(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("data_ref must be safely run-relative")
    return value


def _validate_counts(
    counts: dict[str, int],
    *,
    expected_total: int,
    field_name: str,
) -> dict[str, int]:
    if not counts:
        raise ValueError(f"{field_name} must not be empty")
    total = 0
    for key, value in counts.items():
        if not key:
            raise ValueError(f"{field_name} keys must be non-empty")
        if value < 0:
            raise ValueError(f"{field_name} values must be non-negative")
        total += value
    if total != expected_total:
        raise ValueError(f"{field_name} sum must equal count_exported")
    return counts


def _require_registered_sources(sources: list[str]) -> None:
    for source in sources:
        ensure_reward_source_registered(source)


class _LineageDigests(ContractModel):
    data_ref: NonEmptyStr
    data_ref_sha256: Sha256
    prompt_sha256: Sha256
    completion_sha256: Sha256
    parsed_program_sha256: Sha256
    validation_sha256: Sha256
    env_spec_sha256: Sha256
    stage_g_row_sha256: Sha256

    @field_validator("data_ref")
    @classmethod
    def data_ref_is_safe(cls, value: str) -> str:
        return _validate_relative_data_ref(value)


class ATIFTrialArtifact(_LineageDigests):
    atif_id: NonEmptyStr
    trial_dir: NonEmptyStr
    runtime: NonEmptyStr
    split: NonEmptyStr
    local_record_id: NonEmptyStr
    prompt_id: NonEmptyStr
    env_id: NonEmptyStr
    reward_source: NonEmptyStr
    training_log_reward: float = Field(ge=TRAINING_LOG_REWARD_MIN)
    trajectory_sha256: Sha256
    reward_sha256: Sha256

    @field_validator("atif_id")
    @classmethod
    def atif_id_is_safe(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="atif_id")

    @field_validator("reward_source")
    @classmethod
    def reward_source_is_registered(cls, value: str) -> str:
        ensure_reward_source_registered(value)
        return value

    @field_validator("training_log_reward")
    @classmethod
    def training_log_reward_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("training_log_reward must be finite")
        return value


class MixedATIFTrialArtifact(_LineageDigests):
    atif_id: NonEmptyStr
    trial_dir: NonEmptyStr
    source_run_id: NonEmptyStr
    source_run_dir: NonEmptyStr
    source_atif_dir: NonEmptyStr
    runtime: NonEmptyStr
    split: NonEmptyStr
    local_record_id: NonEmptyStr
    prompt_id: NonEmptyStr
    env_id: NonEmptyStr
    reward_source: NonEmptyStr
    training_log_reward: float = Field(ge=TRAINING_LOG_REWARD_MIN)
    source_trajectory_sha256: Sha256
    source_reward_sha256: Sha256
    trajectory_sha256: Sha256
    reward_sha256: Sha256

    @field_validator("atif_id", "source_run_id")
    @classmethod
    def ids_are_safe(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "id")
        return _validate_safe_id(value, field_name=field_name)

    @field_validator("reward_source")
    @classmethod
    def reward_source_is_registered(cls, value: str) -> str:
        ensure_reward_source_registered(value)
        return value

    @field_validator("training_log_reward")
    @classmethod
    def training_log_reward_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("training_log_reward must be finite")
        return value


class ATIFExportManifest(ContractModel):
    status: Literal["success"]
    schema_version: Literal["ppl_synth_data.atif_export.v1"] = (
        ATIF_EXPORT_MANIFEST_SCHEMA_VERSION
    )
    digest_algorithm: Literal["sha256"]
    run_dir: NonEmptyStr
    out_dir: NonEmptyStr
    metric_name: NonEmptyStr
    baseline: NonEmptyStr
    baseline_value: float
    reward_temperature: float
    runtime: NonEmptyStr
    split: NonEmptyStr
    reward_mode: Literal["semantic_metric", "embedded_reward_artifact"]
    reward_source: NonEmptyStr
    policy_logprobs_present: Literal[False]
    training_input_mode: Literal["message_only_requires_policy_rescore"]
    direct_offline_collect_training_supported: Literal[False]
    identity_mode: Literal["runtime_run_local_record_id"]
    count_input: int = Field(ge=0)
    count_exported: int = Field(ge=0)
    count_unique_atif_ids: int = Field(ge=0)
    split_counts: dict[NonEmptyStr, int]
    atif_ids: list[NonEmptyStr]
    count_dropped: int = Field(ge=0)
    trial_dirs: list[NonEmptyStr]
    trial_artifacts: dict[NonEmptyStr, ATIFTrialArtifact]
    dropped: list[dict[str, JsonValue]]

    @field_validator("reward_source")
    @classmethod
    def reward_source_is_registered(cls, value: str) -> str:
        ensure_reward_source_registered(value)
        return value

    @field_validator("baseline_value", "reward_temperature")
    @classmethod
    def floats_are_finite(cls, value: float, info: object) -> float:
        if not math.isfinite(value):
            field_name = getattr(info, "field_name", "value")
            raise ValueError(f"{field_name} must be finite")
        return value

    @model_validator(mode="after")
    def counts_and_artifacts_match(self) -> "ATIFExportManifest":
        if self.count_input != self.count_exported + self.count_dropped:
            raise ValueError("count_input must equal count_exported + count_dropped")
        if self.count_exported != len(self.trial_artifacts):
            raise ValueError("count_exported must equal trial_artifacts length")
        if self.count_exported != len(self.atif_ids):
            raise ValueError("count_exported must equal atif_ids length")
        if self.count_exported != len(self.trial_dirs):
            raise ValueError("count_exported must equal trial_dirs length")
        if self.count_unique_atif_ids != len(set(self.atif_ids)):
            raise ValueError("count_unique_atif_ids must equal unique atif_ids count")
        if set(self.trial_artifacts) != set(self.atif_ids):
            raise ValueError("trial_artifacts keys must match atif_ids")
        artifact_dirs = {
            artifact.trial_dir for artifact in self.trial_artifacts.values()
        }
        if set(self.trial_dirs) != artifact_dirs:
            raise ValueError("trial_dirs must match trial_artifacts trial_dir values")
        artifact_ids = [artifact.atif_id for artifact in self.trial_artifacts.values()]
        if sorted(artifact_ids) != sorted(self.atif_ids):
            raise ValueError("trial_artifacts atif_id values must match atif_ids")
        split_counts = Counter(
            artifact.split for artifact in self.trial_artifacts.values()
        )
        if dict(split_counts) != dict(self.split_counts):
            raise ValueError("split_counts must match trial_artifacts splits")
        _validate_counts(
            self.split_counts,
            expected_total=self.count_exported,
            field_name="split_counts",
        )
        if any(
            artifact.reward_source != self.reward_source
            for artifact in self.trial_artifacts.values()
        ):
            raise ValueError(
                "trial_artifacts reward_source values must match manifest reward_source"
            )
        return self


class MixedATIFAssemblyManifest(ContractModel):
    status: Literal["success"]
    schema_version: Literal["ppl_synth_data.mixed_atif_assembly.v1"] = (
        MIXED_ATIF_MANIFEST_SCHEMA_VERSION
    )
    digest_algorithm: Literal["sha256"]
    timestamp_utc: NonEmptyStr
    out_dir: NonEmptyStr
    source_run_dirs: dict[NonEmptyStr, NonEmptyStr]
    source_atif_dirs: dict[NonEmptyStr, NonEmptyStr]
    source_manifest_sha256: dict[NonEmptyStr, Sha256]
    trial_source_run_ids: dict[NonEmptyStr, NonEmptyStr]
    source_counts: dict[NonEmptyStr, int]
    runtime_counts: dict[NonEmptyStr, int]
    reward_source_counts: dict[NonEmptyStr, int]
    split_counts: dict[NonEmptyStr, int]
    count_exported: int = Field(ge=0)
    row_count: int = Field(ge=0)
    count_dropped: int = Field(default=0, ge=0)
    count_unique_atif_ids: int = Field(ge=0)
    atif_ids: list[NonEmptyStr]
    trial_dirs: list[NonEmptyStr]
    trial_artifacts: dict[NonEmptyStr, MixedATIFTrialArtifact]

    @model_validator(mode="after")
    def source_maps_and_counts_match(self) -> "MixedATIFAssemblyManifest":
        if self.count_exported != self.row_count:
            raise ValueError("row_count must equal count_exported")
        if self.count_exported != len(self.trial_artifacts):
            raise ValueError("count_exported must equal trial_artifacts length")
        if self.count_exported != len(self.trial_source_run_ids):
            raise ValueError("count_exported must equal trial_source_run_ids length")
        if self.count_exported != len(self.atif_ids):
            raise ValueError("count_exported must equal atif_ids length")
        if self.count_exported != len(self.trial_dirs):
            raise ValueError("count_exported must equal trial_dirs length")
        source_ids = set(self.source_run_dirs)
        if not source_ids:
            raise ValueError("source_run_dirs must not be empty")
        if set(self.source_atif_dirs) != source_ids:
            raise ValueError("source_atif_dirs keys must match source_run_dirs")
        if set(self.source_manifest_sha256) != source_ids:
            raise ValueError("source_manifest_sha256 keys must match source_run_dirs")
        if set(self.trial_source_run_ids) != set(self.trial_artifacts):
            raise ValueError("trial_source_run_ids keys must match trial_artifacts")
        if self.count_unique_atif_ids != len(set(self.atif_ids)):
            raise ValueError("count_unique_atif_ids must equal unique atif_ids count")
        artifact_ids = [artifact.atif_id for artifact in self.trial_artifacts.values()]
        if sorted(artifact_ids) != sorted(self.atif_ids):
            raise ValueError("trial_artifacts atif_id values must match atif_ids")
        artifact_dirs = {
            artifact.trial_dir for artifact in self.trial_artifacts.values()
        }
        if set(self.trial_dirs) != artifact_dirs:
            raise ValueError("trial_dirs must match trial_artifacts trial_dir values")
        for trial_name, source_run_id in self.trial_source_run_ids.items():
            _validate_safe_id(trial_name, field_name="trial_source_run_ids key")
            _validate_safe_id(source_run_id, field_name="source_run_id")
            if source_run_id not in source_ids:
                raise ValueError(
                    "trial_source_run_ids references unknown source_run_id"
                )
            artifact = self.trial_artifacts[trial_name]
            if Path(artifact.trial_dir).name != trial_name:
                raise ValueError(
                    "artifact trial_dir basename must match trial_artifacts key"
                )
            if artifact.source_run_id != source_run_id:
                raise ValueError(
                    "trial_source_run_ids must match artifact source_run_id"
                )
            if artifact.source_run_dir != self.source_run_dirs[source_run_id]:
                raise ValueError("artifact source_run_dir must match source_run_dirs")
            if artifact.source_atif_dir != self.source_atif_dirs[source_run_id]:
                raise ValueError("artifact source_atif_dir must match source_atif_dirs")
        _validate_counts(
            self.source_counts,
            expected_total=self.count_exported,
            field_name="source_counts",
        )
        _validate_counts(
            self.runtime_counts,
            expected_total=self.count_exported,
            field_name="runtime_counts",
        )
        _validate_counts(
            self.reward_source_counts,
            expected_total=self.count_exported,
            field_name="reward_source_counts",
        )
        _validate_counts(
            self.split_counts,
            expected_total=self.count_exported,
            field_name="split_counts",
        )
        if dict(Counter(self.trial_source_run_ids.values())) != dict(
            self.source_counts
        ):
            raise ValueError("source_counts must match trial_source_run_ids")
        if dict(
            Counter(artifact.runtime for artifact in self.trial_artifacts.values())
        ) != dict(self.runtime_counts):
            raise ValueError("runtime_counts must match trial_artifacts")
        if dict(
            Counter(
                artifact.reward_source for artifact in self.trial_artifacts.values()
            )
        ) != dict(self.reward_source_counts):
            raise ValueError("reward_source_counts must match trial_artifacts")
        if dict(
            Counter(artifact.split for artifact in self.trial_artifacts.values())
        ) != dict(self.split_counts):
            raise ValueError("split_counts must match trial_artifacts")
        _require_registered_sources(list(self.reward_source_counts))
        return self


class RichRewardATIFExportManifest(ContractModel):
    status: Literal["success"]
    schema_version: Literal["ppl_synth_data.rich_reward_atif_export.v1"] = (
        RICH_REWARD_ATIF_MANIFEST_SCHEMA_VERSION
    )
    source_atif_dir: NonEmptyStr
    out_dir: NonEmptyStr
    rubric_scores_path: NonEmptyStr
    reward_source: NonEmptyStr
    base_reward_sources: list[NonEmptyStr]
    component_reward_sources: list[NonEmptyStr]
    rubric_id: NonEmptyStr
    rubric_version: NonEmptyStr
    heuristic_weight: float
    process_weight: float
    count_input: int = Field(ge=0)
    count_exported: int = Field(ge=0)
    count_dropped: int = Field(ge=0)
    atif_ids: list[NonEmptyStr]
    dropped: list[dict[str, JsonValue]]
    training_input_mode: Literal["message_only_requires_policy_rescore"]

    @field_validator("reward_source")
    @classmethod
    def reward_source_is_registered(cls, value: str) -> str:
        ensure_reward_source_registered(value)
        return value

    @field_validator("heuristic_weight", "process_weight")
    @classmethod
    def weights_are_finite(cls, value: float, info: object) -> float:
        if not math.isfinite(value):
            field_name = getattr(info, "field_name", "weight")
            raise ValueError(f"{field_name} must be finite")
        return value

    @model_validator(mode="after")
    def counts_and_sources_match(self) -> "RichRewardATIFExportManifest":
        if self.count_input != self.count_exported + self.count_dropped:
            raise ValueError("count_input must equal count_exported + count_dropped")
        if self.count_exported != len(self.atif_ids):
            raise ValueError("count_exported must equal atif_ids length")
        _require_registered_sources(
            [
                self.reward_source,
                *self.base_reward_sources,
                *self.component_reward_sources,
            ]
        )
        return self


class LiveRichRewardManifest(ContractModel):
    status: Literal["success"]
    schema_version: Literal["ppl_synth_data.live_rich_reward.v1"] = (
        LIVE_RICH_REWARD_MANIFEST_SCHEMA_VERSION
    )
    trial_dir: NonEmptyStr
    atif_id: NonEmptyStr
    base_trajectory_path: NonEmptyStr
    reward_source: NonEmptyStr
    component_reward_sources: list[NonEmptyStr]
    component_names: list[NonEmptyStr]
    scalarization: dict[str, JsonValue] | None
    rubric: dict[str, JsonValue]
    written_reward_paths: list[NonEmptyStr]
    written_trajectory_paths: list[NonEmptyStr]

    @field_validator("atif_id")
    @classmethod
    def atif_id_is_safe(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="atif_id")

    @field_validator("reward_source")
    @classmethod
    def reward_source_is_registered(cls, value: str) -> str:
        ensure_reward_source_registered(value)
        return value

    @model_validator(mode="after")
    def sources_and_outputs_are_present(self) -> "LiveRichRewardManifest":
        if not self.component_names:
            raise ValueError("component_names must not be empty")
        if not self.written_reward_paths:
            raise ValueError("written_reward_paths must not be empty")
        if not self.written_trajectory_paths:
            raise ValueError("written_trajectory_paths must not be empty")
        _require_registered_sources(
            [self.reward_source, *self.component_reward_sources]
        )
        return self


ReleaseManifest: TypeAlias = Annotated[
    ATIFExportManifest
    | MixedATIFAssemblyManifest
    | RichRewardATIFExportManifest
    | LiveRichRewardManifest,
    Field(discriminator="schema_version"),
]

ReleaseManifestAdapter: TypeAdapter[ReleaseManifest] = TypeAdapter(ReleaseManifest)

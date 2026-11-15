"""Release manifest and lineage contracts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    Field,
    NonNegativeInt,
    TypeAdapter,
    model_validator,
)

from synthstats_schemas.base import (
    TRAINING_LOG_REWARD_MIN,
    ContractModel,
    JsonValue,
    NonEmptyStr,
    Sha256,
)
from synthstats_schemas.receipt import _validate_canonical_relative_posix_path
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

def _relative_data_ref(value: str) -> str:
    return _validate_canonical_relative_posix_path(
        value, field_name="data_ref"
    )


def _registered_reward_source(name: str) -> str:
    ensure_reward_source_registered(name)
    return name


def _safe_id(value: str) -> str:
    if (
        value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("id must be a single safe path segment")
    return value


SafeId = Annotated[NonEmptyStr, AfterValidator(_safe_id)]
RelativeDataRef = Annotated[NonEmptyStr, AfterValidator(_relative_data_ref)]
RegisteredRewardSource = Annotated[
    NonEmptyStr, AfterValidator(_registered_reward_source)
]
CountMap = Annotated[dict[NonEmptyStr, NonNegativeInt], Field(min_length=1)]


class _LineageDigests(ContractModel):
    data_ref: RelativeDataRef
    data_ref_sha256: Sha256
    prompt_sha256: Sha256
    completion_sha256: Sha256
    parsed_program_sha256: Sha256
    validation_sha256: Sha256
    env_spec_sha256: Sha256
    stage_g_row_sha256: Sha256


class ATIFTrialArtifact(_LineageDigests):
    atif_id: SafeId
    trial_dir: NonEmptyStr
    runtime: NonEmptyStr
    split: NonEmptyStr
    local_record_id: NonEmptyStr
    prompt_id: NonEmptyStr
    env_id: NonEmptyStr
    reward_source: RegisteredRewardSource
    training_log_reward: float = Field(
        ge=TRAINING_LOG_REWARD_MIN, allow_inf_nan=False
    )
    trajectory_sha256: Sha256
    reward_sha256: Sha256


class MixedATIFTrialArtifact(_LineageDigests):
    atif_id: SafeId
    trial_dir: NonEmptyStr
    source_run_id: SafeId
    source_run_dir: NonEmptyStr
    source_atif_dir: NonEmptyStr
    runtime: NonEmptyStr
    split: NonEmptyStr
    local_record_id: NonEmptyStr
    prompt_id: NonEmptyStr
    env_id: NonEmptyStr
    reward_source: RegisteredRewardSource
    training_log_reward: float = Field(
        ge=TRAINING_LOG_REWARD_MIN, allow_inf_nan=False
    )
    source_trajectory_sha256: Sha256
    source_reward_sha256: Sha256
    trajectory_sha256: Sha256
    reward_sha256: Sha256


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
    baseline_value: float = Field(allow_inf_nan=False)
    reward_temperature: float = Field(allow_inf_nan=False)
    runtime: NonEmptyStr
    split: NonEmptyStr
    reward_mode: Literal["semantic_metric", "embedded_reward_artifact"]
    reward_source: RegisteredRewardSource
    policy_logprobs_present: Literal[False]
    training_input_mode: Literal["message_only_requires_policy_rescore"]
    direct_offline_collect_training_supported: Literal[False]
    identity_mode: Literal["runtime_run_local_record_id"]
    count_input: int = Field(ge=0)
    count_exported: int = Field(ge=0)
    count_unique_atif_ids: int = Field(ge=0)
    split_counts: CountMap
    atif_ids: list[NonEmptyStr]
    count_dropped: int = Field(ge=0)
    trial_dirs: list[NonEmptyStr]
    trial_artifacts: dict[NonEmptyStr, ATIFTrialArtifact]
    dropped: list[dict[str, JsonValue]]

    @model_validator(mode="after")
    def counts_and_artifacts_match(self) -> "ATIFExportManifest":
        if self.count_input != self.count_exported + self.count_dropped:
            raise ValueError("count_input must equal count_exported + count_dropped")
        if self.count_dropped != len(self.dropped):
            raise ValueError("count_dropped must equal dropped length")
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
        if any(key != artifact.atif_id for key, artifact in self.trial_artifacts.items()):
            raise ValueError("trial_artifacts keys must match their embedded atif_id")
        if Counter(self.trial_dirs) != Counter(
            artifact.trial_dir for artifact in self.trial_artifacts.values()
        ):
            raise ValueError("trial_dirs must match trial_artifacts trial_dir values")
        artifact_ids = [artifact.atif_id for artifact in self.trial_artifacts.values()]
        if sorted(artifact_ids) != sorted(self.atif_ids):
            raise ValueError("trial_artifacts atif_id values must match atif_ids")
        split_counts = Counter(
            artifact.split for artifact in self.trial_artifacts.values()
        )
        if dict(split_counts) != dict(self.split_counts):
            raise ValueError("split_counts must match trial_artifacts splits")
        if sum(self.split_counts.values()) != self.count_exported:
            raise ValueError("split_counts sum must equal count_exported")
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
    source_run_dirs: dict[SafeId, NonEmptyStr] = Field(min_length=1)
    source_atif_dirs: dict[SafeId, NonEmptyStr]
    source_manifest_sha256: dict[SafeId, Sha256]
    trial_source_run_ids: dict[SafeId, SafeId]
    source_counts: CountMap
    runtime_counts: CountMap
    reward_source_counts: dict[RegisteredRewardSource, NonNegativeInt] = Field(
        min_length=1
    )
    split_counts: CountMap
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
        if set(self.source_atif_dirs) != source_ids:
            raise ValueError("source_atif_dirs keys must match source_run_dirs")
        if set(self.source_manifest_sha256) != source_ids:
            raise ValueError("source_manifest_sha256 keys must match source_run_dirs")
        if set(self.trial_source_run_ids) != set(self.trial_artifacts):
            raise ValueError("trial_source_run_ids keys must match trial_artifacts")
        if any(key != artifact.atif_id for key, artifact in self.trial_artifacts.items()):
            raise ValueError("trial_artifacts keys must match their embedded atif_id")
        if self.count_unique_atif_ids != len(set(self.atif_ids)):
            raise ValueError("count_unique_atif_ids must equal unique atif_ids count")
        artifact_ids = [artifact.atif_id for artifact in self.trial_artifacts.values()]
        if sorted(artifact_ids) != sorted(self.atif_ids):
            raise ValueError("trial_artifacts atif_id values must match atif_ids")
        if Counter(self.trial_dirs) != Counter(
            artifact.trial_dir for artifact in self.trial_artifacts.values()
        ):
            raise ValueError("trial_dirs must match trial_artifacts trial_dir values")
        for trial_name, source_run_id in self.trial_source_run_ids.items():
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
        count_maps = (
            ("source_counts", self.source_counts),
            ("runtime_counts", self.runtime_counts),
            ("reward_source_counts", self.reward_source_counts),
            ("split_counts", self.split_counts),
        )
        for field_name, counts in count_maps:
            if sum(counts.values()) != self.count_exported:
                raise ValueError(f"{field_name} sum must equal count_exported")
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
        return self


class RichRewardATIFExportManifest(ContractModel):
    status: Literal["success"]
    schema_version: Literal["ppl_synth_data.rich_reward_atif_export.v1"] = (
        RICH_REWARD_ATIF_MANIFEST_SCHEMA_VERSION
    )
    source_atif_dir: NonEmptyStr
    out_dir: NonEmptyStr
    rubric_scores_path: NonEmptyStr
    reward_source: RegisteredRewardSource
    base_reward_sources: list[RegisteredRewardSource]
    component_reward_sources: list[RegisteredRewardSource]
    rubric_id: NonEmptyStr
    rubric_version: NonEmptyStr
    heuristic_weight: float = Field(allow_inf_nan=False)
    process_weight: float = Field(allow_inf_nan=False)
    count_input: int = Field(ge=0)
    count_exported: int = Field(ge=0)
    count_dropped: int = Field(ge=0)
    atif_ids: list[NonEmptyStr]
    dropped: list[dict[str, JsonValue]]
    training_input_mode: Literal["message_only_requires_policy_rescore"]

    @model_validator(mode="after")
    def counts_and_sources_match(self) -> "RichRewardATIFExportManifest":
        if self.count_input != self.count_exported + self.count_dropped:
            raise ValueError("count_input must equal count_exported + count_dropped")
        if self.count_dropped != len(self.dropped):
            raise ValueError("count_dropped must equal dropped length")
        if self.count_exported != len(self.atif_ids):
            raise ValueError("count_exported must equal atif_ids length")
        return self


class LiveRichRewardManifest(ContractModel):
    status: Literal["success"]
    schema_version: Literal["ppl_synth_data.live_rich_reward.v1"] = (
        LIVE_RICH_REWARD_MANIFEST_SCHEMA_VERSION
    )
    trial_dir: NonEmptyStr
    atif_id: SafeId
    base_trajectory_path: NonEmptyStr
    reward_source: RegisteredRewardSource
    component_reward_sources: list[RegisteredRewardSource]
    component_names: list[NonEmptyStr] = Field(min_length=1)
    scalarization: dict[str, JsonValue] | None
    rubric: dict[str, JsonValue]
    written_reward_paths: list[NonEmptyStr] = Field(min_length=1)
    written_trajectory_paths: list[NonEmptyStr] = Field(min_length=1)


class SFTReleaseRowArtifact(ContractModel):
    row_id: SafeId
    runtime: SafeId
    split: Literal["train", "validation", "test"]
    split_group_id: SafeId
    row_sha256: Sha256


class SFTReleaseManifest(ContractModel):
    schema_version: Literal["synthstats.sft_release.v1"]
    count_written: int = Field(ge=0)
    count_unique_split_groups: int = Field(ge=0)
    row_split_counts: dict[Literal["train", "validation", "test"], NonNegativeInt]
    split_group_counts: dict[Literal["train", "validation", "test"], NonNegativeInt]
    split_group_assignments: dict[SafeId, Literal["train", "validation", "test"]]
    row_artifacts: dict[SafeId, SFTReleaseRowArtifact]

    @model_validator(mode="after")
    def counts_and_assignments_match(self) -> "SFTReleaseManifest":
        for row_key, artifact in self.row_artifacts.items():
            if row_key != artifact.row_id:
                raise ValueError("row_artifacts keys must match artifact row_id values")

        group_splits: dict[str, set[str]] = {}
        for artifact in self.row_artifacts.values():
            group_splits.setdefault(artifact.split_group_id, set()).add(artifact.split)
        if any(len(splits) != 1 for splits in group_splits.values()):
            raise ValueError("each split_group_id must belong to exactly one split")

        expected_assignments = {
            split_group_id: next(iter(splits))
            for split_group_id, splits in group_splits.items()
        }
        if self.split_group_assignments != expected_assignments:
            raise ValueError(
                "split_group_assignments must match row_artifacts split groups"
            )

        if self.count_written != len(self.row_artifacts):
            raise ValueError("count_written must equal row_artifacts length")
        if self.count_unique_split_groups != len(expected_assignments):
            raise ValueError(
                "count_unique_split_groups must equal unique split_group_id count"
            )
        if self.row_split_counts != dict(
            Counter(artifact.split for artifact in self.row_artifacts.values())
        ):
            raise ValueError("row_split_counts must match row_artifacts splits")
        if self.split_group_counts != dict(Counter(expected_assignments.values())):
            raise ValueError(
                "split_group_counts must match split_group_assignments values"
            )
        return self


ReleaseManifest: TypeAlias = Annotated[
    ATIFExportManifest
    | MixedATIFAssemblyManifest
    | RichRewardATIFExportManifest
    | LiveRichRewardManifest
    | SFTReleaseManifest,
    Field(discriminator="schema_version"),
]

ReleaseManifestAdapter: TypeAdapter[ReleaseManifest] = TypeAdapter(ReleaseManifest)

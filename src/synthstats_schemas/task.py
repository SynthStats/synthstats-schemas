"""Task and evidence contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from synthstats_schemas.base import (
    CARD_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION,
    ContractModel,
    JsonValue,
)
from synthstats_schemas.reward import RewardArtifact, ensure_reward_source_registered


class EvidenceCard(ContractModel):
    schema_version: Literal["ppl.platform.card.v0"] = CARD_SCHEMA_VERSION
    evidence_id: Annotated[str, Field(min_length=1)]
    task_id: Annotated[str, Field(min_length=1)]
    reward_artifact: RewardArtifact
    summary: str = ""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TaskManifest(ContractModel):
    schema_version: Literal["ppl.platform.task.v0"] = TASK_SCHEMA_VERSION
    task_id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    version: Annotated[str, Field(min_length=1)]
    reward_sources: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reward_sources_are_registered_and_unique(self) -> "TaskManifest":
        if len(self.reward_sources) != len(set(self.reward_sources)):
            raise ValueError("reward_sources must be unique")
        for source in self.reward_sources:
            ensure_reward_source_registered(source)
        return self

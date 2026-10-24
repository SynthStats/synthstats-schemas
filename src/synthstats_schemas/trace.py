"""Trace and record contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from synthstats_schemas.base import (
    ATIF_SCHEMA_VERSION,
    ContractModel,
    JsonValue,
    RECORD_SCHEMA_VERSION,
)
from synthstats_schemas.reward import RewardArtifact


class RuntimeDescriptor(ContractModel):
    name: Annotated[str, Field(min_length=1)]
    version: str | None = None
    execution_backend: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PromptView(ContractModel):
    schema_version: Literal["ppl.platform.record.v0"] = RECORD_SCHEMA_VERSION
    view_id: Annotated[str, Field(min_length=1)]
    task_id: str | None = None
    split: str | None = None
    content: str
    format: str = "text/plain"
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class GenerationRecord(ContractModel):
    schema_version: Literal["ppl.platform.record.v0"] = RECORD_SCHEMA_VERSION
    record_id: Annotated[str, Field(min_length=1)]
    task_id: str | None = None
    prompt_id: str | None = None
    split: str | None = None
    runtime: RuntimeDescriptor | None = None
    prompt: PromptView | None = None
    raw_output: str
    candidate_outputs: list[str] = Field(default_factory=list)
    provider_response: dict[str, JsonValue] = Field(default_factory=dict)
    parsed_program: str | None = None
    validation: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TraceStep(ContractModel):
    step_id: str | None = None
    source: str | None = None
    role: str | None = None
    message: JsonValue | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, JsonValue]] = Field(default_factory=list)
    observation: dict[str, JsonValue] | None = None
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TraceRecord(ContractModel):
    schema_version: Literal["ppl.platform.record.v0"] = RECORD_SCHEMA_VERSION
    trace_id: Annotated[str, Field(min_length=1)]
    task_id: str | None = None
    trial_id: str | None = None
    steps: list[TraceStep] = Field(default_factory=list)
    artifacts: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TrialRecord(ContractModel):
    schema_version: Literal["ppl.platform.record.v0"] = RECORD_SCHEMA_VERSION
    trial_id: Annotated[str, Field(min_length=1)]
    task_id: str | None = None
    language: str | None = None
    program: str | None = None
    reward: float | None = None
    runtime: RuntimeDescriptor | None = None
    trace_id: str | None = None
    artifacts: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ATIFRecord(ContractModel):
    """ATIF trajectory shell for offline interchange."""

    schema_version: Literal["ATIF-v1.6"] = ATIF_SCHEMA_VERSION
    session_id: Annotated[str, Field(min_length=1)]
    steps: list[TraceStep]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    reward_artifact: RewardArtifact | None = None

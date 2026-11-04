"""Trace and record contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from harbor.models.trajectories import (
    Agent,
    AudioSource,
    ContentPart,
    FinalMetrics,
    ImageSource,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from pydantic import Field

from synthstats_schemas.base import (
    ContractModel,
    JsonValue,
    RECORD_SCHEMA_VERSION,
)


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


__all__ = [
    "Agent", "AudioSource", "ContentPart", "FinalMetrics", "ImageSource",
    "Metrics", "Observation", "ObservationResult", "Step", "SubagentTrajectoryRef",
    "ToolCall", "Trajectory", "GenerationRecord", "PromptView", "RuntimeDescriptor",
    "TrialRecord",
]

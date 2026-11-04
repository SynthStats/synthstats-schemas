"""Internal probabilistic-program trace records. This format is not ATIF."""

from typing import Annotated, Literal

from pydantic import Field

from synthstats_schemas.base import ContractModel, JsonValue, RECORD_SCHEMA_VERSION


class ProgramTraceStep(ContractModel):
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


class ProgramTraceRecord(ContractModel):
    schema_version: Literal["ppl.platform.record.v0"] = RECORD_SCHEMA_VERSION
    trace_id: Annotated[str, Field(min_length=1)]
    task_id: str | None = None
    trial_id: str | None = None
    steps: list[ProgramTraceStep] = Field(default_factory=list)
    artifacts: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

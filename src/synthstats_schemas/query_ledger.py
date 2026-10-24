"""Environment-owned query accounting facts, independent of trajectory writers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from synthstats_schemas.base import ContractModel

NonnegativeInt = Annotated[int, Field(ge=0)]


class LedgerAllowance(ContractModel):
    """Separate allowances, also used for the amount charged by one request."""

    acquisition_credits: NonnegativeInt
    assistant_turns: NonnegativeInt
    execution_calls: NonnegativeInt
    execution_time_ms: NonnegativeInt
    generation_tokens: NonnegativeInt


class QueryLedgerEvent(ContractModel):
    """One resolved or refused request and its authoritative accounting facts."""

    request_id: Annotated[str, Field(min_length=1)]
    tool: Annotated[str, Field(min_length=1)]
    arguments_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    charge: LedgerAllowance
    remaining: LedgerAllowance
    outcome: Literal["resolved", "refused"]
    observation: JsonValue = None
    refusal_reason: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def require_outcome_payload(self) -> QueryLedgerEvent:
        if self.outcome == "refused":
            if self.observation is not None or self.refusal_reason is None:
                raise ValueError("A refusal carries only its refusal reason, never an observation")
        elif self.refusal_reason is not None:
            raise ValueError("A resolved request cannot carry a refusal reason")
        return self

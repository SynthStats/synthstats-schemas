"""RL data joined to the exact bytes of a stock Harbor ATIF trajectory."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import Field, JsonValue, ValidationInfo, field_validator, model_validator

from synthstats_schemas.base import _REVALIDATION_SENTINEL, ContractModel, require_json_serializable
from synthstats_schemas.query_ledger import QueryLedgerEvent

TokenID = Annotated[int, Field(ge=0)]
LossBit = Annotated[int, Field(ge=0, le=1)]
LogProbability = Annotated[float, Field(le=0, allow_inf_nan=False)]
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1)]


class ATIFServingIdentity(ContractModel):
    """The observed model generation, without a serving-attestation protocol."""

    checkpoint_step: Annotated[int, Field(ge=0)]
    weights_sha256: SHA256
    vllm_model_id: Identifier
    model_id: str | None = None
    sampler_settings: dict[str, JsonValue] | None = None

    @field_validator("sampler_settings")
    @classmethod
    def finite_sampler(cls, value):
        return require_json_serializable(value, field_name="sampler_settings")


class ATIFRLCall(ContractModel):
    """One policy call and the environment suffix following it."""

    step_id: Annotated[int, Field(ge=1)]
    call_id: Identifier
    prompt_ids: list[TokenID] = Field(min_length=1)
    response_ids: list[TokenID] = Field(min_length=1)
    environment_ids: list[TokenID]
    segment: list[TokenID] = Field(min_length=2, max_length=2)
    logprobs: list[LogProbability] = Field(min_length=1)
    message: str
    tool_call: dict[str, JsonValue]
    observation: dict[str, JsonValue] | None
    turn_reward: dict[str, JsonValue]
    sampler: dict[str, JsonValue]
    serving: ATIFServingIdentity
    ledger_event_ids: list[Identifier]

    @field_validator("tool_call", "observation", "turn_reward", "sampler")
    @classmethod
    def finite_payload(cls, value, info: ValidationInfo):
        return require_json_serializable(value, field_name=info.field_name)

    @model_validator(mode="after")
    def aligned_call(self):
        if len(self.response_ids) != len(self.logprobs):
            raise ValueError("Call logprobs must cover every generated token")
        if self.segment[1] - self.segment[0] != len(self.response_ids) + len(self.environment_ids):
            raise ValueError("Call segment must cover its generated and environment tokens")
        if self.tool_call.get("tool_call_id") != self.call_id:
            raise ValueError("Call ID must match its tool call")
        return self


class ATIFRLSidecar(ContractModel):
    """A trajectory's RL arrays, calls, terminal reward and query accounting.

    Validation requires ``context={"trajectory_bytes": raw_atif_bytes}``.
    ``prompt_ids`` is the initial prefix; response positions have their own
    equally sized token, mask and behavior-probability arrays.
    """

    schema_version: Literal["synthstats.atif-rl-sidecar.v1"] = "synthstats.atif-rl-sidecar.v1"
    session_id: Identifier
    trajectory_sha256: SHA256
    prompt_ids: list[TokenID] = Field(min_length=1)
    response_ids: list[TokenID] = Field(min_length=1)
    loss_mask: list[LossBit] = Field(min_length=1)
    logprobs: list[LogProbability | None] = Field(min_length=1)
    calls: list[ATIFRLCall] = Field(min_length=1)
    reward: dict[str, JsonValue] = Field(min_length=1)
    query_ledger_events: list[QueryLedgerEvent] = Field(min_length=1)

    @field_validator("reward")
    @classmethod
    def finite_reward(cls, value):
        return require_json_serializable(value, field_name="reward")

    @model_validator(mode="after")
    def joined_records(self, info: ValidationInfo):
        n = len(self.response_ids)
        if len(self.loss_mask) != n or len(self.logprobs) != n:
            raise ValueError("Response IDs, loss mask and logprobs must have equal lengths")
        if any((probability is None) != (mask == 0)
               for probability, mask in zip(self.logprobs, self.loss_mask, strict=True)):
            raise ValueError("A logprob must be null exactly at an environment position")
        if len({call.call_id for call in self.calls}) != len(self.calls):
            raise ValueError("Duplicate call ID")
        if len({call.step_id for call in self.calls}) != len(self.calls):
            raise ValueError("Each call must name a different trajectory step")
        cursor, prompt, event_ids = 0, self.prompt_ids, []
        for call in self.calls:
            start, end = call.segment
            if start != cursor or end > n:
                raise ValueError("Call segments must partition the response without overlaps or gaps")
            boundary = start + len(call.response_ids)
            if (self.response_ids[start:boundary] != call.response_ids
                    or self.response_ids[boundary:end] != call.environment_ids
                    or self.loss_mask[start:boundary] != [1] * len(call.response_ids)
                    or self.loss_mask[boundary:end] != [0] * len(call.environment_ids)
                    or self.logprobs[start:boundary] != call.logprobs):
                raise ValueError("Call tokens, masks or logprobs differ from the response segment")
            if call.prompt_ids != prompt:
                raise ValueError("Call prompt differs from the exact preceding prefix and suffixes")
            prompt = call.prompt_ids + call.response_ids + call.environment_ids
            cursor = end
            if any(event_id not in (call.call_id, call.call_id + ":result")
                   for event_id in call.ledger_event_ids):
                raise ValueError("Ledger event does not belong to its call")
            event_ids.extend(call.ledger_event_ids)
        if cursor != n:
            raise ValueError("Call segments must cover the complete response")
        ledger_ids = [event.request_id for event in self.query_ledger_events]
        if len(set(ledger_ids)) != len(ledger_ids) or event_ids != ledger_ids:
            raise ValueError("Calls must join every ledger event exactly once, in order")

        raw = (info.context or {}).get("trajectory_bytes")
        if not isinstance(raw, bytes):
            if (info.context or {}).get("_revalidated_snapshot") is _REVALIDATION_SENTINEL:
                # The base's dump-time revalidation of an already-validated
                # instance: the exact bytes bound this sidecar at construction
                # and are not part of the model; every invariant above has
                # already revalidated. The check is by identity against the
                # base's private sentinel object, so the public context
                # parameter cannot forge this path. External construction
                # still requires the bytes.
                return self
            raise ValueError("Exact trajectory_bytes are required to validate a sidecar")
        if sha256(raw).hexdigest() != self.trajectory_sha256:
            raise ValueError("trajectory_sha256 differs from the exact trajectory bytes")
        try:
            trajectory = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("trajectory_bytes must contain ATIF JSON") from error
        if (not isinstance(trajectory, dict)
                or trajectory.get("schema_version") != "ATIF-v1.6"
                or trajectory.get("session_id") != self.session_id):
            raise ValueError("Sidecar must join the same ATIF-v1.6 session")
        steps = trajectory.get("steps")
        if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
            raise ValueError("Trajectory must contain its steps")
        step_ids = [step.get("step_id") for step in steps]
        if any(type(step_id) is not int for step_id in step_ids) or len(set(step_ids)) != len(steps):
            raise ValueError("Trajectory steps must have unique integer IDs")
        agents = {step["step_id"]: step for step in steps if step.get("source") == "agent"}
        if set(agents) != {call.step_id for call in self.calls}:
            raise ValueError("Every agent step must join exactly one sidecar call")
        for call in self.calls:
            step = agents[call.step_id]
            if (step.get("message") != call.message
                    or step.get("tool_calls") != [call.tool_call]
                    or step.get("observation") != call.observation):
                raise ValueError("Sidecar action or observation differs from its trajectory step")
            metrics = step.get("metrics") or {}
            for key, expected in (("prompt_token_ids", call.prompt_ids),
                                  ("completion_token_ids", call.response_ids),
                                  ("logprobs", call.logprobs)):
                if metrics.get(key) is not None and metrics[key] != expected:
                    raise ValueError("Sidecar tokens or probabilities differ from trajectory metrics")
        return self

"""Contracts for token-aligned action sequences used in trajectory balance (TB)
and subtrajectory balance (SubTB) training.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, TypeAdapter, model_validator

from synthstats_schemas.base import ContractModel
from synthstats_schemas.reward import RewardArtifact


class _BaseTrainingTrajectory(ContractModel):
    trajectory_id: Annotated[str, Field(min_length=1)]
    token_ids: list[int]
    logprobs: list[float]
    loss_mask: list[bool]
    reward_artifact: RewardArtifact

    @model_validator(mode="after")
    def token_arrays_are_valid(self) -> "_BaseTrainingTrajectory":
        n_tokens = len(self.token_ids)
        if n_tokens == 0:
            raise ValueError("token_ids must not be empty")
        if len(self.logprobs) != n_tokens:
            raise ValueError("logprobs length must match token_ids length")
        if len(self.loss_mask) != n_tokens:
            raise ValueError("loss_mask length must match token_ids length")
        if not any(self.loss_mask):
            raise ValueError("loss_mask must contain at least one trained token")
        for token_id in self.token_ids:
            if isinstance(token_id, bool) or token_id < 0:
                raise ValueError("token_ids must be non-negative integers")
        for logprob in self.logprobs:
            if not math.isfinite(float(logprob)):
                raise ValueError("logprobs must be finite")
        return self


class TBTrainingTrajectory(_BaseTrainingTrajectory):
    objective: Literal["tb"] = "tb"
    eos_logprobs: list[float] | None = None

    @model_validator(mode="after")
    def eos_is_optional_but_aligned(self) -> "TBTrainingTrajectory":
        if self.eos_logprobs is None:
            return self
        if len(self.eos_logprobs) != len(self.token_ids):
            raise ValueError("eos_logprobs length must match token_ids length")
        for value in self.eos_logprobs:
            if not math.isfinite(float(value)):
                raise ValueError("eos_logprobs must be finite")
        return self


class SubTBTrainingTrajectory(_BaseTrainingTrajectory):
    objective: Literal["subtb"] = "subtb"
    eos_logprobs: list[float]

    @model_validator(mode="after")
    def eos_is_required_and_aligned(self) -> "SubTBTrainingTrajectory":
        if len(self.eos_logprobs) != len(self.token_ids):
            raise ValueError("eos_logprobs length must match token_ids length")
        for value in self.eos_logprobs:
            if not math.isfinite(float(value)):
                raise ValueError("eos_logprobs must be finite")
        return self


TrainingTrajectory: TypeAlias = Annotated[
    TBTrainingTrajectory | SubTBTrainingTrajectory,
    Field(discriminator="objective"),
]

TrainingTrajectoryAdapter: TypeAdapter[TrainingTrajectory] = TypeAdapter(
    TrainingTrajectory
)

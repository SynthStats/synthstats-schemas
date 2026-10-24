"""Base types for SynthStats wire contracts."""

from __future__ import annotations

import json
from typing import Any, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

REWARD_ARTIFACT_SCHEMA_VERSION: Final[Literal["synthstats.harbor.reward.v1"]] = (
    "synthstats.harbor.reward.v1"
)
RECORD_SCHEMA_VERSION: Final[Literal["ppl.platform.record.v0"]] = (
    "ppl.platform.record.v0"
)
TASK_SCHEMA_VERSION: Final[Literal["ppl.platform.task.v0"]] = "ppl.platform.task.v0"
CARD_SCHEMA_VERSION: Final[Literal["ppl.platform.card.v0"]] = "ppl.platform.card.v0"
ATIF_SCHEMA_VERSION: Final[Literal["ATIF-v1.6"]] = "ATIF-v1.6"

TRAINING_LOG_REWARD_MIN: Final[float] = -20.0

JsonValue: TypeAlias = Any
MetricValue: TypeAlias = int | float | bool | str | None


class ContractModel(BaseModel):
    """Strict immutable base for cross-repo wire contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def require_json_serializable(value: Any, *, field_name: str) -> Any:
    """Reject values that cannot be represented as JSON."""

    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc
    return value

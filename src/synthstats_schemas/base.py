"""Base types for SynthStats wire contracts."""

from __future__ import annotations

import json
from typing import Annotated, Any, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, StringConstraints
from pydantic import JsonValue as PydanticJsonValue

REWARD_ARTIFACT_SCHEMA_VERSION: Final[Literal["synthstats.harbor.reward.v1"]] = (
    "synthstats.harbor.reward.v1"
)
RECORD_SCHEMA_VERSION: Final[Literal["ppl.platform.record.v0"]] = (
    "ppl.platform.record.v0"
)
TASK_SCHEMA_VERSION: Final[Literal["ppl.platform.task.v0"]] = "ppl.platform.task.v0"
CARD_SCHEMA_VERSION: Final[Literal["ppl.platform.card.v0"]] = "ppl.platform.card.v0"

TRAINING_LOG_REWARD_MIN: Final[float] = -20.0

JsonValue: TypeAlias = PydanticJsonValue
MetricValue: TypeAlias = int | float | bool | str | None

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class ContractModel(BaseModel):
    """Strict immutable base for cross-repo wire contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    def _revalidated_snapshot(self) -> "ContractModel":
        payload = BaseModel.model_dump(
            self,
            mode="python",
            round_trip=True,
            warnings="error",
        )
        snapshot = type(self).model_validate(payload)
        _restore_fields_set(self, snapshot)
        return snapshot

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize only state that still satisfies the complete contract."""

        return BaseModel.model_dump(self._revalidated_snapshot(), **kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize JSON only after revalidating mutable nested values."""

        return BaseModel.model_dump_json(self._revalidated_snapshot(), **kwargs)


def _restore_fields_set(source: Any, target: Any) -> None:
    if isinstance(source, BaseModel) and isinstance(target, BaseModel):
        object.__setattr__(
            target,
            "__pydantic_fields_set__",
            set(source.__pydantic_fields_set__),
        )
        for field_name in type(source).model_fields:
            _restore_fields_set(
                getattr(source, field_name),
                getattr(target, field_name),
            )
        return
    if isinstance(source, dict) and isinstance(target, dict):
        for key in source.keys() & target.keys():
            _restore_fields_set(source[key], target[key])
        return
    if isinstance(source, list | tuple) and isinstance(target, list | tuple):
        for source_item, target_item in zip(source, target, strict=True):
            _restore_fields_set(source_item, target_item)


def require_json_serializable(value: Any, *, field_name: str) -> Any:
    """Reject values that cannot be represented as JSON."""

    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc
    return value

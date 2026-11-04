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

TRAINING_LOG_REWARD_MIN: Final[float] = -20.0

JsonValue: TypeAlias = Any
MetricValue: TypeAlias = int | float | bool | str | None


# The internal-revalidation marker: a private object compared by identity, so
# the public validation-context parameter can never forge it.
_REVALIDATION_SENTINEL = object()


class ContractModel(BaseModel):
    """Strict immutable base for cross-repo wire contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def _revalidated_snapshot(self) -> "ContractModel":
        payload = BaseModel.model_dump(
            self,
            mode="python",
            round_trip=True,
            warnings="error",
        )
        # The sentinel marks an internal revalidation of an already-validated
        # instance: constructors that require external validation context (the
        # RL sidecar's exact trajectory bytes) skip only their external
        # byte-join on this path; every internal invariant still revalidates.
        # It is a private object checked by identity, not a truthy key: the
        # public context parameter cannot forge it.
        snapshot = type(self).model_validate(
            payload, context={"_revalidated_snapshot": _REVALIDATION_SENTINEL}
        )
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

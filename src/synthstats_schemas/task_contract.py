"""Typed program and interaction requirements shared by task consumers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Annotated, Literal, TypeGuard, cast

from pydantic import Field, JsonValue, StrictFloat, StrictInt, StrictStr, model_validator

from synthstats_schemas.base import ContractModel


class EntryPoint(ContractModel):
    function: str
    parameter: str
    returns: str


class DataPassing(ContractModel):
    container: Literal["dict"]
    column_type: Literal["numpy.ndarray"]
    row_count_key: str
    keys: tuple[str, ...]
    target_key: str

    @model_validator(mode="after")
    def require_consistent_columns(self) -> DataPassing:
        if not self.keys:
            raise ValueError("keys must be non-empty")
        if len(self.keys) != len(set(self.keys)):
            raise ValueError("keys must be unique")
        if self.row_count_key in self.keys:
            raise ValueError("keys must not contain row_count_key")
        if self.target_key not in self.keys:
            raise ValueError("target_key must be one of keys")
        return self


class ImportRule(ContractModel):
    module: str
    requires_alias: bool


class Convention(ContractModel):
    id: str
    scorer_code: str | None


class Environment(ContractModel):
    image_digest: str
    python: str
    packages: dict[str, str]


FiniteNumber = StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)]
DesignValue = int | float | str


def _finite_numeric(value: object) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class RealDomain(ContractModel):
    """Finite numeric inputs, optionally bounded on either side, inclusively."""

    kind: Literal["real"]
    minimum: FiniteNumber | None = None
    maximum: FiniteNumber | None = None
    units: str | None = None

    @model_validator(mode="after")
    def ordered_bounds(self) -> RealDomain:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self

    def accepts(self, value: object) -> bool:
        return (
            _finite_numeric(value)
            and (self.minimum is None or value >= self.minimum)
            and (self.maximum is None or value <= self.maximum)
        )


class IntegerDomain(ContractModel):
    """Integer inputs with optional inclusive bounds; booleans are not integers."""

    kind: Literal["integer"]
    minimum: StrictInt | None = None
    maximum: StrictInt | None = None
    units: str | None = None

    @model_validator(mode="after")
    def ordered_bounds(self) -> IntegerDomain:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self

    def accepts(self, value: object) -> bool:
        return (
            isinstance(value, int) and not isinstance(value, bool)
            and _finite_numeric(value)
            and (self.minimum is None or value >= self.minimum)
            and (self.maximum is None or value <= self.maximum)
        )


class NumericLevelsDomain(ContractModel):
    """Numeric levels use numeric equality, so 1 and 1.0 denote the same level."""

    kind: Literal["numeric_levels"]
    levels: Annotated[list[FiniteNumber], Field(min_length=1)]
    units: str | None = None

    @model_validator(mode="after")
    def unique_levels(self) -> NumericLevelsDomain:
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("numeric levels must be unique under numeric equality")
        return self

    def accepts(self, value: object) -> bool:
        return _finite_numeric(value) and value in self.levels


class TypedLevelsDomain(ContractModel):
    """Reference-coded levels require equality of both literal value and type."""

    kind: Literal["typed_levels"]
    levels: Annotated[list[FiniteNumber | StrictStr], Field(min_length=1)]
    units: str | None = None

    @model_validator(mode="after")
    def unique_levels(self) -> TypedLevelsDomain:
        if len({(type(level), level) for level in self.levels}) != len(self.levels):
            raise ValueError("typed levels must be unique")
        return self

    def accepts(self, value: object) -> bool:
        return any(type(value) is type(level) and value == level for level in self.levels)


DesignDomain = Annotated[
    RealDomain | IntegerDomain | NumericLevelsDomain | TypedLevelsDomain,
    Field(discriminator="kind"),
]


class PoolInteraction(ContractModel):
    """Fixed-catalogue outcomes selected adaptively or supplied as a precommitted set."""

    mode: Literal["pool"]
    query_budget: Annotated[StrictInt, Field(ge=0)]
    observation_key: str
    # The legacy adaptive default has no wire field, preserving existing hashes.
    acquisition_policy: Literal["adaptive", "precommitted"] = Field(
        default="adaptive", exclude_if=lambda value: value == "adaptive",
    )


class DesignedInteraction(ContractModel):
    """Designed observations selected adaptively or supplied as a precommitted set."""

    mode: Literal["designed"]
    query_budget: Annotated[StrictInt, Field(ge=0)]
    observation_key: str
    # The legacy adaptive default has no wire field, preserving existing hashes.
    acquisition_policy: Literal["adaptive", "precommitted"] = Field(
        default="adaptive", exclude_if=lambda value: value == "adaptive",
    )
    design_space: Annotated[
        dict[Annotated[str, Field(min_length=1)], DesignDomain], Field(min_length=1),
    ]

    def validate_design(self, design: Mapping[str, object]) -> dict[str, DesignValue]:
        """Validate the public query payload without coercing its literal types."""
        if not isinstance(design, Mapping) or design.keys() != self.design_space.keys():
            raise ValueError("design keys must equal the declared input names")
        result: dict[str, DesignValue] = {}
        for name, domain in self.design_space.items():
            value = design[name]
            if not domain.accepts(value):
                raise ValueError(f"design input {name!r} is outside its declared domain")
            result[name] = cast(DesignValue, value)
        return result


InteractionContract = Annotated[
    PoolInteraction | DesignedInteraction, Field(discriminator="mode"),
]


class ScorerRules(ContractModel):
    """A JSON snapshot of the scorer registry and the pin of its exported bytes."""

    model_config = {"allow_inf_nan": False}

    registry_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    registry: dict[str, JsonValue]


class TaskContractV1(ContractModel):
    schema_version: Literal["synthstats.task_contract.v1"]
    language: Literal["pymc"]
    entry_point: EntryPoint
    data: DataPassing
    provided_names: dict[str, str]
    allowed_imports: tuple[ImportRule, ...]
    conventions: tuple[Convention, ...]
    environment: Environment

    @property
    def interaction_mode(self) -> Literal["pool"]:
        """V1 describes fixed-catalogue acquisition and has no interaction wire field."""
        return "pool"


class TaskContract(ContractModel):
    schema_version: Literal["synthstats.task_contract.v2"]
    language: Literal["pymc"]
    entry_point: EntryPoint
    data: DataPassing
    provided_names: dict[str, str]
    allowed_imports: tuple[ImportRule, ...]
    conventions: tuple[Convention, ...]
    environment: Environment
    interaction: InteractionContract
    scorer_rules: ScorerRules | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @property
    def interaction_mode(self) -> Literal["pool", "designed"]:
        """Read the interaction declared by this V2 contract."""
        return self.interaction.mode

    @model_validator(mode="after")
    def require_consistent_interaction(self) -> TaskContract:
        if self.interaction.observation_key != self.data.target_key:
            raise ValueError("interaction observation_key must equal data target_key")
        if isinstance(self.interaction, DesignedInteraction):
            predictors = set(self.data.keys) - {self.data.target_key}
            if self.interaction.design_space.keys() != predictors:
                raise ValueError("design_space must contain exactly the program predictor columns")
        return self


def parse_task_contract(payload: str | bytes | Mapping[str, object]) -> TaskContractV1 | TaskContract:
    """Dispatch the declared wire version without upgrading or filling missing fields."""
    values = json.loads(payload) if isinstance(payload, str | bytes) else payload
    if not isinstance(values, Mapping):
        raise ValueError("task contract must be an object with schema_version")
    version = values.get("schema_version")
    if version == "synthstats.task_contract.v1":
        model = TaskContractV1
    elif version == "synthstats.task_contract.v2":
        model = TaskContract
    else:
        raise ValueError(f"unsupported task contract schema_version: {version!r}")
    if isinstance(payload, str | bytes):
        return model.model_validate_json(payload)
    return model.model_validate(payload)


def canonical_json(contract: TaskContractV1 | TaskContract) -> bytes:
    """Serialize the contract's wire fields as compact, sorted-key UTF-8 JSON."""
    return _canonical_json(contract.model_dump(mode="json"))


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def contract_sha256(contract: TaskContractV1 | TaskContract) -> str:
    """Return the hexadecimal SHA-256 of the complete canonical contract."""
    return hashlib.sha256(canonical_json(contract)).hexdigest()


def contract_template_sha256(contract: TaskContractV1 | TaskContract) -> str:
    """Identify cohort requirements, excluding task-specific columns and supports."""
    payload = contract.model_dump(
        mode="json",
        exclude={
            "data": {"keys", "target_key"},
            "interaction": {"observation_key", "design_space"},
        },
    )
    return hashlib.sha256(_canonical_json(payload)).hexdigest()

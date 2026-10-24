"""Reward provenance and reward artefact contracts."""

from __future__ import annotations

import math
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal

from pydantic import Field, field_validator, model_validator

from synthstats_schemas.base import (
    REWARD_ARTIFACT_SCHEMA_VERSION,
    TRAINING_LOG_REWARD_MIN,
    ContractModel,
    JsonValue,
    MetricValue,
    require_json_serializable,
)


class EvidenceKind(StrEnum):
    verifier = "verifier"
    heuristic = "heuristic"
    audit = "audit"


class TruthOwner(StrEnum):
    verifier = "verifier"
    heuristic = "heuristic"
    auditor = "auditor"


class RewardSourceClass(StrEnum):
    verifier_owned = "verifier_owned"
    heuristic = "heuristic"
    audit = "audit"


class RewardComponentKind(StrEnum):
    """Semantic role of one reward component inside an aggregate reward."""

    terminal_verifier = "terminal_verifier"
    terminal_heuristic = "terminal_heuristic"
    process_information_gain = "process_information_gain"
    process_rubric = "process_rubric"
    penalty = "penalty"
    diagnostic = "diagnostic"


class RewardScalarizationMethod(StrEnum):
    """How producer-owned reward components were collapsed to one log reward."""

    weighted_sum = "weighted_sum"
    verifier_gated_weighted_sum = "verifier_gated_weighted_sum"
    logaddexp = "logaddexp"
    manual = "manual"


class RewardSourceNotRegisteredError(ValueError):
    """Raised when a reward artifact names an unknown reward source."""


def _tuple_from_json_array(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return value


def _enum_from_wire_string(value: Any, enum_type: type[StrEnum]) -> Any:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    return value


class RewardSourceSpec(ContractModel):
    """Immutable provenance declaration for one reward source."""

    name: Annotated[str, Field(min_length=1)]
    evidence_kind: EvidenceKind
    truth_owner: TruthOwner
    allowed_producers: tuple[Annotated[str, Field(min_length=1)], ...]
    description: str = ""

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def evidence_kind_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, EvidenceKind)

    @field_validator("truth_owner", mode="before")
    @classmethod
    def truth_owner_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, TruthOwner)

    @field_validator("allowed_producers", mode="before")
    @classmethod
    def allowed_producers_accept_json_arrays(cls, value: Any) -> Any:
        return _tuple_from_json_array(value)

    @field_validator("name")
    @classmethod
    def name_is_clean(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError("name must not contain leading or trailing whitespace")
        return value

    @model_validator(mode="after")
    def provenance_is_consistent(self) -> "RewardSourceSpec":
        if not self.allowed_producers:
            raise ValueError("allowed_producers must not be empty")
        if (
            self.evidence_kind is EvidenceKind.verifier
            and self.truth_owner is not TruthOwner.verifier
        ):
            raise ValueError("verifier evidence must be verifier-owned")
        if (
            self.evidence_kind is EvidenceKind.heuristic
            and self.truth_owner is not TruthOwner.heuristic
        ):
            raise ValueError("heuristic evidence must be heuristic-owned")
        if (
            self.evidence_kind is EvidenceKind.audit
            and self.truth_owner is not TruthOwner.auditor
        ):
            raise ValueError("audit evidence must be auditor-owned")
        return self

    @property
    def source_class(self) -> RewardSourceClass:
        if self.evidence_kind is EvidenceKind.verifier:
            return RewardSourceClass.verifier_owned
        if self.evidence_kind is EvidenceKind.audit:
            return RewardSourceClass.audit
        return RewardSourceClass.heuristic


_CATALOG_SPECS: Final[tuple[RewardSourceSpec, ...]] = (
    RewardSourceSpec(
        name="validation.compilation",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        allowed_producers=(
            "ppl_synth_data.eval.sandbox.trivial",
            "ppl_synth_data.harbor.validator_template",
        ),
        description="Compilation or type-check verifier reward.",
    ),
    RewardSourceSpec(
        name="validation.formatting",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        allowed_producers=("ppl_synth_data.harbor.validator_template",),
        description="Structured-output formatting verifier reward.",
    ),
    RewardSourceSpec(
        name="validation.pymc_semantic_holdout_logp",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        allowed_producers=("ppl_synth_data.validation.pymc_semantics",),
        description="PyMC semantic holdout log-probability verifier reward.",
    ),
    RewardSourceSpec(
        name="validation.pymc_elpd",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        allowed_producers=("ppl_synth_data.harbor.validator_template",),
        description="PyMC held-out expected log predictive density verifier reward.",
    ),
    RewardSourceSpec(
        name="validation.lazyppl_elpd",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        allowed_producers=("ppl_synth_data.docker.lazyppl_harbor.score",),
        description="LazyPPL expected log predictive density verifier reward.",
    ),
    RewardSourceSpec(
        name="validation.expected_information_gain",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        allowed_producers=("ppl_synth_data.eval.orchestrator",),
        description="Verifier-owned Bayesian expected information gain reward.",
    ),
    RewardSourceSpec(
        name="validation.predictive_metric",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        allowed_producers=("ppl_synth_data.eval.orchestrator",),
        description="Verifier-owned predictive-quality metric reward.",
    ),
    RewardSourceSpec(
        name="heuristic.llm_rubric",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        allowed_producers=(
            "ppl_synth_data.reward.scalarization",
            "ppl_synth_data.dataset.rich_reward_export",
        ),
        description="LLM rubric score for a synthesis trace.",
    ),
    RewardSourceSpec(
        name="heuristic.pymc_elpd_error",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        allowed_producers=("ppl_synth_data.harbor.validator_template",),
        description="Heuristic floor for a PyMC ELPD scoring error.",
    ),
    RewardSourceSpec(
        name="heuristic.lazyppl_elpd_error",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        allowed_producers=("ppl_synth_data.docker.lazyppl_harbor.score",),
        description="Heuristic floor for a LazyPPL ELPD scoring error.",
    ),
    RewardSourceSpec(
        name="heuristic.no_submission",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        allowed_producers=("ppl_synth_data.eval.orchestrator",),
        description="Terminal heuristic reward for a trial without a submission.",
    ),
    RewardSourceSpec(
        name="heuristic.expected_information_gain_proxy",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        allowed_producers=("ppl_synth_data.dataset.rich_reward_export",),
        description=(
            "Heuristic proxy for trace-level information value. This is not "
            "the verifier-owned Bayesian expected-information-gain reward."
        ),
    ),
    RewardSourceSpec(
        name="heuristic.composite_reward",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        allowed_producers=(
            "ppl_synth_data.reward.scalarization",
            "ppl_synth_data.dataset.rich_reward_export",
        ),
        description="Mixed-source reward aggregate containing heuristic components.",
    ),
    RewardSourceSpec(
        name="heuristic.runtime_import",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        allowed_producers=("ppl_synth_data.runtime_registry",),
        description="Runtime import or availability heuristic.",
    ),
    RewardSourceSpec(
        name="heuristic.judge_exception",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        allowed_producers=(
            "ppl_synth_data.harbor.validator_template",
            "ppl_synth_data.eval.orchestrator",
        ),
        description="Exception while scoring a trial.",
    ),
)

_CATALOG: Final[dict[str, RewardSourceSpec]] = {
    spec.name: spec for spec in _CATALOG_SPECS
}

if len(_CATALOG) != len(_CATALOG_SPECS):
    raise RuntimeError("duplicate reward source in synthstats-schemas catalog")

REWARD_SOURCE_CATALOG: Final[MappingProxyType[str, RewardSourceSpec]] = (
    MappingProxyType(dict(_CATALOG))
)


def get_reward_source(name: str) -> RewardSourceSpec:
    try:
        return REWARD_SOURCE_CATALOG[name]
    except KeyError:
        raise RewardSourceNotRegisteredError(
            f"reward_source {name!r} is not registered in synthstats-schemas"
        ) from None


def ensure_reward_source_registered(name: str) -> RewardSourceSpec:
    return get_reward_source(name)


def list_reward_sources() -> tuple[str, ...]:
    return tuple(sorted(REWARD_SOURCE_CATALOG))


class RewardComponent(ContractModel):
    """One auditable reward component before producer-side scalarization."""

    name: Annotated[str, Field(min_length=1)]
    kind: RewardComponentKind
    value: float
    source: Annotated[str, Field(min_length=1)]
    evidence_kind: EvidenceKind
    truth_owner: TruthOwner
    producer: Annotated[str, Field(min_length=1)]
    weight: float = 1.0
    log_contribution: float | None = Field(default=None, ge=TRAINING_LOG_REWARD_MIN)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    turn_index: int | None = Field(default=None, ge=0)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("kind", mode="before")
    @classmethod
    def kind_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, RewardComponentKind)

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def evidence_kind_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, EvidenceKind)

    @field_validator("truth_owner", mode="before")
    @classmethod
    def truth_owner_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, TruthOwner)

    @field_validator("name", "source", "producer")
    @classmethod
    def strings_are_clean(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError(
                "string fields must not contain leading or trailing whitespace"
            )
        return value

    @field_validator("value", "weight", "log_contribution")
    @classmethod
    def floats_are_finite(cls, value: float | None, info: object) -> float | None:
        if value is None:
            return None
        if not math.isfinite(float(value)):
            field_name = getattr(info, "field_name", "float")
            raise ValueError(f"{field_name} must be finite")
        return float(value)

    @field_validator("metrics")
    @classmethod
    def component_metrics_are_finite(
        cls, value: dict[str, MetricValue]
    ) -> dict[str, MetricValue]:
        for key, metric in value.items():
            if isinstance(metric, bool) or metric is None or isinstance(metric, str):
                continue
            if not math.isfinite(float(metric)):
                raise ValueError(f"metric {key!r} must be finite")
        return value

    @field_validator("diagnostics", "provenance")
    @classmethod
    def component_json_fields_are_json(
        cls,
        value: dict[str, JsonValue],
        info: object,
    ) -> dict[str, JsonValue]:
        field_name = getattr(info, "field_name", "field")
        return require_json_serializable(value, field_name=field_name)

    @model_validator(mode="after")
    def provenance_matches_catalog(self) -> "RewardComponent":
        spec = get_reward_source(self.source)
        if self.evidence_kind is not spec.evidence_kind:
            raise ValueError(
                f"evidence_kind mismatch for reward component {self.name!r}"
            )
        if self.truth_owner is not spec.truth_owner:
            raise ValueError(f"truth_owner mismatch for reward component {self.name!r}")
        if self.producer not in spec.allowed_producers:
            allowed = ", ".join(spec.allowed_producers)
            raise ValueError(
                f"producer {self.producer!r} is not allowed for reward component "
                f"{self.name!r}; allowed: {allowed}"
            )
        return self


class RewardScalarization(ContractModel):
    """Producer-owned declaration for component-to-log-reward aggregation."""

    method: RewardScalarizationMethod
    component_names: tuple[Annotated[str, Field(min_length=1)], ...]
    weights: dict[str, float] = Field(default_factory=dict)
    lower_log_reward_bound: float = Field(default=TRAINING_LOG_REWARD_MIN)
    temperature: float | None = Field(default=None, gt=0.0)
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("component_names", mode="before")
    @classmethod
    def component_names_accept_json_arrays(cls, value: Any) -> Any:
        return _tuple_from_json_array(value)

    @field_validator("method", mode="before")
    @classmethod
    def method_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, RewardScalarizationMethod)

    @field_validator("weights")
    @classmethod
    def weights_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        for key, weight in value.items():
            if not math.isfinite(float(weight)):
                raise ValueError(f"weight {key!r} must be finite")
        return value

    @field_validator("lower_log_reward_bound", "temperature")
    @classmethod
    def scalarization_floats_are_finite(
        cls, value: float | None, info: object
    ) -> float | None:
        if value is None:
            return None
        if not math.isfinite(float(value)):
            field_name = getattr(info, "field_name", "float")
            raise ValueError(f"{field_name} must be finite")
        return float(value)

    @field_validator("diagnostics")
    @classmethod
    def diagnostics_are_json(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return require_json_serializable(value, field_name="diagnostics")

    @model_validator(mode="after")
    def component_names_are_unique_and_weighted(self) -> "RewardScalarization":
        if not self.component_names:
            raise ValueError("component_names must not be empty")
        if len(set(self.component_names)) != len(self.component_names):
            raise ValueError("component_names must be unique")
        missing = set(self.weights) - set(self.component_names)
        if missing:
            raise ValueError(f"weights reference unknown components: {sorted(missing)}")
        if self.lower_log_reward_bound < TRAINING_LOG_REWARD_MIN:
            raise ValueError("lower_log_reward_bound cannot undercut the ABI floor")
        return self


class RubricCriterion(ContractModel):
    """One named criterion in an auditable LLM/evolving rubric."""

    criterion_id: Annotated[str, Field(min_length=1)]
    description: Annotated[str, Field(min_length=1)]
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def weight_is_finite(cls, value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError("weight must be finite")
        return float(value)


class RewardRubric(ContractModel):
    """Immutable rubric version used by rubric-derived reward components."""

    rubric_id: Annotated[str, Field(min_length=1)]
    version: Annotated[str, Field(min_length=1)]
    title: Annotated[str, Field(min_length=1)]
    criteria: tuple[RubricCriterion, ...]
    parent_version: str | None = None
    evolution_note: str = ""
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("criteria", mode="before")
    @classmethod
    def criteria_accept_json_arrays(cls, value: Any) -> Any:
        return _tuple_from_json_array(value)

    @field_validator("provenance")
    @classmethod
    def provenance_is_json(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return require_json_serializable(value, field_name="provenance")

    @model_validator(mode="after")
    def criteria_are_nonempty_and_unique(self) -> "RewardRubric":
        if not self.criteria:
            raise ValueError("criteria must not be empty")
        ids = [criterion.criterion_id for criterion in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("criterion_id values must be unique")
        return self


class RewardArtifact(ContractModel):
    """Canonical reward payload emitted by Harbor and consumed by training."""

    schema_version: Literal["synthstats.harbor.reward.v1"] = (
        REWARD_ARTIFACT_SCHEMA_VERSION
    )
    reward_source: Annotated[str, Field(min_length=1)]
    evidence_kind: EvidenceKind
    truth_owner: TruthOwner
    producer: Annotated[str, Field(min_length=1)]
    training_log_reward: float = Field(ge=TRAINING_LOG_REWARD_MIN)
    reward_transform: dict[str, JsonValue]
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    reward: float | None = None
    components: dict[str, float] = Field(default_factory=dict)
    reward_components: tuple[RewardComponent, ...] = Field(default_factory=tuple)
    scalarization: RewardScalarization | None = None
    rubric: RewardRubric | None = None

    @field_validator("reward_components", mode="before")
    @classmethod
    def reward_components_accept_json_arrays(cls, value: Any) -> Any:
        return _tuple_from_json_array(value)

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def evidence_kind_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, EvidenceKind)

    @field_validator("truth_owner", mode="before")
    @classmethod
    def truth_owner_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, TruthOwner)

    @field_validator("training_log_reward")
    @classmethod
    def training_log_reward_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("training_log_reward must be finite")
        return value

    @field_validator("reward_transform", "diagnostics", "provenance")
    @classmethod
    def object_fields_are_json(
        cls, value: dict[str, JsonValue], info: object
    ) -> dict[str, JsonValue]:
        field_name = getattr(info, "field_name", "field")
        return require_json_serializable(value, field_name=field_name)

    @field_validator("metrics")
    @classmethod
    def metrics_are_finite(
        cls, value: dict[str, MetricValue]
    ) -> dict[str, MetricValue]:
        for key, metric in value.items():
            if isinstance(metric, bool) or metric is None or isinstance(metric, str):
                continue
            if not math.isfinite(float(metric)):
                raise ValueError(f"metric {key!r} must be finite")
        return value

    @field_validator("components")
    @classmethod
    def components_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        for key, component in value.items():
            if not math.isfinite(float(component)):
                raise ValueError(f"component {key!r} must be finite")
        return value

    @model_validator(mode="after")
    def provenance_matches_catalog(self) -> "RewardArtifact":
        spec = get_reward_source(self.reward_source)
        if self.evidence_kind is not spec.evidence_kind:
            raise ValueError(
                f"evidence_kind mismatch for reward_source={self.reward_source!r}"
            )
        if self.truth_owner is not spec.truth_owner:
            raise ValueError(
                f"truth_owner mismatch for reward_source={self.reward_source!r}"
            )
        if self.producer not in spec.allowed_producers:
            allowed = ", ".join(spec.allowed_producers)
            raise ValueError(
                f"producer {self.producer!r} is not allowed for "
                f"reward_source={self.reward_source!r}; allowed: {allowed}"
            )
        if self.scalarization is not None:
            component_names = {component.name for component in self.reward_components}
            expected_names = set(self.scalarization.component_names)
            if component_names != expected_names:
                raise ValueError(
                    "scalarization component_names must match reward_components names"
                )
        if self.rubric is not None:
            has_rubric_component = any(
                component.kind is RewardComponentKind.process_rubric
                for component in self.reward_components
            )
            if not has_rubric_component:
                raise ValueError("rubric requires at least one rubric reward component")
        return self

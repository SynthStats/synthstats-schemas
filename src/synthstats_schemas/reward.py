"""Reward provenance and reward artifact contracts."""

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


class Disposition(StrEnum):
    """Canonical verifier trial disposition across Harbor, schemas, and gates."""

    scored = "scored"
    refused = "refused"
    unjudged = "unjudged"
    # The trial never produced a reward because our infrastructure failed
    # (a timeout, a resource fault, an unresolved execution, an unreliable
    # sampler). Null values exclude infrastructure faults from training and scoring.
    infrastructure = "infrastructure"


UNJUDGED_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "task_data_invalid",
        "task_score_metadata_invalid",
        "invalid_reward_reference",
        "reference_transform_nonfinite",
        "verifier_contract_failure",
        "verifier_l5_error",
        "verifier_timeout",
        "verifier_resource_failure",
        "verifier_execution_unresolved",
        "verifier_predictive_call_error",
        "failure_origin_unresolved",
        "unsupported_verifier_profile",
        "sampler_unreliable",
        "metric_unavailable",
    }
)

INFRASTRUCTURE_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "verifier_timeout",
        "verifier_resource_failure",
        "verifier_execution_unresolved",
        "verifier_predictive_call_error",
        "sampler_unreliable",
    }
)


class RewardSourceNotRegisteredError(ValueError):
    """Raised when a reward artifact names an unknown reward source."""


class RewardSourceNotTrainableError(ValueError):
    """Raised when a registered reward source must not drive training."""


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


def _reject_non_finite_json(value: Any, *, field_name: str) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain non-finite floats")
        return
    if isinstance(value, int | str):
        return
    if isinstance(value, list):
        for item in value:
            _reject_non_finite_json(item, field_name=field_name)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field_name} object keys must be non-empty strings")
            _reject_non_finite_json(item, field_name=field_name)
        return
    raise ValueError(f"{field_name} must be JSON-compatible")


def _validate_jsonish(value: Any, *, field_name: str) -> Any:
    require_json_serializable(value, field_name=field_name)
    _reject_non_finite_json(value, field_name=field_name)
    return value


class RewardSourceSpec(ContractModel):
    """Immutable provenance declaration for one reward source."""

    name: Annotated[str, Field(min_length=1)]
    evidence_kind: EvidenceKind
    truth_owner: TruthOwner
    trainable: bool
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
        name="validation.pymc_elpd",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        trainable=False,
        allowed_producers=("ppl_synth_data.harbor.pymc_verifier",),
        description="Historical PyMC verifier reward from bounded baseline-normalized ELPD.",
    ),
    RewardSourceSpec(
        name="validation.lazyppl_elpd",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        trainable=False,
        allowed_producers=(
            "ppl_synth_data.docker.lazyppl_harbor.score",
            "ppl_synth_data.harbor.lazyppl_verifier",
        ),
        description="Historical LazyPPL verifier reward from bounded baseline-normalized ELPD.",
    ),
    RewardSourceSpec(
        name="validation.pymc_semantic_holdout_logp",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        trainable=False,
        allowed_producers=("ppl_synth_data.validation.pymc_semantics",),
        description="Historical PyMC semantic holdout log-probability verifier reward.",
    ),
    RewardSourceSpec(
        name="validation.compilation",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        trainable=True,
        allowed_producers=(
            "ppl_synth_data.validation.compilation",
            "ppl_synth_data.harbor.validator_template",
        ),
        description="Compilation or type-check verifier reward.",
    ),
    RewardSourceSpec(
        name="validation.formatting",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        trainable=True,
        allowed_producers=(
            "ppl_synth_data.validation.formatting",
            "ppl_synth_data.harbor.validator_template",
        ),
        description="Structured-output formatting verifier reward.",
    ),
    RewardSourceSpec(
        name="validation.expected_information_gain",
        evidence_kind=EvidenceKind.verifier,
        truth_owner=TruthOwner.verifier,
        trainable=True,
        allowed_producers=(
            "ppl_synth_data.eval.metrics.eig",
            "ppl_synth_data.eval.orchestrator",
        ),
        description="Verifier-owned Bayesian expected information gain reward.",
    ),
    RewardSourceSpec(
        name="heuristic.llm_rubric",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=True,
        allowed_producers=(
            "ppl_synth_data.validation.llm_rubric",
            "ppl_synth_data.eval.rubrics.llm",
        ),
        description="LLM rubric or critique-derived heuristic reward.",
    ),
    RewardSourceSpec(
        name="heuristic.expected_information_gain_proxy",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=True,
        allowed_producers=(
            "ppl_synth_data.validation.process_information_gain",
            "ppl_synth_data.reward.scalarization",
        ),
        description=(
            "Heuristic proxy for trace-level information value. This is not "
            "the verifier-owned Bayesian expected-information-gain reward."
        ),
    ),
    RewardSourceSpec(
        name="heuristic.composite_reward",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=True,
        allowed_producers=("ppl_synth_data.reward.scalarization",),
        description="Mixed-source reward aggregate containing heuristic components.",
    ),
    RewardSourceSpec(
        name="heuristic.runtime_import",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=False,
        allowed_producers=("ppl_synth_data.runtime_registry",),
        description="Runtime import or availability heuristic.",
    ),
    RewardSourceSpec(
        name="heuristic.postprocess_fix_pilot_programs",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=False,
        allowed_producers=("ppl_synth_data.scripts.fix_pilot_programs",),
        description="Post-processing heuristic applied to pilot programs.",
    ),
    RewardSourceSpec(
        name="heuristic.lazyppl_elpd_error",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=False,
        allowed_producers=(
            "ppl_synth_data.harbor.templates.test_lazyppl",
            "ppl_synth_data.docker.lazyppl_harbor.score",
        ),
        description="LazyPPL verifier failure artifact.",
    ),
    RewardSourceSpec(
        name="heuristic.pymc_elpd_error",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=False,
        allowed_producers=("ppl_synth_data.harbor.pymc_verifier",),
        description=(
            "Explicit non-trainable artifact emitted while candidate-side "
            "PyMC held-out verification is unavailable."
        ),
    ),
    RewardSourceSpec(
        name="heuristic.judge_exception",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=False,
        allowed_producers=(
            "ppl_synth_data.eval.orchestrator",
            "ppl_synth_data.harbor.validator_template",
        ),
        description="Exception while scoring a trial.",
    ),
    RewardSourceSpec(
        name="heuristic.no_submission",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=False,
        allowed_producers=("ppl_synth_data.eval.orchestrator",),
        description="Trial ended without a submitted program.",
    ),
    RewardSourceSpec(
        name="unjudged",
        evidence_kind=EvidenceKind.heuristic,
        truth_owner=TruthOwner.heuristic,
        trainable=False,
        allowed_producers=(
            "ppl_synth_data.gates.reward_variance",
            "ppl_synth_data.harbor.pymc_verifier",
            "ppl_synth_data.harbor.validator_template",
        ),
        description=(
            "Non-trainable disposition sentinel when a trial or gate sample "
            "could not be judged."
        ),
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


def ensure_reward_source_trainable(name: str) -> RewardSourceSpec:
    spec = get_reward_source(name)
    if not spec.trainable:
        raise RewardSourceNotTrainableError(
            f"reward_source {name!r} is registered but not trainable"
        )
    return spec


def list_reward_sources() -> tuple[str, ...]:
    return tuple(sorted(REWARD_SOURCE_CATALOG))


class RewardComponent(ContractModel):
    """One auditable reward component before producer-side scalarization."""

    name: Annotated[str, Field(min_length=1)]
    kind: RewardComponentKind
    value: float | None
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
        return _validate_jsonish(value, field_name=field_name)

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
        return _validate_jsonish(value, field_name="diagnostics")

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
        return _validate_jsonish(value, field_name="provenance")

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
    training_log_reward: float | None = Field(ge=TRAINING_LOG_REWARD_MIN)
    reward_transform: dict[str, JsonValue]
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    reward: float | None = None
    components: dict[str, float | None] = Field(default_factory=dict)
    reward_components: tuple[RewardComponent, ...] = Field(default_factory=tuple)
    scalarization: RewardScalarization | None = None
    rubric: RewardRubric | None = None

    @model_validator(mode="before")
    @classmethod
    def backfill_components_from_typed(cls, data: Any) -> Any:
        """Derive the legacy ``components`` map when a producer omits it.

        Producers that emit only typed ``reward_components`` (for example the
        ppl_synth_data Harbor validator template) carry the same information;
        an explicitly supplied map is still checked against the typed list.
        """
        if not isinstance(data, dict) or data.get("components"):
            return data
        typed = data.get("reward_components")
        if not typed:
            return data
        derived: dict[str, Any] = {}
        for component in typed:
            if isinstance(component, dict):
                if "value" not in component:
                    return data
                name, value = component.get("name"), component.get("value")
            else:
                if not hasattr(component, "value"):
                    return data
                name = getattr(component, "name", None)
                value = getattr(component, "value", None)
            if name is None:
                return data
            derived[name] = value
        return {**data, "components": derived}

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
    def training_log_reward_is_finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("training_log_reward must be finite")
        return value

    @field_validator("reward")
    @classmethod
    def reward_is_finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("reward must be finite")
        return value

    @field_validator("reward_transform", "diagnostics", "provenance")
    @classmethod
    def object_fields_are_json(
        cls, value: dict[str, JsonValue], info: object
    ) -> dict[str, JsonValue]:
        field_name = getattr(info, "field_name", "field")
        return _validate_jsonish(value, field_name=field_name)

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
    def components_are_finite(
        cls, value: dict[str, float | None]
    ) -> dict[str, float | None]:
        for key, component in value.items():
            if component is None:
                continue
            if not math.isfinite(float(component)):
                raise ValueError(f"component {key!r} must be finite")
        return value

    @model_validator(mode="after")
    def provenance_matches_catalog(self) -> "RewardArtifact":
        spec = get_reward_source(self.reward_source)
        raw_disposition = (
            self.diagnostics.get("disposition")
            if isinstance(self.diagnostics, dict)
            else None
        )
        if raw_disposition is not None:
            if not isinstance(raw_disposition, str):
                raise ValueError("diagnostics.disposition must be a valid Disposition string")
            _enum_from_wire_string(raw_disposition, Disposition)
        is_unjudged = raw_disposition == Disposition.unjudged
        declares_unjudged = (
            is_unjudged
            or self.reward_source == Disposition.unjudged
            or any(
                c.source == Disposition.unjudged
                or (
                    isinstance(c.diagnostics, dict)
                    and c.diagnostics.get("disposition") == Disposition.unjudged
                )
                for c in self.reward_components
            )
        )
        has_null_reward_or_component = (
            self.training_log_reward is None
            or any(v is None for v in self.components.values())
            or any(c.value is None for c in self.reward_components)
        )
        is_infrastructure = raw_disposition == Disposition.infrastructure
        if not is_unjudged and not is_infrastructure and has_null_reward_or_component:
            raise ValueError(
                "null training_log_reward or component value is only permitted "
                "when diagnostics.disposition is 'unjudged' or 'infrastructure'"
            )
        if is_infrastructure:
            # Null values exclude infrastructure faults from training and scoring.
            numeric = []
            if self.training_log_reward is not None:
                numeric.append("training_log_reward")
            if self.reward is not None:
                numeric.append("reward")
            numeric.extend(
                f"components[{name!r}]"
                for name, value in self.components.items()
                if value is not None
            )
            numeric.extend(
                f"reward_components[{component.name!r}].value"
                for component in self.reward_components
                if component.value is not None
            )
            numeric.extend(
                f"reward_components[{component.name!r}].log_contribution"
                for component in self.reward_components
                if component.log_contribution is not None
            )
            if numeric:
                raise ValueError(
                    "an infrastructure fault carries null values, never numeric "
                    f"ones; numeric fields: {', '.join(numeric)}"
                )
            reason_code = (
                self.diagnostics.get("reason_code")
                if isinstance(self.diagnostics, dict)
                else None
            )
            if (
                not isinstance(reason_code, str)
                or reason_code not in INFRASTRUCTURE_REASON_CODES
            ):
                raise ValueError(
                    "infrastructure reward artifact requires diagnostics.reason_code "
                    f"in INFRASTRUCTURE_REASON_CODES; got {reason_code!r}"
                )
            # No early return: the producer, evidence, ownership, coherence
            # and scalarization checks below apply to infrastructure rows too.
        if declares_unjudged:
            if not is_unjudged:
                raise ValueError(
                    "unjudged reward artifact requires diagnostics.disposition == 'unjudged'"
                )
            if spec.trainable or any(
                get_reward_source(c.source).trainable for c in self.reward_components
            ):
                raise ValueError(
                    "unjudged reward artifact must not use a trainable reward_source"
                )
            reason_code = (
                self.diagnostics.get("reason_code")
                if isinstance(self.diagnostics, dict)
                else None
            )
            if not isinstance(reason_code, str) or reason_code not in UNJUDGED_REASON_CODES:
                raise ValueError(
                    "unjudged reward artifact requires diagnostics.reason_code "
                    f"in UNJUDGED_REASON_CODES; got {reason_code!r}"
                )
            if self.diagnostics.get("training_target_status") != "absent":
                raise ValueError(
                    "unjudged reward artifact requires "
                    "diagnostics.training_target_status == 'absent'"
                )
            if self.diagnostics.get("refusal_code") is not None:
                raise ValueError(
                    "unjudged reward artifact requires diagnostics.refusal_code is None"
                )
            if (
                not isinstance(self.reward_transform, dict)
                or self.reward_transform.get("authoritative") is not False
            ):
                raise ValueError(
                    "unjudged reward artifact requires reward_transform.authoritative is False"
                )
            if self.reward is not None or self.training_log_reward is not None:
                raise ValueError(
                    "unjudged reward artifact requires reward is None and "
                    "training_log_reward is None"
                )
            if any(v is not None for v in self.components.values()) or any(
                c.value is not None or c.log_contribution is not None
                for c in self.reward_components
            ):
                raise ValueError(
                    "unjudged reward artifact requires null component value and "
                    "log_contribution"
                )
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

        if self.reward_components:
            typed_components = {
                component.name: component.value for component in self.reward_components
            }
            if len(typed_components) != len(self.reward_components):
                raise ValueError("reward_components names must be unique")
            if set(self.components) != set(typed_components):
                raise ValueError(
                    "components must exactly match reward_components names"
                )
            mismatched_values = sorted(
                name
                for name, value in self.components.items()
                if value != typed_components[name]
            )
            if mismatched_values:
                raise ValueError(
                    "components values must exactly match reward_components values: "
                    f"{mismatched_values}"
                )

        if self.reward_source == "heuristic.composite_reward":
            if not self.reward_components:
                raise ValueError(
                    "heuristic.composite_reward requires non-empty reward_components"
                )
            if self.scalarization is None:
                raise ValueError("heuristic.composite_reward requires scalarization")

        component_names = {component.name for component in self.reward_components}
        if self.scalarization is not None:
            expected_names = set(self.scalarization.component_names)
            if component_names != expected_names:
                raise ValueError(
                    "scalarization component_names must match reward_components names"
                )

        if self.rubric is not None:
            has_rubric_component = any(
                component.kind is RewardComponentKind.process_rubric
                or component.source == "heuristic.llm_rubric"
                for component in self.reward_components
            )
            if not has_rubric_component:
                raise ValueError("rubric requires at least one rubric reward component")

        declares_aggregation = (
            bool(self.reward_components) or self.scalarization is not None
        )
        if spec.trainable and declares_aggregation:
            if not self.reward_components:
                raise ValueError(
                    "trainable scalarization requires non-empty reward_components"
                )
            if self.scalarization is None:
                raise ValueError(
                    "trainable reward_components require auditable scalarization"
                )
            for component in self.reward_components:
                ensure_reward_source_trainable(component.source)
            if self.scalarization.method not in {
                RewardScalarizationMethod.weighted_sum,
                RewardScalarizationMethod.verifier_gated_weighted_sum,
            }:
                raise ValueError(
                    "trainable reward scalarization method is not auditable"
                )
            if set(self.scalarization.weights) != component_names:
                raise ValueError(
                    "trainable reward scalarization weights must exactly cover "
                    "reward_components"
                )
            for component in self.reward_components:
                if self.scalarization.weights[component.name] != component.weight:
                    raise ValueError(
                        "trainable reward scalarization weights must exactly match "
                        "reward component weights"
                    )
                if component.log_contribution is None:
                    raise ValueError(
                        "trainable reward scalarization requires "
                        "component log_contribution values"
                    )
            if (
                self.scalarization.method
                is RewardScalarizationMethod.verifier_gated_weighted_sum
            ):
                declared_terminal_verifiers = [
                    component
                    for component in self.reward_components
                    if component.kind is RewardComponentKind.terminal_verifier
                ]
                if not declared_terminal_verifiers:
                    raise ValueError(
                        "verifier_gated_weighted_sum requires a terminal verifier "
                        "component"
                    )
                if any(
                    component.evidence_kind is not EvidenceKind.verifier
                    or component.truth_owner is not TruthOwner.verifier
                    for component in declared_terminal_verifiers
                ):
                    raise ValueError(
                        "terminal verifier gates must be verifier-owned evidence"
                    )
            if self.training_log_reward is None:
                raise ValueError(
                    "trainable reward scalarization requires finite training_log_reward"
                )
            expected_log_reward = sum(
                self.scalarization.weights[component.name]
                * component.log_contribution
                for component in self.reward_components
                if component.log_contribution is not None
            )
            if (
                self.scalarization.method
                is RewardScalarizationMethod.verifier_gated_weighted_sum
            ):
                if any(
                    component.log_contribution
                    == self.scalarization.lower_log_reward_bound
                    for component in declared_terminal_verifiers
                ):
                    expected_log_reward = self.scalarization.lower_log_reward_bound
            expected_log_reward = max(
                expected_log_reward,
                self.scalarization.lower_log_reward_bound,
            )
            if not math.isclose(
                self.training_log_reward,
                expected_log_reward,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "trainable reward training_log_reward must match auditable "
                    "scalarization"
                )
        return self

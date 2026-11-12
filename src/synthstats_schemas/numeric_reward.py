"""Wire shapes for the numeric PyMC verifier's scored and refused rewards."""

import math
from typing import Annotated, Any, Iterable, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from synthstats_schemas.base import ContractModel, JsonValue
from synthstats_schemas.canonical import canonical_json_bytes
from synthstats_schemas.verdicts import CascadeLayer

NUMERIC_ADMISSION_CONTRACT = "synthstats.harbor.numeric_admission.v1"
Finite = Annotated[float, Field(allow_inf_nan=False)]
NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonnegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PyMCProvenance(ContractModel):
    runtime: Literal["pymc"]
    verifier: Literal["harbor_pymc_l0_l6"]
    profile: Literal["strict"]
    admission_contract: Literal["synthstats.harbor.numeric_admission.v1"]


class CompilationProvenance(ContractModel):
    runtime: Literal["pymc"]
    verifier: Literal["harbor_pymc_l0_l5"]
    profile: Literal["strict"]
    admission_contract: Literal["synthstats.harbor.numeric_admission.v1"]


class ScoreMetrics(ContractModel):
    avg_holdout_logp: Finite
    holdout_elpd: Finite
    train_elpd_loo: Finite | None
    pareto_k_max: Finite | None
    baseline_score: Finite
    gold_score: Finite | None
    temperature: PositiveFinite
    passed_layers: NonnegativeInt
    total_layers: NonnegativeInt

    @model_validator(mode="after")
    def layer_counts_agree(self) -> "ScoreMetrics":
        if self.passed_layers > self.total_layers:
            raise ValueError("passed_layers cannot exceed total_layers")
        return self


class ScoreComponentMetrics(ContractModel):
    baseline_score: Finite
    temperature: PositiveFinite
    train_elpd_loo: Finite | None
    pareto_k_max: Finite | None
    hidden_n_obs: PositiveInt


class AuthoritativeTransform(ContractModel):
    authoritative: bool

    @field_validator("authoritative")
    @classmethod
    def must_be_authoritative(cls, value: bool) -> bool:
        if not value:
            raise ValueError("numeric verifier reward must be authoritative")
        return value


class ScoreTransform(AuthoritativeTransform):
    kind: Literal["baseline_normalized_log_reward"]
    baseline_score: Finite
    gold_score: Finite | None
    temperature: PositiveFinite
    lower_log_reward_bound: Literal[-20.0]


class ReferenceScores(ContractModel):
    baseline_score: Finite
    gold_score: Finite | None


class ScoreScalarization(ContractModel):
    method: Literal["weighted_sum"]
    component_names: list[str]
    weights: dict[str, Finite]
    lower_log_reward_bound: Literal[-20.0]
    temperature: PositiveFinite
    diagnostics: ReferenceScores


class ScoreComponent(ContractModel):
    name: Literal["holdout_elpd"]
    kind: Literal["terminal_verifier"]
    value: Finite
    source: Literal["validation.pymc_elpd"]
    evidence_kind: Literal["verifier"]
    truth_owner: Literal["verifier"]
    producer: Literal["ppl_synth_data.harbor.pymc_verifier"]
    weight: Finite
    log_contribution: Finite
    metrics: ScoreComponentMetrics
    diagnostics: dict[str, JsonValue]
    provenance: PyMCProvenance


class RefusalMetrics(ContractModel):
    passed_layers: NonnegativeInt
    total_layers: NonnegativeInt
    train_elpd_loo: Finite | None = None
    pareto_k_max: Finite | None = None


class RefusalTransform(AuthoritativeTransform):
    kind: Literal["baseline_normalized_log_reward"]
    lower_log_reward_bound: Literal[-20.0]
    error_policy: Literal["candidate_refusal_zero"]


class RefusalScalarization(ContractModel):
    method: Literal["manual"]
    component_names: list[str]
    weights: dict[str, Finite]
    lower_log_reward_bound: Literal[-20.0]


class RefusalComponent(ContractModel):
    name: Literal["pymc_elpd_error"]
    kind: Literal["penalty"]
    value: Finite
    source: Literal["validation.pymc_elpd"]
    evidence_kind: Literal["verifier"]
    truth_owner: Literal["verifier"]
    producer: Literal["ppl_synth_data.harbor.pymc_verifier"]
    weight: Finite
    log_contribution: Finite
    metrics: dict[str, JsonValue]
    diagnostics: dict[str, JsonValue]
    provenance: PyMCProvenance


class RefusalDensityGate(ContractModel):
    name: Literal["min_predictive_scale"]
    applicable: bool | None
    assessed: bool
    passed: bool | None
    status: Literal[
        "unknown", "unassessed", "applied", "not_applicable_discrete",
        "not_applicable_policy_removed", "not_applicable_no_acquired_scale",
    ]
    data_std: NonnegativeFinite | None
    min_pred_std: NonnegativeFinite | None
    threshold: NonnegativeFinite | None
    detail: str

    @model_validator(mode="after")
    def status_matches_detail(self) -> "RefusalDensityGate":
        if self.detail != f"density_gate_status={self.status}":
            raise ValueError("density-gate detail must match its status")
        return self


class CompilationMetrics(ContractModel):
    cascade_reward: Finite
    passed_layers: NonnegativeInt
    total_layers: NonnegativeInt


class CompilationTransform(AuthoritativeTransform):
    kind: Literal["binary_log_reward"]
    pass_log_reward: Finite
    lower_log_reward_bound: Literal[-20.0]


class CompilationScalarizationDiagnostics(ContractModel):
    reward_transform: Literal["binary_log_reward"]


class CompilationScalarization(ContractModel):
    method: Literal["weighted_sum"]
    component_names: list[str]
    weights: dict[str, Finite]
    lower_log_reward_bound: Literal[-20.0]
    diagnostics: CompilationScalarizationDiagnostics


class CompilationComponentDiagnostics(ContractModel):
    failed_layer: CascadeLayer
    failure_reason: str
    semantic_status: Literal["critical"]
    disposition: Literal["refused"]
    reason_code: str

    @field_validator("failed_layer", mode="before")
    @classmethod
    def _cascade_layer(cls, value: Any) -> Any:
        """The wire carries the layer's name; CascadeLayer is the one enum.

        The strict contract takes only enum instances, so the recorded
        name is admitted through the enum.
        """
        return CascadeLayer(value) if isinstance(value, str) else value


class PolicyRefusedComponentDiagnostics(ContractModel):
    """The policy refusal's component diagnostics: no failed_layer.

    A policy refusal names no cascade layer: the policy refused after the
    verifier passed passed_layers of total_layers.
    """

    failure_reason: str
    semantic_status: Literal["critical"]
    disposition: Literal["refused"]
    reason_code: str


class CompilationComponent(ContractModel):
    name: Literal["compilation_cascade"]
    kind: Literal["terminal_verifier"]
    value: Finite
    source: Literal["validation.compilation"]
    evidence_kind: Literal["verifier"]
    truth_owner: Literal["verifier"]
    producer: Literal["ppl_synth_data.harbor.validator_template"]
    weight: Finite
    confidence: Finite
    log_contribution: Finite
    metrics: CompilationMetrics
    diagnostics: CompilationComponentDiagnostics
    provenance: CompilationProvenance


class CompilationRefusedReward(ContractModel):
    # The row kind: a literal selects between this row and the
    # policy refusal through the discriminated union below. The default
    # keeps the retained refusal rows (recorded before the kind existed)
    # on this member; new producer rows state it.
    kind: Literal["compilation_refusal"] = "compilation_refusal"
    schema_version: Literal["synthstats.harbor.reward.v1"]
    reward_source: Literal["validation.compilation"]
    evidence_kind: Literal["verifier"]
    truth_owner: Literal["verifier"]
    producer: Literal["ppl_synth_data.harbor.validator_template"]
    training_log_reward: Finite
    reward: Finite
    provenance: CompilationProvenance
    diagnostics: dict[str, JsonValue]
    components: dict[str, Finite]
    metrics: CompilationMetrics
    reward_transform: CompilationTransform
    reward_components: Annotated[list[CompilationComponent], Field(min_length=1, max_length=1)]
    scalarization: CompilationScalarization

    @model_validator(mode="after")
    def copies_agree(self) -> "CompilationRefusedReward":
        component = self.reward_components[0]
        summary = component.diagnostics
        if (
            self.reward != 0.0
            or self.training_log_reward != -20.0
            or self.reward_transform.pass_log_reward != 0.0
            or self.metrics.cascade_reward != 0.0
            or self.components != {component.name: 0.0}
            or component.value != 0.0
            or component.weight != 1.0
            or component.confidence != 1.0
            or component.log_contribution != -20.0
            or component.provenance != self.provenance
            or self.scalarization.component_names != [component.name]
            or self.scalarization.weights != {component.name: 1.0}
            or any(
                self.diagnostics.get(key) != getattr(summary, key)
                for key in (
                    "failed_layer", "failure_reason", "semantic_status", "disposition", "reason_code"
                )
            )
        ):
            raise ValueError("compilation refusal copies disagree")
        return self


class PolicyRefusedComponent(ContractModel):
    name: Literal["compilation_cascade"]
    kind: Literal["terminal_verifier"]
    value: Finite
    source: Literal["validation.compilation"]
    evidence_kind: Literal["verifier"]
    truth_owner: Literal["verifier"]
    producer: Literal[
        "ppl_synth_data.harbor.validator_template", "ppl_synth_data.harbor.pymc_verifier",
    ]
    weight: Finite
    confidence: Finite
    log_contribution: Finite
    metrics: CompilationMetrics
    diagnostics: PolicyRefusedComponentDiagnostics
    provenance: CompilationProvenance


def _refusal_copies_agree(
    row: Any, component: Any, *, extras: Iterable[bool] = (), message: str
) -> None:
    """The refusal law's single home (the per-contract copies fold here):
    reward 0, training log -20, weight 1, log_contribution -20 and the
    scalarization's one-component shape. Every per-contract copies_agree calls
    it with its contract's extras and its own raise word, so a law change
    edits this alone."""
    if not (
        row.reward == 0.0
        and row.training_log_reward == -20.0
        and row.components == {component.name: 0.0}
        and component.value == 0.0
        and component.weight == 1.0
        and component.log_contribution == -20.0
        and component.provenance == row.provenance
        and row.scalarization.component_names == [component.name]
        and row.scalarization.weights == {component.name: 1.0}
        and all(extras)
    ):
        raise ValueError(message)


def _score_copies_agree(
    row: Any, component: Any, *, extras: Iterable[bool] = (), message: str
) -> None:
    """The scored law's single home (the same fold): the temperature rule and
    every copy that must agree with the metrics."""
    baseline, gold = row.metrics.baseline_score, row.metrics.gold_score
    expected_temperature = (
        abs(gold - baseline) + 1e-8 if gold is not None else max(abs(baseline), 1.0)
    )
    # Both producers round ELPD to four decimals and the two rewards to six.
    # Their transform uses the unrounded ELPD, so check the possible interval.
    elpd = row.metrics.holdout_elpd
    temperature = row.metrics.temperature

    def normalized(score: float) -> float:
        difference = score - baseline
        return (difference / temperature if math.isfinite(difference)
                else score / temperature - baseline / temperature)

    lo = max(normalized(math.nextafter(elpd - 0.00005, -math.inf)), -20.0)
    hi = max(normalized(math.nextafter(elpd + 0.00005, math.inf)), -20.0)
    lo = max(lo, math.nextafter(row.training_log_reward - 0.0000005, -math.inf))
    hi = min(hi, math.nextafter(row.training_log_reward + 0.0000005, math.inf))
    lower_reward = math.exp(min(lo, 20.0))
    upper_reward = math.exp(min(hi, 20.0))
    if (lo > hi or row.reward < 0 or row.reward + 0.0000005 < lower_reward
            or row.reward - 0.0000005 > upper_reward):
        raise ValueError("scored reward does not follow the declared transform")
    if not (
        row.metrics.temperature == expected_temperature
        and row.reward_transform.baseline_score == baseline
        and row.reward_transform.gold_score == gold
        and row.reward_transform.temperature == row.metrics.temperature
        and row.scalarization.temperature == row.metrics.temperature
        and row.scalarization.diagnostics.baseline_score == baseline
        and row.scalarization.diagnostics.gold_score == gold
        and row.scalarization.component_names == [component.name]
        and row.scalarization.weights == {component.name: 1.0}
        and component.weight == 1.0
        and component.value == row.metrics.holdout_elpd
        and component.log_contribution == row.training_log_reward
        and row.components == {component.name: component.value}
        and component.metrics.baseline_score == baseline
        and component.metrics.temperature == row.metrics.temperature
        and component.provenance == row.provenance
        and all(extras)
    ):
        raise ValueError(message)


class PolicyRefusedReward(ContractModel):
    """The policy refusal's own row kind.

    The policy refused after the verifier passed passed_layers of
    total_layers: the row carries those counts and no failed_layer. The
    refusal is the policy's own act and keeps the refusal floor, so the
    copies agree on the floor exactly as the compilation refusal does.
    """

    kind: Literal["policy_refusal"]
    schema_version: Literal["synthstats.harbor.reward.v1"]
    reward_source: Literal["validation.compilation"]
    evidence_kind: Literal["verifier"]
    truth_owner: Literal["verifier"]
    producer: Literal[
        "ppl_synth_data.harbor.validator_template", "ppl_synth_data.harbor.pymc_verifier",
    ]
    training_log_reward: Finite
    reward: Finite
    provenance: CompilationProvenance
    diagnostics: dict[str, JsonValue]
    components: dict[str, Finite]
    metrics: CompilationMetrics
    reward_transform: CompilationTransform
    reward_components: Annotated[list[PolicyRefusedComponent], Field(min_length=1, max_length=1)]
    scalarization: CompilationScalarization

    @model_validator(mode="after")
    def copies_agree(self) -> "PolicyRefusedReward":
        component = self.reward_components[0]
        summary = component.diagnostics
        _refusal_copies_agree(
            self,
            component,
            extras=(
                self.reward_transform.pass_log_reward == 0.0,
                self.metrics.cascade_reward == 0.0,
                component.confidence == 1.0,
                all(
                    self.diagnostics.get(key) == getattr(summary, key)
                    for key in ("failure_reason", "semantic_status", "disposition", "reason_code")
                ),
            ),
            message="policy refusal copies disagree",
        )
        return self


# The refused rows of the compilation producer: a literal kind selects the
# row through the stock discriminated union. The
# sentinel shape (a policy refusal wearing a compilation refusal's
# failed_layer) has no path through this union.
RefusedCompilationReward = Annotated[
    CompilationRefusedReward | PolicyRefusedReward,
    Field(discriminator="kind"),
]

_REFUSED_COMPILATION = TypeAdapter(RefusedCompilationReward)


def validate_refused_compilation_row(value: Any) -> None:
    """Validate one refused row of the compilation producer.

    The discriminated union selects the row by its kind. A row recorded
    before the kind existed states none: it is the compilation member, so
    the discriminator's default is stamped before the selection. Validate
    only: the wire payload stays the caller's own.
    """
    if isinstance(value, dict) and "kind" not in value:
        value = {**value, "kind": "compilation_refusal"}
    _REFUSED_COMPILATION.validate_python(value)


class _PyMCEnvelope(ContractModel):
    schema_version: Literal["synthstats.harbor.reward.v1"]
    reward_source: Literal["validation.pymc_elpd"]
    evidence_kind: Literal["verifier"]
    truth_owner: Literal["verifier"]
    producer: Literal["ppl_synth_data.harbor.pymc_verifier"]
    training_log_reward: Finite
    reward: Finite
    provenance: PyMCProvenance
    diagnostics: dict[str, JsonValue]
    components: dict[str, Finite]


class PyMCScoredReward(_PyMCEnvelope):
    metrics: ScoreMetrics
    reward_transform: ScoreTransform
    reward_components: Annotated[list[ScoreComponent], Field(min_length=1, max_length=1)]
    scalarization: ScoreScalarization

    @model_validator(mode="after")
    def copies_agree(self) -> "PyMCScoredReward":
        _score_copies_agree(
            self, self.reward_components[0], message="numeric score copies disagree"
        )
        return self


class PyMCRefusedReward(_PyMCEnvelope):
    metrics: RefusalMetrics
    reward_transform: RefusalTransform
    reward_components: Annotated[list[RefusalComponent], Field(min_length=1, max_length=1)]
    scalarization: RefusalScalarization

    @model_validator(mode="after")
    def copies_agree(self) -> "PyMCRefusedReward":
        component = self.reward_components[0]
        _refusal_copies_agree(
            self,
            component,
            extras=(not component.metrics,),
            message="numeric refusal copies disagree",
        )
        diagnostics = self.diagnostics
        gates = diagnostics.get("density_gates")
        gate_fields = ("applicable", "assessed", "passed", "status", "threshold")
        if gates is not None or any(f"density_gate_{key}" in diagnostics for key in gate_fields):
            if not isinstance(gates, list) or len(gates) != 1:
                raise ValueError("numeric refusal requires one complete density gate")
            RefusalDensityGate.model_validate(gates[0])
            for key in gate_fields:
                outer = f"density_gate_{key}"
                if (
                    outer not in diagnostics
                    or canonical_json_bytes(diagnostics[outer]) != canonical_json_bytes(gates[0][key])
                ):
                    raise ValueError("density-gate copies disagree")
        return self


# The stock PyMC verifier (ppl_synth_data.harbor.pymc_verifier) builds the program
# once, fits it with pm.sample and scores the held-out rows with pm.set_data and
# pm.compute_log_likelihood. It has no L0-L6 cascade, so its rows carry no layer
# counts and declare their own contract.
STOCK_ADMISSION_CONTRACT = "synthstats.harbor.stock_admission.v1"


class StockProvenance(ContractModel):
    runtime: Literal["pymc"]
    verifier: Literal["pymc_verifier"]
    admission_contract: Literal["synthstats.harbor.stock_admission.v1"]


class StockScoreMetrics(ContractModel):
    avg_holdout_logp: Finite
    holdout_elpd: Finite
    train_elpd_loo: Finite | None
    pareto_k_max: Finite | None
    baseline_score: Finite
    gold_score: Finite | None
    temperature: PositiveFinite
    hidden_n_obs: PositiveInt


class StockScoreComponent(ContractModel):
    name: Literal["holdout_elpd"]
    kind: Literal["terminal_verifier"]
    value: Finite
    source: Literal["validation.pymc_elpd"]
    evidence_kind: Literal["verifier"]
    truth_owner: Literal["verifier"]
    producer: Literal["ppl_synth_data.harbor.pymc_verifier"]
    weight: Finite
    log_contribution: Finite
    metrics: ScoreComponentMetrics
    diagnostics: dict[str, JsonValue]
    provenance: StockProvenance


class StockRefusalMetrics(ContractModel):
    hidden_n_obs: NonnegativeInt


class StockRefusalComponent(ContractModel):
    name: Literal["pymc_elpd_error"]
    kind: Literal["penalty"]
    value: Finite
    source: Literal["validation.pymc_elpd"]
    evidence_kind: Literal["verifier"]
    truth_owner: Literal["verifier"]
    producer: Literal["ppl_synth_data.harbor.pymc_verifier"]
    weight: Finite
    log_contribution: Finite
    metrics: dict[str, JsonValue]
    diagnostics: dict[str, JsonValue]
    provenance: StockProvenance


class _StockEnvelope(_PyMCEnvelope):
    """The PyMC envelope with stock-verifier provenance."""

    provenance: StockProvenance


class StockScoredReward(_StockEnvelope):
    metrics: StockScoreMetrics
    reward_transform: ScoreTransform
    reward_components: Annotated[list[StockScoreComponent], Field(min_length=1, max_length=1)]
    scalarization: ScoreScalarization

    @model_validator(mode="after")
    def copies_agree(self) -> "StockScoredReward":
        component = self.reward_components[0]
        # The total is rounded to four decimals; the mean is emitted unrounded.
        estimated_total = self.metrics.avg_holdout_logp * self.metrics.hidden_n_obs
        if (not math.isfinite(estimated_total)
                or abs(estimated_total - self.metrics.holdout_elpd) > 0.00005 + math.ulp(estimated_total)):
            raise ValueError("stock score mean and total disagree")
        _score_copies_agree(
            self,
            component,
            extras=(
                component.metrics.hidden_n_obs == self.metrics.hidden_n_obs,
                component.metrics.train_elpd_loo == self.metrics.train_elpd_loo,
                component.metrics.pareto_k_max == self.metrics.pareto_k_max,
            ),
            message="stock score copies disagree",
        )
        return self


class StockRefusedReward(_StockEnvelope):
    metrics: StockRefusalMetrics
    reward_transform: RefusalTransform
    reward_components: Annotated[list[StockRefusalComponent], Field(min_length=1, max_length=1)]
    scalarization: RefusalScalarization

    @model_validator(mode="after")
    def copies_agree(self) -> "StockRefusedReward":
        _refusal_copies_agree(
            self,
            self.reward_components[0],
            extras=(not self.reward_components[0].metrics,),
            message="stock refusal copies disagree",
        )
        return self

"""Contracts for choosing one of N candidate programs generated across model
families for supervised fine-tuning (SFT).

ELPD means expected log predictive density, a score for predictive fit.
A selection record documents the checks and scores used for ELPD-guided
selection of one program from a candidate population:

- no candidate enters fitness ranking unless it passes the cheap crash gate,
  every integrity gate, and every density-degeneracy gate (eligibility first);
- ranking is enabled only when ``holdout_n >= MIN_HOLDOUT_FOR_SELECTION``;
  below that threshold, the record stores eligibility without ranking;
- the ranking margin uses a bootstrap confidence interval that accounts for
  posterior draws, held-out observations and repeated data splits.

The same record serves the PyMC and LazyPPL selection verifiers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Final, Literal

from pydantic import Field, field_validator, model_validator

from synthstats_schemas.base import ContractModel

SELECTION_RECORD_SCHEMA_VERSION: Final[Literal["synthstats.selection.record.v1"]] = (
    "synthstats.selection.record.v1"
)
SOURCE_PLAN_CARD_SCHEMA_VERSION: Final[
    Literal["synthstats.selection.source_plan.v1"]
] = "synthstats.selection.source_plan.v1"

# Minimum held-out observations required before argmax ranking is trusted.
# Below this, candidates are kept gate-only (eligible, unranked): ranking N
# programs by held-out log-likelihood over 2-3 rows is split-selection noise.
MIN_HOLDOUT_FOR_SELECTION: Final[int] = 8


class SelectionRuntime(StrEnum):
    pymc = "pymc"
    lazyppl = "lazyppl"


class SelectionMode(StrEnum):
    """Outcome of running one candidate through the selection verifier."""

    selected = "selected"  # eligible, holdout adequate, argmax winner
    gate_only = "gate_only"  # eligible, holdout too small to rank
    rejected = "rejected"  # failed a gate; never enters ranking


def _tuple_from_json_array(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _enum_from_wire_string(value: Any, enum_type: type[StrEnum]) -> Any:
    if isinstance(value, str) and not isinstance(value, enum_type):
        return enum_type(value)
    return value


class GateOutcome(ContractModel):
    """Result of one gate layer (crash gate, an integrity layer, a density check)."""

    name: Annotated[str, Field(min_length=1)]
    passed: bool
    detail: str = ""


class FitnessRecord(ContractModel):
    """Held-out fitness used to rank eligible candidates.

    ``metric`` is the held-out scoring rule (e.g. ``avg_holdout_logp``), NOT the
    cheap in-sample crash-gate ELPD. ``runner_up_margin`` is the bootstrap-CI
    gap to the second-best eligible candidate; ``None`` when there is no runner
    up (single eligible candidate).
    """

    metric: Annotated[str, Field(min_length=1)]
    draws: Annotated[int, Field(ge=1)]
    holdout_n: Annotated[int, Field(ge=1)]
    n_splits: Annotated[int, Field(ge=1)] = 1
    point_estimate: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    runner_up_margin: float | None = None

    @model_validator(mode="after")
    def ci_brackets_estimate(self) -> "FitnessRecord":
        if not (self.bootstrap_ci_low <= self.point_estimate <= self.bootstrap_ci_high):
            raise ValueError(
                "point_estimate must lie within [bootstrap_ci_low, bootstrap_ci_high]"
            )
        return self


class CandidateKey(ContractModel):
    """Widened dedup / quota key: one program candidate within a task population.

    Best-of-N produces N candidates per task across model families, so the key
    is (task_id, family, candidate_idx); a bare task_id would mis-attribute
    sibling candidates as duplicates of one another.
    """

    task_id: Annotated[str, Field(min_length=1)]
    family: Annotated[str, Field(min_length=1)]
    candidate_idx: Annotated[int, Field(ge=0)]


class SourcePlanCard(ContractModel):
    """Private generation-time structural plan for one task.

    Never serialized into the trainee-visible prompt (it would leak the intended
    distribution family); it drives the generator and the validation recipe so
    richness is planned, not hoped for. ``n_floor`` raises the EnvSpec
    observation count for selection-intended tasks so the held-out split is large
    enough to rank.
    """

    schema_version: Literal["synthstats.selection.source_plan.v1"] = (
        SOURCE_PLAN_CARD_SCHEMA_VERSION
    )
    runtime: SelectionRuntime
    cell_key: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    structural_signature: Annotated[str, Field(min_length=1)]
    required_structure: tuple[str, ...] = ()
    allowed_degrade_ladder: tuple[str, ...] = ()
    forbidden_fallbacks: tuple[str, ...] = ()
    observable_cues: tuple[str, ...] = ()
    validation_recipe: Annotated[str, Field(min_length=1)]
    n_floor: Annotated[int, Field(ge=1)]

    @field_validator("runtime", mode="before")
    @classmethod
    def runtime_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, SelectionRuntime)

    @field_validator(
        "required_structure",
        "allowed_degrade_ladder",
        "forbidden_fallbacks",
        "observable_cues",
        mode="before",
    )
    @classmethod
    def sequences_accept_json_arrays(cls, value: Any) -> Any:
        return _tuple_from_json_array(value)


class SelectionRecord(ContractModel):
    """Audit trail for selecting (or rejecting) one candidate program.

    Export-time invariant (enforced by consumers): no row ships without a
    SelectionRecord. The validators here enforce the eligibility and
    holdout-size invariants so a reward-hacked or under-evidenced program cannot
    be silently promoted to SFT ground truth.
    """

    schema_version: Literal["synthstats.selection.record.v1"] = (
        SELECTION_RECORD_SCHEMA_VERSION
    )
    runtime: SelectionRuntime
    candidate_key: CandidateKey
    selection_mode: SelectionMode
    eligible: bool
    cheap_gate: GateOutcome
    integrity_gates: tuple[GateOutcome, ...]
    density_gates: tuple[GateOutcome, ...]
    fitness: FitnessRecord | None = None
    source_plan_signature: Annotated[str, Field(min_length=1)]
    winner_reason: str = ""

    @field_validator("runtime", mode="before")
    @classmethod
    def runtime_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, SelectionRuntime)

    @field_validator("selection_mode", mode="before")
    @classmethod
    def mode_accepts_wire_string(cls, value: Any) -> Any:
        return _enum_from_wire_string(value, SelectionMode)

    @field_validator("integrity_gates", "density_gates", mode="before")
    @classmethod
    def gate_sequences_accept_json_arrays(cls, value: Any) -> Any:
        return _tuple_from_json_array(value)

    @model_validator(mode="after")
    def eligibility_and_mode_are_consistent(self) -> "SelectionRecord":
        all_gates_pass = (
            self.cheap_gate.passed
            and all(g.passed for g in self.integrity_gates)
            and all(g.passed for g in self.density_gates)
        )
        # Eligibility is exactly "passed every gate": no candidate may be marked
        # eligible while any gate failed, and none may be marked ineligible while
        # all gates passed. This is the "eligibility before ranking" invariant.
        if self.eligible != all_gates_pass:
            raise ValueError(
                "eligible must equal (cheap_gate passed AND all integrity_gates "
                "passed AND all density_gates passed)"
            )

        if self.selection_mode is SelectionMode.rejected:
            if self.eligible:
                raise ValueError("rejected candidate must not be eligible")
            if self.fitness is not None:
                raise ValueError("rejected candidate must not carry a fitness record")
            return self

        # selected / gate_only both require eligibility.
        if not self.eligible:
            raise ValueError(
                f"selection_mode={self.selection_mode.value} requires an eligible candidate"
            )

        if self.selection_mode is SelectionMode.selected:
            if self.fitness is None:
                raise ValueError("selected candidate must carry a fitness record")
            if self.fitness.holdout_n < MIN_HOLDOUT_FOR_SELECTION:
                raise ValueError(
                    "selected candidate requires "
                    f"holdout_n >= {MIN_HOLDOUT_FOR_SELECTION}; got "
                    f"{self.fitness.holdout_n}. Use gate_only below this threshold."
                )
        return self

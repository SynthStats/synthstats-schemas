"""Launch gate for SynthStats training runs.

Call ``check_launch`` before renting a GPU, calling a paid training API or starting a sandbox.
A field that does not apply to a run is declared null, [] or "not_applicable"; the group rules below say
which fields those may be, so an invented value and a forgotten one are both refused.
"""

from __future__ import annotations

import netrc
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, ValidationError, ValidationInfo, field_validator, model_validator

from synthstats_schemas.base import ContractModel

WANDB_OFF_MODES = frozenset({"disabled", "offline", "dryrun"})
SKYRL_DEFAULT_RUN_NAME = "test_run"
QUERY_TOOL = "query"
INAPPLICABLE_WORDS = frozenset({"none", "not_applicable"})
IGNORED = object()  # a binding's expected value when the native field has no meaning for this run
UNSUPPORTED = object()  # a binding's expected value when the backend cannot run the declared setting
_ABSENT = object()

Count = Annotated[int, Field(ge=1)]
NonNegativeCount = Annotated[int, Field(ge=0)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
DropoutRate = Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)]
TopP = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
Names = Annotated[list[str], Field(min_length=1)]


def _undeclared(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() in ("", "???"))


def _inapplicable(value: Any) -> bool:
    return value is None or value == [] or (isinstance(value, str) and value in INAPPLICABLE_WORDS)


class AxisGroup(ContractModel):
    """One axis group. A null, an empty string or Hydra's "???" placeholder, alone or inside a list, leaves a
    field undeclared, except that null declares "does not apply" for a field named in ``nullable``; the
    group's rules then check it."""

    nullable: ClassVar[frozenset[str]] = frozenset()

    @field_validator("*", mode="before")
    @classmethod
    def _declared(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None and info.field_name in cls.nullable:
            return value
        if _undeclared(value) or (isinstance(value, list) and any(_undeclared(item) for item in value)):
            raise ValueError("undeclared")
        return value


class ModelAxis(AxisGroup):
    family: str
    size: str
    thinking: bool
    variant: str
    revision: str
    chat_template: str


class ObjectiveAxis(AxisGroup):
    name: str
    implementation: str
    reference: str
    loss_reduction: str
    beta: NonNegativeFloat
    log_z: str
    opd_stage: bool


class AdapterAxis(AxisGroup):
    nullable: ClassVar[frozenset[str]] = frozenset({"rank", "alpha", "dropout"})

    kind: Literal["lora", "full"]
    rank: int | None
    alpha: PositiveFloat | None
    scaling: str
    dropout: DropoutRate | None
    initialization: str
    target_policy: str

    @model_validator(mode="after")
    def _fields_match_kind(self) -> AdapterAxis:
        settings = (self.alpha, self.scaling, self.dropout, self.initialization)
        if self.kind == "lora":
            # SkyRL turns LoRA on only for a rank above 0, so a "lora" run at rank 0 would train every weight.
            if self.rank is None or self.rank < 1 or any(_inapplicable(value) for value in settings):
                raise ValueError("lora needs a rank of at least 1 and concrete alpha, scaling, dropout and "
                                 "initialization")
        elif not all(_inapplicable(value) for value in (self.rank, *settings)):
            raise ValueError("full fine-tuning has no LoRA values: rank, alpha and dropout are null, scaling "
                             "and initialization not_applicable")
        return self


class RewardComponent(AxisGroup):
    name: str
    kind: Literal["heldout_fit", "marginal_likelihood", "llm_judge", "eig", "discovery", "format",
                  "budget_penalty"]


# A veto set names refusal categories; any other name is refused before a launch.
RefusalCategory = Literal["L0_parse", "L1_ast_safety", "L2_model_build", "L3_harbor_pymc_strict_profile",
                          "L4_exploit_heuristics", "L5_data_perturbation", "binding_refused", "density_gate_failed",
                          "nonfinite", "unjudged", "isolation_failed", "sampling_failed"]


class RewardAxis(AxisGroup):
    """validity_gate turns the integrity veto on: a refused row in ``integrity_categories`` earns neither fit nor judge credit.
    ``refused_judge_cap`` limits the judge credit of rows that train as refused; null declares that there is no cap."""

    nullable: ClassVar[frozenset[str]] = frozenset({"refused_judge_cap"})

    components: Annotated[list[RewardComponent], Field(min_length=1)]
    combine: Literal["weighted_sum", "product", "per_component_normalised"]
    weights: dict[str, Finite]
    validity_gate: bool
    refused_judge_cap: Probability | None
    integrity_categories: list[RefusalCategory]

    @model_validator(mode="after")
    def _one_weight_per_component(self) -> RewardAxis:
        names = [component.name for component in self.components]
        if len(set(names)) != len(names) or set(self.weights) != set(names):
            raise ValueError("the reward needs uniquely named components and one weight for each")
        return self

    @model_validator(mode="after")
    def _veto_and_cap_can_run(self) -> RewardAxis:
        if len(set(self.integrity_categories)) != len(self.integrity_categories):
            raise ValueError("the veto set names a category twice")
        if self.validity_gate != bool(self.integrity_categories):
            raise ValueError("the validity gate is on exactly when the veto set names its categories")
        if self.refused_judge_cap is not None and not any(c.kind == "llm_judge" for c in self.components):
            raise ValueError("a cap on the judge credit of refused programs needs an llm_judge component")
        return self


class JudgeAxis(AxisGroup):
    nullable: ClassVar[frozenset[str]] = frozenset({"rubric_dropout", "max_judge_masked_share", "attempts"})

    models: list[str]
    rubric: str
    rubric_sha256: str
    rubric_mode: str
    rubric_evolution: str
    rubric_dropout: Probability | None
    # The trainer stops the run when a larger share of one update's rows than this has no judge verdict after the attempts below.
    max_judge_masked_share: Probability | None
    # Judge calls per row, the first one included. The share above is a share of rows left without a verdict by all of them.
    attempts: Count | None
    eval_judge: str
    granularity: str

    @model_validator(mode="after")
    def _mode_matches_evolution(self) -> JudgeAxis:
        if self.rubric_mode == "evolving":
            agrees = not _inapplicable(self.rubric_evolution)
        else:
            known = self.rubric_mode == "static" or _inapplicable(self.rubric_mode)
            agrees = known and _inapplicable(self.rubric_evolution)
        if not agrees:
            raise ValueError("rubric_mode is static or evolving, and only an evolving rubric names its evolution")
        return self


class EnvAxis(AxisGroup):
    environments: Names
    mode: str
    query_budget: NonNegativeCount
    initial_observations: NonNegativeCount
    observation_noise: NonNegativeFloat
    split: str


class ToolsAxis(AxisGroup):
    exposed: list[str]
    call_format: str
    max_turns: Count
    max_tokens_per_turn: Count
    sandbox: str
    named_control: bool

    @model_validator(mode="after")
    def _agent_or_named_control(self) -> ToolsAxis:
        # A multi-turn tool-using run without tools is
        # admitted only as the named control.
        settings = (self.call_format, self.sandbox)
        if not self.exposed:
            if not self.named_control or not all(_inapplicable(value) for value in settings):
                raise ValueError("a run without tools is the named control, with call_format and sandbox "
                                 "not_applicable")
        elif self.named_control:
            raise ValueError("the named tool-less control exposes tools")
        elif QUERY_TOOL not in self.exposed or self.max_turns < 2 or any(_inapplicable(v) for v in settings):
            # One turn leaves no room to call a tool and read its result before submitting.
            raise ValueError("a tool-using run needs the query tool, max_turns >= 2, a call_format and a sandbox")
        return self


class RolloutAxis(AxisGroup):
    samples_per_task: Count
    temperature: NonNegativeFloat
    top_p: TopP
    asynchronous: bool
    max_staleness: NonNegativeCount
    off_policy_correction: str


class OptimAxis(AxisGroup):
    lr: PositiveFloat
    batch_size: Count
    steps: Count
    warmup: NonNegativeCount
    seed: NonNegativeCount  # one launch is one training run; replication launches more runs


class SearchAxis(AxisGroup):
    nullable: ClassVar[frozenset[str]] = frozenset({"allocation_usd"})

    method: Literal["none", "optuna_tpe", "wandb_bayes"]
    pruner: Literal["none", "successive_halving", "hyperband", "median"]
    stage: str
    metric: str
    direction: str
    study_id: str
    trial_id: str
    resource_schedule: str
    allocation_usd: PositiveFloat | None

    @model_validator(mode="after")
    def _trial_identified(self) -> SearchAxis:
        named = (self.stage, self.metric, self.study_id, self.trial_id, self.resource_schedule)
        if self.method == "none":
            if not all(_inapplicable(value) for value in (self.pruner, *named, self.direction, self.allocation_usd)):
                raise ValueError("a run without a search has none, null or not_applicable search settings")
        elif (any(_inapplicable(value) for value in named) or self.direction not in ("maximize", "minimize")
              or self.allocation_usd is None):
            raise ValueError("a search trial names its stage, metric, direction (maximize or minimize), study, "
                             "trial, resource schedule and allocation")
        return self


class EvalAxis(AxisGroup):
    tasks: str
    metrics: Names
    seeds: Count


class CostAxis(AxisGroup):
    gpu_type: str
    gpu_count: NonNegativeCount
    usd_ceiling: PositiveFloat
    judge_usd_ceiling: NonNegativeFloat

    @model_validator(mode="after")
    def _gpus_declared(self) -> CostAxis:
        # A hosted run (Tinker) rents no GPU; it declares none rather than inventing one.
        if _inapplicable(self.gpu_type) != (self.gpu_count == 0):
            raise ValueError("gpu_type is not_applicable exactly when gpu_count is 0")
        return self


class LoggingAxis(AxisGroup):
    backend: Literal["wandb"]
    entity: str
    project: str
    run_name: str

    @model_validator(mode="after")
    def _private_and_named(self) -> LoggingAxis:
        if self.run_name == SKYRL_DEFAULT_RUN_NAME:
            raise ValueError("run_name is SkyRL's default")
        return self


class ExperimentAxes(ContractModel):
    """Every experiment axis of a training run, plus logging for W&B."""

    model: ModelAxis
    objective: ObjectiveAxis
    adapter: AdapterAxis
    reward: RewardAxis
    judge: JudgeAxis
    env: EnvAxis
    tools: ToolsAxis
    rollout: RolloutAxis
    optim: OptimAxis
    search: SearchAxis
    eval: EvalAxis
    cost: CostAxis
    logging: LoggingAxis

    @model_validator(mode="after")
    def _groups_agree(self) -> ExperimentAxes:
        judge = self.judge
        settings = (judge.rubric, judge.rubric_sha256, judge.rubric_mode, judge.rubric_dropout,
                    judge.max_judge_masked_share, judge.attempts, judge.granularity)
        if any(component.kind == "llm_judge" for component in self.reward.components):
            if not judge.models or any(_inapplicable(value) for value in (*judge.models, *settings)):
                raise ValueError("the reward has an llm_judge component, so the judge needs its models, rubric, "
                                 "rubric_sha256, rubric_mode, rubric_dropout, max_judge_masked_share, attempts and granularity")
        elif judge.models or not all(_inapplicable(value) for value in (*settings, judge.rubric_evolution)):
            # Judge settings on a run without a judge would record a judge that never ran.
            raise ValueError("the reward has no llm_judge component, so the judge fields are [], null or "
                             "not_applicable")
        if self.tools.exposed and self.env.query_budget < 1:
            raise ValueError("tools are exposed but env.query_budget < 1")
        return self


# Field names that may appear in a refusal; any other part of an error location is a config key, which could
# hold a value pasted by mistake, so it is shown as <key>.
_LOCATION_NAMES = frozenset(name for model in (ExperimentAxes, *AxisGroup.__subclasses__())
                            for name in model.model_fields)


def _lora_only(value: Callable[[ExperimentAxes], Any]) -> Callable[[ExperimentAxes], Any]:
    return lambda axes: value(axes) if axes.adapter.kind == "lora" else IGNORED


def _tools_only(value: Callable[[ExperimentAxes], Any]) -> Callable[[ExperimentAxes], Any]:
    return lambda axes: value(axes) if axes.tools.exposed else IGNORED


# The native settings of each objective the stock backend runs. The stock route for the
# reference-regularised Dr GRPO-style objective: KL to the frozen reference in the loss, no
# group-standard-deviation normalisation. Another objective joins when its native settings are fixed.
SKYRL_OBJECTIVES: Mapping[str, Mapping[str, Any]] = {
    "dr_grpo_reference_regularised": {
        "policy_loss_type": "regular",
        "advantage_estimator": "grpo",
        "grpo_norm_by_std": False,
        "use_kl_loss": True,
        "use_kl_in_reward": False,
    },
}


def _objective_setting(field: str, *, refuse_unknown: bool = False) -> Callable[[ExperimentAxes], Any]:
    # One field reports an objective the backend cannot run; the others then have nothing to compare.
    def expected(axes: ExperimentAxes) -> Any:
        settings = SKYRL_OBJECTIVES.get(axes.objective.name)
        if settings is None:
            return UNSUPPORTED if refuse_unknown else IGNORED
        return settings[field]

    return expected


def _kl_coefficient(axes: ExperimentAxes) -> Any:
    settings = SKYRL_OBJECTIVES.get(axes.objective.name)
    return axes.objective.beta if settings is not None and settings["use_kl_loss"] else IGNORED


SKYRL_TARGET_MODULES = {"all_supported_linear": "all-linear"}

# SkyRL 0.3.0 paths, read from SkyRLTrainConfig.from_cli_overrides([]). SkyRL runs full fine-tuning at rank 0.
# Fields with backend-specific vocabularies (model path, objective, reward, Harbor agent settings) join this table
# once the SkyRL adapter fixes their names.
SKYRL_BINDINGS: Mapping[str, Callable[[ExperimentAxes], Any]] = {
    "trainer.policy.model.lora.rank": lambda axes: axes.adapter.rank if axes.adapter.kind == "lora" else 0,
    "trainer.policy.model.lora.alpha": _lora_only(lambda axes: axes.adapter.alpha),
    "trainer.policy.model.lora.dropout": _lora_only(lambda axes: axes.adapter.dropout),
    "trainer.policy.model.lora.target_modules": _lora_only(
        lambda axes: SKYRL_TARGET_MODULES.get(axes.adapter.target_policy, axes.adapter.target_policy)),
    "trainer.policy.optimizer_config.lr": lambda axes: axes.optim.lr,
    "trainer.policy.optimizer_config.num_warmup_steps": lambda axes: axes.optim.warmup,
    "trainer.train_batch_size": lambda axes: axes.optim.batch_size,
    "trainer.seed": lambda axes: axes.optim.seed,
    "trainer.algorithm.temperature": lambda axes: axes.rollout.temperature,
    "trainer.algorithm.loss_reduction": lambda axes: axes.objective.loss_reduction,
    "trainer.algorithm.policy_loss_type": _objective_setting("policy_loss_type", refuse_unknown=True),
    "trainer.algorithm.advantage_estimator": _objective_setting("advantage_estimator"),
    "trainer.algorithm.grpo_norm_by_std": _objective_setting("grpo_norm_by_std"),
    "trainer.algorithm.use_kl_loss": _objective_setting("use_kl_loss"),
    "trainer.algorithm.use_kl_in_reward": _objective_setting("use_kl_in_reward"),
    "trainer.algorithm.kl_loss_coef": _kl_coefficient,
    "trainer.logger": lambda axes: axes.logging.backend,
    "trainer.project_name": lambda axes: axes.logging.project,
    "trainer.run_name": lambda axes: axes.logging.run_name,
    "generator.max_turns": lambda axes: axes.tools.max_turns,
    "generator.n_samples_per_prompt": lambda axes: axes.rollout.samples_per_task,
    # A synthstats runtime field, not a stock SkyRL default; the run config sets it from the same deployment value.
    "generator.sandbox_provider": _tools_only(lambda axes: axes.tools.sandbox),
    "generator.sampling_params.temperature": lambda axes: axes.rollout.temperature,
    "generator.sampling_params.top_p": lambda axes: axes.rollout.top_p,
}

# The Tinker driver passes the values it actually uses under SkyRL's names ; Tinker
# manages LoRA alpha, dropout and target modules itself, and logs through its own W&B call.
TINKER_BINDINGS: Mapping[str, Callable[[ExperimentAxes], Any]] = {
    key: SKYRL_BINDINGS[key] for key in (
        "trainer.policy.model.lora.rank", "trainer.policy.optimizer_config.lr",
        "trainer.policy.optimizer_config.num_warmup_steps", "trainer.train_batch_size", "trainer.seed",
        "trainer.project_name", "trainer.run_name", "generator.max_turns", "generator.n_samples_per_prompt",
        "generator.sampling_params.temperature", "generator.sampling_params.top_p")
}

# The gate, not the launcher, holds each backend's table, so a launcher cannot pass a weaker list.
BACKENDS: Mapping[str, Mapping[str, Callable[[ExperimentAxes], Any]]] = {
    "skyrl_0_3": SKYRL_BINDINGS,
    "tinker": TINKER_BINDINGS,
}


class LaunchRefused(RuntimeError):
    """A training launch is refused; ``problems`` lists (where, reason) pairs, which hold no secret value."""

    def __init__(self, problems: Sequence[tuple[str, str]]) -> None:
        self.problems = tuple(problems)
        super().__init__("LAUNCH REFUSED: " + "; ".join(f"{where}: {reason}" for where, reason in self.problems))


def check_launch(
    resolved: Mapping[str, Any],
    *,
    environ: Mapping[str, str],
    netrc_path: Path | None,
    backend: str,
    wandb_probe: Callable[[str, str], None],
    allowed_wandb_entities: Collection[str],
    trainer_root: Sequence[str] = (),
) -> ExperimentAxes:
    """Return the declared axes of ``resolved``, or raise LaunchRefused listing every problem found.

    ``backend`` names a table in BACKENDS; each native field it lists, read under ``trainer_root``, must hold
    the value that the axes require there, because the trainer uses that value. ``wandb_probe(entity,
    project)`` must raise unless the W&B key can write to that project and the project is private; it runs
    last, and only for an otherwise valid launch, so a refused config makes no call.
    ``allowed_wandb_entities`` is an explicit account allowlist; an empty collection refuses every account.
    """
    problems: list[tuple[str, str]] = []
    raw = resolved.get("axes")
    axes = None
    try:
        axes = ExperimentAxes.model_validate(raw if isinstance(raw, Mapping) else {})
    except ValidationError as error:
        problems.extend(_axis_problems(error))
    if environ.get("WANDB_MODE", "").strip().lower() in WANDB_OFF_MODES:
        problems.append(("WANDB_MODE", "wandb_off"))
    if not _has_wandb_credential(environ, netrc_path):
        problems.append(("wandb_credential", "missing"))
    bindings = BACKENDS.get(backend)
    if bindings is None:
        problems.append(("backend", "unregistered"))
    if axes is not None:
        if axes.logging.entity not in allowed_wandb_entities:
            problems.append(("axes.logging.entity", "not_allowed"))
        # SkyRL 0.3.0 calls wandb.init without an entity, so the environment decides where the run lands.
        if environ.get("WANDB_ENTITY", "").strip() != axes.logging.entity:
            problems.append(("WANDB_ENTITY", "differs_from_axes"))
        native = _lookup(resolved, trainer_root)
        for key, expected in (bindings or {}).items():
            required = expected(axes)
            if required is IGNORED:
                continue
            if required is UNSUPPORTED:
                problems.append((key, "unsupported_by_backend"))
                continue
            actual = _lookup(native, key.split("."))
            if actual is _ABSENT:
                problems.append((key, "not_bound"))
            elif not _same(actual, required):
                problems.append((key, "mismatch"))
    if axes is not None and not problems:
        try:
            wandb_probe(axes.logging.entity, axes.logging.project)
        except Exception as error:  # its message can carry account details, so only the class is reported
            problems.append(("wandb_probe", type(error).__name__))
    if problems:
        raise LaunchRefused(problems)
    return axes


def _same(actual: Any, required: Any) -> bool:
    # True == 1 and 32.0 == 32 in Python, so types must agree; an int may stand for a declared float.
    if isinstance(required, float) and not isinstance(actual, bool):
        return isinstance(actual, (int, float)) and actual == required
    return type(actual) is type(required) and actual == required


def _axis_problems(error: ValidationError) -> Iterator[tuple[str, str]]:
    for item in error.errors():
        parts = [str(part) if isinstance(part, int) or part in _LOCATION_NAMES else "<key>" for part in item["loc"]]
        reason = str(item.get("ctx", {}).get("error", item["msg"])) if item["type"] == "value_error" else item["type"]
        yield (".".join(["axes", *parts])), reason


def _has_wandb_credential(environ: Mapping[str, str], netrc_path: Path | None) -> bool:
    """True when WANDB_API_KEY is set, or the netrc file has an api.wandb.ai entry with a password.

    Only presence is checked; the value is never returned or printed.
    """
    if environ.get("WANDB_API_KEY", "").strip():
        return True
    if netrc_path is None:
        return False
    try:
        entry = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError):
        return False
    return bool(entry and entry[2])


def _lookup(tree: Any, path: Sequence[str]) -> Any:
    node = tree
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return _ABSENT
        node = node[part]
    return node

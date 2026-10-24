"""The launch gate refuses a training run unless every experiment axis is
declared and Weights & Biases logging is enabled.

"""

from __future__ import annotations

import ast
from importlib.metadata import distribution
from pathlib import Path

import pytest

from synthstats_schemas import launch as gate

# The twelve axis groups of ExperimentAxes, plus logging.
AXIS_GROUPS = ("model", "objective", "adapter", "reward", "judge", "env", "tools",
               "rollout", "optim", "search", "eval", "cost", "logging")

# A refused row in one of
# these categories earns neither fit nor judge credit. The other refusal categories keep bounded judge credit.
DEFAULT_VETO = ("L1_ast_safety", "L2_model_build", "L3_harbor_pymc_strict_profile", "L4_exploit_heuristics",
                  "L5_data_perturbation", "binding_refused", "density_gate_failed", "isolation_failed")

# A complete declaration: a multi-turn agent with a 50/50 fit and judge reward.
FULL_AXES = {
    "model": {"family": "qwen3.5", "size": "4B", "thinking": False, "variant": "instruct",
              "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "chat_template": "qwen3.5-default"},
    "objective": {"name": "dr_grpo_reference_regularised", "implementation": "skyrl_0_3:dr_grpo",
                  "reference": "frozen_initial_policy", "loss_reduction": "token_mean", "beta": 0.25,
                  "log_z": "not_applicable", "opd_stage": False},
    # alpha follows the rank, as an int, the way a config that sets alpha from rank resolves it.
    "adapter": {"kind": "lora", "rank": 32, "alpha": 32, "scaling": "standard_alpha_over_rank", "dropout": 0.0,
                "initialization": "standard_zero_initial_update", "target_policy": "all_supported_linear"},
    "reward": {"components": [{"name": "fit", "kind": "heldout_fit"}, {"name": "judge", "kind": "llm_judge"}],
               "combine": "weighted_sum", "weights": {"fit": 0.5, "judge": 0.5}, "validity_gate": True,
               "refused_judge_cap": None, "integrity_categories": list(DEFAULT_VETO)},
    "judge": {"models": ["glm-5.3-flash-free"], "rubric": "rubric-v1", "rubric_sha256": "a" * 64,
              "rubric_mode": "static", "rubric_evolution": "not_applicable", "rubric_dropout": 0.0,
              "max_judge_masked_share": 0.25, "attempts": 3, "eval_judge": "none", "granularity": "binary"},
    "env": {"environments": ["example-pool"], "mode": "acquisition", "query_budget": 8,
            "initial_observations": 2, "observation_noise": 0.0, "split": "by-source-group"},
    "tools": {"exposed": ["list_candidates", "query", "inspect_data", "run_program", "submit"],
              "call_format": "harbor-task", "max_turns": 16, "max_tokens_per_turn": 4096,
              "sandbox": "daytona-1cpu-4gb", "named_control": False},
    "rollout": {"samples_per_task": 8, "temperature": 1.0, "top_p": 1.0, "asynchronous": False,
                "max_staleness": 0, "off_policy_correction": "none"},
    "optim": {"lr": 1.0e-6, "batch_size": 4, "steps": 12, "warmup": 0, "seed": 0},
    "search": {"method": "none", "pruner": "none", "stage": "not_applicable", "metric": "not_applicable",
               "direction": "not_applicable", "study_id": "not_applicable", "trial_id": "not_applicable",
               "resource_schedule": "not_applicable", "allocation_usd": None},
    "eval": {"tasks": "example-tasks", "metrics": ["heldout_utility", "judge_score"], "seeds": 1},
    "cost": {"gpu_type": "H200", "gpu_count": 1, "usd_ceiling": 60.0, "judge_usd_ceiling": 0.0},
    "logging": {"backend": "wandb", "entity": "example-user", "project": "synthstats",
                "run_name": "qwen35-4b-fit-judge-s0"},
}
FULL_FT = {"kind": "full", "rank": None, "alpha": None, "scaling": "not_applicable", "dropout": None,
           "initialization": "not_applicable", "target_policy": "all_trainable_parameters"}
JUDGE_OFF = {"models": [], "rubric": "not_applicable", "rubric_sha256": "not_applicable",
             "rubric_mode": "not_applicable", "rubric_evolution": "not_applicable", "rubric_dropout": None,
             "max_judge_masked_share": None, "attempts": None, "eval_judge": "none", "granularity": "not_applicable"}
FIT_ONLY = {"components": [{"name": "fit", "kind": "heldout_fit"}], "combine": "weighted_sum",
            "weights": {"fit": 1.0}, "validity_gate": True, "refused_judge_cap": None,
            "integrity_categories": list(DEFAULT_VETO)}
TOOL_LESS_CONTROL = {"exposed": [], "call_format": "not_applicable", "max_turns": 1, "max_tokens_per_turn": 2048,
                     "sandbox": "not_applicable", "named_control": True}
TPE_TRIAL = {"method": "optuna_tpe", "pruner": "successive_halving", "stage": "rank_lr", "metric": "dev_utility",
             "direction": "maximize", "study_id": "example-study", "trial_id": "3",
             "resource_schedule": "updates:4,8,12", "allocation_usd": 300.0}
WANDB_ON = {"WANDB_API_KEY": "test-key-not-real", "WANDB_ENTITY": "example-user"}
DELETE = object()


def _probe_ok(entity, project):
    return None


def _axes_with(**changes):
    """FULL_AXES with changes: "group" replaces a group, "group__field" a leaf; the value DELETE removes the key."""
    axes = {group: dict(fields) for group, fields in FULL_AXES.items()}
    for key, value in changes.items():
        group, _, field = key.partition("__")
        target, name = (axes[group], field) if field else (axes, group)
        if value is DELETE:
            del target[name]
        else:
            target[name] = value
    return axes


def _skyrl_native_for(axes):
    """The SkyRL 0.3.0 values a correct launcher sets for these axes: LoRA is off at rank 0, and "all-linear" is
    SkyRL's name for every supported linear layer."""
    adapter, objective, rollout = axes["adapter"], axes["objective"], axes["rollout"]
    optim, logging = axes["optim"], axes["logging"]
    lora = {"rank": adapter["rank"]} if adapter["kind"] == "lora" else {"rank": 0}
    if adapter["kind"] == "lora":
        lora.update(alpha=adapter["alpha"], dropout=adapter["dropout"], target_modules="all-linear")
    return {"trainer": {"policy": {"model": {"lora": lora},
                                   "optimizer_config": {"lr": optim["lr"], "num_warmup_steps": optim["warmup"]}},
                        "algorithm": {"temperature": rollout["temperature"],
                                      "loss_reduction": objective["loss_reduction"],
                                      # The stock route for the reference-regularised Dr GRPO-style objective
                                      # uses KL to the frozen reference in the loss, no
                                      # group-standard-deviation normalisation.
                                      "policy_loss_type": "regular", "advantage_estimator": "grpo",
                                      "grpo_norm_by_std": False, "use_kl_loss": True, "use_kl_in_reward": False,
                                      "kl_loss_coef": objective["beta"]},
                        "train_batch_size": optim["batch_size"], "seed": optim["seed"], "logger": logging["backend"],
                        "project_name": logging["project"], "run_name": logging["run_name"]},
            "generator": {"max_turns": axes["tools"]["max_turns"], "n_samples_per_prompt": rollout["samples_per_task"],
                          "sandbox_provider": axes["tools"]["sandbox"],
                          "sampling_params": {"temperature": rollout["temperature"], "top_p": rollout["top_p"]}}}


def _launch(axes, changes=None):
    """A resolved SkyRL config: the axes plus the native values; changes map dotted native paths to values."""
    try:
        native = _skyrl_native_for(axes)
    except (KeyError, TypeError):  # an incomplete declaration; the gate refuses it before comparing natives
        native = {}
    for dotted, value in (changes or {}).items():
        *parents, leaf = dotted.split(".")
        node = native
        for part in parents:
            node = node[part]
        if value is DELETE:
            del node[leaf]
        else:
            node[leaf] = value
    return {"axes": axes, **native}


def _check(resolved, environ=WANDB_ON, **options):
    options = {"backend": "skyrl_0_3", "netrc_path": None, "wandb_probe": _probe_ok, "allowed_wandb_entities": frozenset({"example-user"}), **options}
    return gate.check_launch(resolved, environ=environ, **options)


def _refused(resolved, environ=WANDB_ON, **options):
    with pytest.raises(gate.LaunchRefused) as refused:
        _check(resolved, environ, **options)
    return refused.value


def _places(refusal):
    return {place for place, _ in refusal.problems}


def test_refuses_missing_axes_and_disabled_logging():
    # A configuration without axes and with logging disabled is refused.
    resolved = {}

    refusal = _refused(resolved, environ={"WANDB_MODE": "disabled"})

    assert {f"axes.{group}" for group in AXIS_GROUPS} <= _places(refusal)
    assert {"WANDB_MODE", "wandb_credential"} <= _places(refusal)


def test_returns_the_declared_axes_for_a_fully_declared_config():
    axes = _check(_launch(FULL_AXES))

    assert axes.adapter.rank == 32
    assert axes.tools.max_turns == 16


def test_reports_every_undeclared_leaf_in_one_pass():
    # A missing key, a null, Hydra's "???" placeholder and an empty string all leave an axis undeclared.
    axes = _axes_with(adapter__rank=DELETE, judge__rubric_mode=None, model__family="???", tools__call_format="")

    assert _places(_refused(_launch(axes))) == {"axes.adapter.rank", "axes.judge.rubric_mode",
                                                "axes.model.family", "axes.tools.call_format"}


@pytest.mark.parametrize("key, value, place", [("judge__models", ["???"], "axes.judge.models"),
                                               ("tools__exposed", ["query", ""], "axes.tools.exposed"),
                                               ("eval__metrics", [None], "axes.eval.metrics")])
def test_refuses_an_undeclared_element_inside_a_list(key, value, place):
    assert _places(_refused(_launch(_axes_with(**{key: value})))) == {place}


def test_reports_values_of_the_wrong_type():
    # A YAML or JSON composer can deliver strings where numbers or booleans belong; the gate does not coerce them.
    axes = _axes_with(adapter__rank="32", model__thinking="false", optim__lr="1e-6", tools__max_turns=True)

    assert _places(_refused(_launch(axes))) == {"axes.adapter.rank", "axes.model.thinking", "axes.optim.lr",
                                                "axes.tools.max_turns"}


@pytest.mark.parametrize("key, value", [
    ("optim__lr", -1.0), ("optim__lr", float("nan")), ("optim__batch_size", 0), ("optim__steps", 0),
    ("optim__seed", -1), ("env__initial_observations", -1), ("rollout__top_p", 2.0), ("rollout__temperature", -0.5), ("rollout__samples_per_task", 0),
    ("tools__max_tokens_per_turn", 0), ("adapter__dropout", 1.5), ("judge__rubric_dropout", 1.5),
    ("judge__max_judge_masked_share", 1.5), ("judge__max_judge_masked_share", -0.1),
    ("judge__max_judge_masked_share", float("nan")),
    ("judge__attempts", 0), ("judge__attempts", -1), ("judge__attempts", True), ("judge__attempts", "3"),
    ("judge__attempts", 2.5),
    ("objective__beta", -1.0), ("env__environments", []), ("env__query_budget", -1), ("eval__metrics", []),
    ("eval__seeds", 0), ("cost__gpu_count", -1), ("cost__usd_ceiling", 0.0), ("reward__combine", "unsupported"),
])
def test_refuses_a_value_outside_its_domain(key, value):
    group, _, field = key.partition("__")

    assert _places(_refused(_launch(_axes_with(**{key: value})))) == {f"axes.{group}.{field}"}


def test_the_refusal_message_names_each_problem_and_no_secret_value():
    axes = _axes_with(adapter__rank=DELETE, model__family="???", judge__rubric_mode=None)
    environ = {"WANDB_API_KEY": "secret-value-xyz", "WANDB_ENTITY": "example-user", "WANDB_MODE": "disabled"}

    message = str(_refused(_launch(axes), environ=environ))

    assert message.startswith("LAUNCH REFUSED")
    for expected in ("axes.adapter.rank: missing", "axes.model.family: undeclared",
                     "axes.judge.rubric_mode: undeclared", "WANDB_MODE: wandb_off"):
        assert expected in message
    assert "secret-value-xyz" not in message


def test_a_refusal_never_repeats_a_dictionary_key_or_an_unknown_field_name():
    # Keys come from the config; one could hold a credential pasted by mistake.
    axes = _axes_with(reward__weights={"fit": 0.5, "judge": 0.5, "sk-secret-weight-key": "bad"},
                      optim={**FULL_AXES["optim"], "sk-secret-field-name": 1})

    refusal = _refused(_launch(axes))

    assert _places(refusal) == {"axes.reward.weights.<key>", "axes.optim.<key>"}
    assert "sk-secret" not in str(refusal)


@pytest.mark.parametrize("mode", ["disabled", "offline", "dryrun", " Disabled "])
def test_refuses_a_wandb_mode_that_keeps_the_run_off_the_dashboard(mode):
    assert _places(_refused(_launch(FULL_AXES), environ={**WANDB_ON, "WANDB_MODE": mode})) == {"WANDB_MODE"}


def test_accepts_a_netrc_entry_for_wandb_as_the_credential(tmp_path):
    netrc_file = tmp_path / "netrc"
    netrc_file.write_text("machine api.wandb.ai\n  login user\n  password netrc-secret-value\n")

    axes = _check(_launch(FULL_AXES), environ={"WANDB_ENTITY": "example-user"}, netrc_path=netrc_file)

    assert axes.logging.backend == "wandb"


def test_a_netrc_without_a_wandb_entry_is_not_a_credential(tmp_path):
    # A .netrc file that exists proves nothing: only an api.wandb.ai entry with a password counts.
    netrc_file = tmp_path / "netrc"
    netrc_file.write_text("machine github.com\n  login user\n  password other-secret-value\n")

    refusal = _refused(_launch(FULL_AXES), environ={"WANDB_ENTITY": "example-user"}, netrc_path=netrc_file)

    assert _places(refusal) == {"wandb_credential"}
    assert "other-secret-value" not in str(refusal)


@pytest.mark.parametrize("environ", [{"WANDB_API_KEY": "test-key-not-real"},
                                     {"WANDB_API_KEY": "test-key-not-real", "WANDB_ENTITY": "example-team"}])
def test_refuses_a_wandb_entity_in_the_environment_other_than_the_declared_one(environ):
    # SkyRL 0.3.0 calls wandb.init without an entity, so WANDB_ENTITY decides where the run lands; unset, the
    # default entity does.
    assert _places(_refused(_launch(FULL_AXES), environ=environ)) == {"WANDB_ENTITY"}


@pytest.mark.parametrize("field, value", [("run_name", "test_run"), ("backend", "console")])
def test_refuses_logging_that_is_shared_unnamed_or_not_wandb(field, value):
    environ = WANDB_ON
    expected = "axes.logging.backend" if field == "backend" else "axes.logging"

    assert _places(_refused(_launch(_axes_with(**{f"logging__{field}": value})), environ=environ)) == {expected}


@pytest.mark.parametrize("field, value", [("rank", 0), ("rank", -1), ("rank", None), ("alpha", None),
                                          ("dropout", None), ("scaling", "not_applicable"),
                                          ("initialization", "not_applicable")])
def test_refuses_lora_without_a_positive_rank_and_concrete_values(field, value):
    # SkyRL turns LoRA on only for a rank above 0, so "lora" at rank 0 silently trains every weight.
    assert _places(_refused(_launch(_axes_with(**{f"adapter__{field}": value})))) == {"axes.adapter"}


def test_accepts_full_fine_tuning_declared_with_inapplicable_lora_fields():
    axes = _check(_launch(_axes_with(adapter=FULL_FT)))

    assert axes.adapter.kind == "full"
    assert axes.adapter.rank is None


@pytest.mark.parametrize("field, value", [("rank", 0), ("rank", 32), ("alpha", 16.0), ("dropout", 0.0),
                                          ("scaling", "standard_alpha_over_rank")])
def test_refuses_full_fine_tuning_that_carries_a_lora_value(field, value):
    assert _places(_refused(_launch(_axes_with(adapter={**FULL_FT, field: value})))) == {"axes.adapter"}


def test_refuses_an_adapter_kind_outside_lora_and_full():
    assert _places(_refused(_launch(_axes_with(adapter__kind="qlora")))) == {"axes.adapter.kind"}


@pytest.mark.parametrize("gpu_type, gpu_count", [("H200", 0), ("not_applicable", 1)])
def test_refuses_a_gpu_type_and_count_that_disagree(gpu_type, gpu_count):
    cost = {**FULL_AXES["cost"], "gpu_type": gpu_type, "gpu_count": gpu_count}

    assert _places(_refused(_launch(_axes_with(cost=cost)))) == {"axes.cost"}


def test_accepts_a_hosted_run_that_rents_no_gpu():
    # A Tinker run trains on hosted compute, so it declares no GPU rather than inventing one.
    axes = _check(_launch(_axes_with(cost={**FULL_AXES["cost"], "gpu_type": "not_applicable", "gpu_count": 0})),
                  backend="tinker")

    assert axes.cost.gpu_count == 0


def test_refuses_reward_weights_that_do_not_cover_the_components():
    assert _places(_refused(_launch(_axes_with(reward__weights={"fit": 1.0})))) == {"axes.reward"}


# The judge-credit cap and veto set are reward settings. The gate refuses a launch
# that leaves either undeclared, and a declaration that cannot run.
@pytest.mark.parametrize("field", ["refused_judge_cap", "integrity_categories"])
def test_refuses_a_reward_that_leaves_the_cap_or_the_veto_set_out(field):
    reward = {key: value for key, value in FULL_AXES["reward"].items() if key != field}

    assert _places(_refused(_launch(_axes_with(reward=reward)))) == {f"axes.reward.{field}"}


@pytest.mark.parametrize("cap", [None, 0.0, 0.25, 1.0])
def test_accepts_a_cap_declared_as_null_or_as_a_probability(cap):
    axes = _check(_launch(_axes_with(reward__refused_judge_cap=cap)))

    assert axes.reward.refused_judge_cap == cap


@pytest.mark.parametrize("cap", [-0.1, 1.5, float("nan"), float("inf"), "0.25", "???", [0.25]])
def test_refuses_a_cap_that_is_not_null_or_a_probability(cap):
    assert _places(_refused(_launch(_axes_with(reward__refused_judge_cap=cap)))) == {"axes.reward.refused_judge_cap"}


def test_refuses_a_cap_for_a_reward_without_a_judge():
    reward = {**FIT_ONLY, "refused_judge_cap": 0.25}

    assert _places(_refused(_launch(_axes_with(reward=reward, judge=JUDGE_OFF)))) == {"axes.reward"}


def test_accepts_the_declared_veto_categories():
    axes = _check(_launch(FULL_AXES))

    assert axes.reward.integrity_categories == list(DEFAULT_VETO)
    assert len(DEFAULT_VETO) == 8 and len(set(DEFAULT_VETO)) == 8


def test_refuses_a_validity_gate_with_no_veto_set_and_a_veto_set_without_the_gate():
    gate_on = {**FULL_AXES["reward"], "integrity_categories": []}
    gate_off = {**FULL_AXES["reward"], "validity_gate": False}

    assert _places(_refused(_launch(_axes_with(reward=gate_on)))) == {"axes.reward"}
    assert _places(_refused(_launch(_axes_with(reward=gate_off)))) == {"axes.reward"}


def test_accepts_a_reward_without_a_validity_gate_and_with_an_empty_veto_set():
    reward = {**FULL_AXES["reward"], "validity_gate": False, "integrity_categories": []}

    assert _check(_launch(_axes_with(reward=reward))).reward.integrity_categories == []


@pytest.mark.parametrize("categories, place", [
    (["L1_ast_safety", "L9_made_up"], "axes.reward.integrity_categories.1"),
    (["L1_ast_safety", "L1_ast_safety"], "axes.reward"),
    (["scored"], "axes.reward.integrity_categories.0"),
])
def test_refuses_a_veto_category_that_no_refused_row_carries_and_a_repeated_one(categories, place):
    assert _places(_refused(_launch(_axes_with(reward__integrity_categories=categories)))) == {place}


def test_refuses_a_reward_component_of_an_unknown_kind():
    components = [{"name": "fit", "kind": "heldout_fit"}, {"name": "judge", "kind": "vibes"}]

    assert _places(_refused(_launch(_axes_with(reward__components=components)))) == {
        "axes.reward.components.1.kind"}


@pytest.mark.parametrize("judge", [JUDGE_OFF, {**FULL_AXES["judge"], "rubric_sha256": "not_applicable"},
                                   {**FULL_AXES["judge"], "max_judge_masked_share": None},
                                   {**FULL_AXES["judge"], "attempts": None}])
def test_refuses_an_active_judge_whose_fields_are_left_inapplicable(judge):
    refusal = _refused(_launch(_axes_with(judge=judge)))

    assert _places(refusal) == {"axes"}
    assert "judge" in refusal.problems[0][1]


# The trainer stops the run when a larger share of one update's rows than this has no judge verdict. A limit of 1.0 leaves only the
# per-prompt rule. The value is a choice, so a launch that leaves it open does not start and a run's W&B config records the choice.
@pytest.mark.parametrize("limit", [0.0, 0.25, 1.0])
def test_accepts_a_declared_judge_masked_share_limit(limit):
    axes = _check(_launch(_axes_with(judge__max_judge_masked_share=limit)))

    assert axes.judge.max_judge_masked_share == limit


@pytest.mark.parametrize("limit", [DELETE, "???", ""])
def test_refuses_an_open_judge_masked_share_limit(limit):
    refusal = _refused(_launch(_axes_with(judge__max_judge_masked_share=limit)))

    assert _places(refusal) == {"axes.judge.max_judge_masked_share"}


def test_refuses_a_judge_masked_share_limit_for_a_reward_without_a_judge():
    judge = {**JUDGE_OFF, "max_judge_masked_share": 0.25}

    refusal = _refused(_launch(_axes_with(reward=FIT_ONLY, judge=judge)))

    assert _places(refusal) == {"axes"}
    assert "judge" in refusal.problems[0][1]


# The limit above is a share of rows that have no verdict after the judge's attempts, so the number of attempts per row is part of the
# setting : with one attempt per row a healthy update would stop far more often. A launch declares it.
@pytest.mark.parametrize("attempts", [1, 3, 5])
def test_accepts_a_declared_number_of_judge_attempts(attempts):
    axes = _check(_launch(_axes_with(judge__attempts=attempts)))

    assert axes.judge.attempts == attempts


@pytest.mark.parametrize("attempts", [DELETE, "???", ""])
def test_refuses_an_open_number_of_judge_attempts(attempts):
    refusal = _refused(_launch(_axes_with(judge__attempts=attempts)))

    assert _places(refusal) == {"axes.judge.attempts"}


def test_refuses_judge_attempts_for_a_reward_without_a_judge():
    judge = {**JUDGE_OFF, "attempts": 3}

    refusal = _refused(_launch(_axes_with(reward=FIT_ONLY, judge=judge)))

    assert _places(refusal) == {"axes"}
    assert "judge" in refusal.problems[0][1]


def test_refuses_judge_values_for_a_reward_without_a_judge():
    # A fit-only run that still carries judge settings would record a judge that never ran.
    refusal = _refused(_launch(_axes_with(reward=FIT_ONLY)))

    assert _places(refusal) == {"axes"}
    assert "judge" in refusal.problems[0][1]


def test_accepts_a_fit_only_reward_with_the_judge_declared_inapplicable():
    axes = _check(_launch(_axes_with(reward=FIT_ONLY, judge=JUDGE_OFF)))

    assert axes.judge.models == []


@pytest.mark.parametrize("mode, evolution", [("evolving", "not_applicable"), ("static", "add-and-prune-v1"),
                                             ("sometimes", "not_applicable")])
def test_refuses_a_rubric_mode_that_disagrees_with_its_evolution_protocol(mode, evolution):
    axes = _axes_with(judge={**FULL_AXES["judge"], "rubric_mode": mode, "rubric_evolution": evolution})

    assert _places(_refused(_launch(axes))) == {"axes.judge"}


def test_accepts_an_evolving_rubric_with_its_protocol():
    axes = _check(_launch(_axes_with(judge__rubric_mode="evolving", judge__rubric_evolution="add-and-prune-v1")))

    assert axes.judge.rubric_evolution == "add-and-prune-v1"


@pytest.mark.parametrize("search", [{**TPE_TRIAL, "metric": "not_applicable"}, {**TPE_TRIAL, "trial_id": "none"},
                                    {**TPE_TRIAL, "direction": "up"}, {**TPE_TRIAL, "allocation_usd": None},
                                    {**TPE_TRIAL, "resource_schedule": "not_applicable"},
                                    {**FULL_AXES["search"], "metric": "dev_utility"}])
def test_refuses_a_search_declaration_that_is_incomplete_or_carries_values_without_a_search(search):
    assert _places(_refused(_launch(_axes_with(search=search)))) == {"axes.search"}


@pytest.mark.parametrize("field, value", [("method", "grid"), ("pruner", "asha")])
def test_refuses_a_search_controller_or_pruner_outside_the_supported_set(field, value):
    # The supported search methods are optuna_tpe and wandb_bayes; an unknown method must not pass as a declared search.
    assert _places(_refused(_launch(_axes_with(search={**TPE_TRIAL, field: value})))) == {f"axes.search.{field}"}


def test_accepts_a_declared_optuna_trial():
    axes = _check(_launch(_axes_with(search=TPE_TRIAL)))

    assert axes.search.trial_id == "3"


def test_refuses_exposed_tools_with_a_single_turn():
    # One turn leaves no room to call a tool and read its result before submitting.
    assert _places(_refused(_launch(_axes_with(tools__max_turns=1)))) == {"axes.tools"}


def test_refuses_tools_without_the_query_tool():
    assert _places(_refused(_launch(_axes_with(tools__exposed=["inspect_data", "submit"])))) == {"axes.tools"}


@pytest.mark.parametrize("tools", [{**TOOL_LESS_CONTROL, "named_control": False},
                                   {**FULL_AXES["tools"], "named_control": True},
                                   {**FULL_AXES["tools"], "call_format": "not_applicable"},
                                   {**FULL_AXES["tools"], "sandbox": "not_applicable"},
                                   {**TOOL_LESS_CONTROL, "call_format": "harbor-task"}])
def test_refuses_tool_settings_that_disagree_with_the_tools_exposed(tools):
    assert _places(_refused(_launch(_axes_with(tools=tools, env__query_budget=0)))) == {"axes.tools"}


def test_accepts_the_named_tool_less_control_at_one_turn():
    axes = _check(_launch(_axes_with(tools=TOOL_LESS_CONTROL, env__query_budget=0)))

    assert axes.tools.exposed == []


def test_refuses_a_tool_using_run_with_no_query_budget():
    refusal = _refused(_launch(_axes_with(env__query_budget=0)))

    assert _places(refusal) == {"axes"}
    assert "query_budget" in refusal.problems[0][1]


def test_refuses_a_backend_without_a_registered_binding_table():
    # The gate, not the launcher, says which native fields a backend must match, so a launcher cannot pass a
    # weaker list; the backend name is not echoed.
    refusal = _refused(_launch(FULL_AXES), backend="my-own-trainer")

    assert _places(refusal) == {"backend"}
    assert "my-own-trainer" not in str(refusal)


# The SkyRL 0.3.0 fields that carry declared adapter, objective, optimiser, rollout, turn and W&B settings: the
# native values and their neighbours.
SKYRL_FIELDS = ("trainer.policy.model.lora.rank", "trainer.policy.model.lora.alpha",
                "trainer.policy.model.lora.dropout", "trainer.policy.model.lora.target_modules",
                "trainer.policy.optimizer_config.lr", "trainer.policy.optimizer_config.num_warmup_steps",
                "trainer.train_batch_size", "trainer.seed", "trainer.algorithm.temperature",
                "trainer.algorithm.loss_reduction", "trainer.algorithm.policy_loss_type",
                "trainer.algorithm.advantage_estimator", "trainer.algorithm.grpo_norm_by_std",
                "trainer.algorithm.use_kl_loss", "trainer.algorithm.use_kl_in_reward", "trainer.algorithm.kl_loss_coef",
                "trainer.logger", "trainer.project_name", "trainer.run_name", "generator.max_turns",
                "generator.n_samples_per_prompt", "generator.sandbox_provider", "generator.sampling_params.temperature",
                "generator.sampling_params.top_p")
# Fields of the synthstats runtime config rather than of stock SkyRL 0.3.0's defaults.
SYNTHSTATS_RUNTIME_FIELDS = {"generator.sandbox_provider"}


def _value_at(tree, dotted):
    for part in dotted.split("."):
        tree = tree[part]
    return tree


def _other(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "-other"
    return value + 1 if isinstance(value, int) else value * 2 + 1


def test_skyrl_bindings_cover_every_skyrl_field_that_carries_a_declared_axis():
    assert set(SKYRL_FIELDS) <= set(gate.BACKENDS["skyrl_0_3"])


@pytest.mark.parametrize("field", SKYRL_FIELDS)
def test_refuses_each_skyrl_field_that_differs_from_its_axis(field):
    resolved = _launch(FULL_AXES)
    resolved = _launch(FULL_AXES, {field: _other(_value_at(resolved, field))})

    assert _refused(resolved).problems == ((field, "mismatch"),)


def test_refuses_skyrl_fields_that_contradict_or_omit_their_axes():
    # Native values must agree with the declared adapter and rollout settings.
    resolved = _launch(FULL_AXES, {"trainer.policy.model.lora.rank": 0, "trainer.policy.model.lora.alpha": 16,
                                   "generator.max_turns": DELETE})

    assert set(_refused(resolved).problems) == {("trainer.policy.model.lora.rank", "mismatch"),
                                                ("trainer.policy.model.lora.alpha", "mismatch"),
                                                ("generator.max_turns", "not_bound")}


def test_accepts_skyrl_fields_equal_to_their_axes_under_a_nested_root():
    native = _launch(FULL_AXES)
    resolved = {"axes": native.pop("axes"), "skyrl": native}

    assert _check(resolved, trainer_root=("skyrl",)).optim.lr == 1.0e-6


def test_full_fine_tuning_binds_to_skyrl_rank_zero_and_needs_no_lora_settings():
    resolved = _launch(_axes_with(adapter=FULL_FT))

    assert resolved["trainer"]["policy"]["model"]["lora"] == {"rank": 0}
    assert _check(resolved)

    resolved["trainer"]["policy"]["model"]["lora"]["rank"] = 32
    assert _refused(resolved).problems == (("trainer.policy.model.lora.rank", "mismatch"),)


@pytest.mark.parametrize("native", [True, 32.0])
def test_a_native_value_of_another_type_does_not_stand_for_a_declared_integer(native):
    # True == 1 and 32.0 == 32 in Python; SkyRL's rank field is an int.
    rank = 1 if native is True else 32
    axes = _axes_with(adapter__rank=rank, adapter__alpha=rank)

    refusal = _refused(_launch(axes, {"trainer.policy.model.lora.rank": native}))

    assert refusal.problems == (("trainer.policy.model.lora.rank", "mismatch"),)


def test_accepts_an_integer_native_value_for_a_declared_float():
    resolved = _launch(FULL_AXES, {"generator.sampling_params.temperature": 1, "trainer.algorithm.temperature": 1})

    assert _check(resolved).rollout.temperature == 1.0


def test_refuses_an_objective_the_stock_backend_cannot_run():
    # The stock backend runs the objectives in SKYRL_OBJECTIVES; any other objective is refused.
    refusal = _refused(_launch(_axes_with(objective__name="vargrad_tb")))

    assert refusal.problems == (("trainer.algorithm.policy_loss_type", "unsupported_by_backend"),)


def test_the_tool_less_control_ignores_the_sandbox_provider():
    resolved = _launch(_axes_with(tools=TOOL_LESS_CONTROL, env__query_budget=0),
                       {"generator.sandbox_provider": "local-docker"})

    assert _check(resolved).tools.exposed == []


def test_refuses_a_hard_coded_tinker_value_that_contradicts_its_axis():
    # A backend that binds a native value other than the declared axis value is refused;
    # here the declared rank is 8 and the native rank is 32.
    resolved = _launch(_axes_with(adapter__rank=8, adapter__alpha=8), {"trainer.policy.model.lora.rank": 32})

    assert _refused(resolved, backend="tinker").problems == (("trainer.policy.model.lora.rank", "mismatch"),)


def test_every_bound_field_exists_in_the_real_skyrl_config():
    # Read the installed wheel's typed declarations without importing its GPU runtime.
    path = distribution("skyrl").locate_file("skyrl/train/config/config.py")
    module = ast.parse(path.read_text())
    classes = {
        node.name: {
            field.target.id: ast.unparse(field.annotation)
            for field in node.body
            if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
        }
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }

    unbound = []
    for dotted in set(SKYRL_FIELDS).union(*gate.BACKENDS.values()) - SYNTHSTATS_RUNTIME_FIELDS:
        fields = classes["SkyRLTrainConfig"]
        for part in dotted.split("."):
            if part not in fields:
                unbound.append(dotted)
                break
            fields = classes.get(fields[part], {})
    assert unbound == []


def test_a_failing_wandb_probe_refuses_the_launch_without_its_message():
    def probe(entity, project):
        raise PermissionError("details-that-stay-private")

    refusal = _refused(_launch(FULL_AXES), wandb_probe=probe)

    assert refusal.problems == (("wandb_probe", "PermissionError"),)
    assert "details-that-stay-private" not in str(refusal)


def test_the_wandb_probe_runs_once_and_only_for_an_otherwise_valid_launch():
    calls = []

    def probe(*target):
        calls.append(target)

    _refused(_launch(_axes_with(model__family="???")), wandb_probe=probe)
    _refused(_launch(FULL_AXES), environ={**WANDB_ON, "WANDB_MODE": "disabled"}, wandb_probe=probe)
    _check(_launch(FULL_AXES), wandb_probe=probe)

    assert calls == [("example-user", "synthstats")]


def test_another_declared_account_reaches_the_required_privacy_probe():
    calls = []
    axes = _axes_with(logging__entity="example-team")
    result = _check(_launch(axes), environ={**WANDB_ON, "WANDB_ENTITY": "example-team"},
                    allowed_wandb_entities={"example-user", "example-team"},
                    wandb_probe=lambda *target: calls.append(target))
    assert result.logging.entity == "example-team"
    assert calls == [("example-team", "synthstats")]


def test_refuses_an_entity_outside_the_explicit_allowed_set():
    refusal = _refused(_launch(FULL_AXES), allowed_wandb_entities=frozenset({"another-user"}))
    assert refusal.problems == (("axes.logging.entity", "not_allowed"),)


def test_refuses_an_empty_explicit_allowed_set():
    refusal = _refused(_launch(FULL_AXES), allowed_wandb_entities=frozenset())
    assert refusal.problems == (("axes.logging.entity", "not_allowed"),)

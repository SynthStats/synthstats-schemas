# Changelog

<!-- towncrier release notes start -->

## 0.23.0 (2026-11-15)

### Terms

| Term | Meaning |
| --- | --- |
| Verifier host | The component that runs the verifier outside the sandbox of the submitted program. It includes the trusted runner. A fault of the verifier host itself, such as running out of memory, is not the program's. A failure that the host observes in the program, such as a compile error, is. |

### Added

| Name | Change | Action |
| --- | --- | --- |
| `ATTRIBUTION_TABLE` | Rows for failures that the verifier host observes. `CANDIDATE` (the program's own failure, observed in the submitted program): `submission_missing` at `L0_parse`, `submission_refused` at `L1_ast_safety`, `compile_failed`, `compile_timeout` and `compile_resources` at `L2_model_build`, `timeout`, `memory_limit`, `packet_too_large` and `packet_rejected` at `L3_harbor_pymc_strict_profile`, and `off_support`, `nonfinite`, `wrong_shape` and `non_numeric` at `L6_holdout_elpd` with status `None` (predictions that the scoring of the held-out rows refuses). `UNCHECKED` without origin evidence (a fault of the verifier host itself), all with status `"unjudged"`: `verifier_resource_failure` at `L0_parse`, `L2_model_build`, `L3_harbor_pymc_strict_profile` and `L6b_predictive_reducer`; `verifier_timeout` at `L2_model_build`, `L3_harbor_pymc_strict_profile` and `L6b_predictive_reducer`; `verifier_execution_unresolved` at `L3_harbor_pymc_strict_profile` and `L6b_predictive_reducer`; `compiler_unavailable` and `task_data_mismatch` at `L2_model_build`; `invalid_reward_declaration` at `L6_holdout_elpd`; `reducer_environment_mismatch` at `L6b_predictive_reducer`. Every earlier row keeps its answer | Call the `"unjudged"` rows with `fault_origin` or, for a timeout, `timeout_origin` from the verifier host |
| `CascadeLayer.L6B_PREDICTIVE_REDUCER` | The value `"L6b_predictive_reducer"` names the step where the trusted side turns the program's packet into the predictive score. Only the reducer's own faults use it | |
| `FaultOrigin` | Values `"candidate"`, `"trusted_runner"` and `"unresolved"`. The trusted side establishes the origin of a fault of the verifier host, and `attribution_for` applies the value it receives. Import it from `synthstats_schemas.verdicts` | |
| `UNJUDGED_REASON_CODES` | Adds `compiler_unavailable`, `invalid_reward_declaration`, `task_data_mismatch` and `reducer_environment_mismatch` | |
| `attribution_for`: `fault_origin` | Keyword-only. For `verifier_resource_failure`, `verifier_execution_unresolved`, `task_data_invalid`, `invalid_reward_reference`, `compiler_unavailable`, `task_data_mismatch`, `invalid_reward_declaration` and `reducer_environment_mismatch`, `"trusted_runner"` returns `INFRASTRUCTURE` and `"candidate"` returns `CANDIDATE`. `"unresolved"` or no value returns the table row. Other reasons do not use it to decide. The value is checked on every call: one that `FaultOrigin` does not list raises `ValueError`, whatever the reason. A call on an earlier row answers as before when `fault_origin` is absent | Pass `fault_origin="trusted_runner"` when the trusted side established that the fault arose in the verifier host or its runner |

### Changed

| Name | Change | Action |
| --- | --- | --- |
| Finite-float fields | The reward component, scalarization, rubric, reward artifact and training trajectory contracts refuse NaN and infinite floats with the same `finite_number` error as before. The shared strict contract configuration (`allow_inf_nan=False`) that every contract model carries now does the refusing, in place of per-field validators | None |
| `ImageDigest`, `Sha256` (qualified image receipt) | The generated JSON Schema of these strings carries the `pattern` and no longer carries `minLength` and `maxLength`. The pattern fixes the length, so the accepted strings are the same | Regenerate a stored copy of the JSON Schema |
| `ATIFExportManifest`, `MixedATIFAssemblyManifest` | `trial_dirs` must list each directory as many times as `trial_artifacts` does. 0.22.1 compared the two as sets, so a list that repeated a directory or omitted one of two equal directories passed | List each trial directory once for each trial artifact |
| `ATIFExportManifest`, `RichRewardATIFExportManifest` | `count_dropped` must equal the length of `dropped`. A manifest with another count is refused with `count_dropped must equal dropped length` | Set `count_dropped` to `len(dropped)` |
| `StockScoredReward` | The check that the mean and the total agree allows the four-decimal half-step plus two ulp of the product of the mean and the observation count. It allowed one ulp. A row whose total differs from that product by up to that amount now validates | None |
| `private_project_probe` | The default W&B query creates `wandb.Api()` with the SDK's default timeout. 0.22.1 passed a 30-second timeout. When the installed SDK has no `Api._service_api`, the error names the installed `wandb` version | None |

## 0.22.1 (2026-11-12)

0.22.0 and 0.22.1 are published together; when upgrading from 0.20.0, read both sections.

### Terms

| Term | Meaning |
| --- | --- |
| Admission contract | A declared or expected identifier selects contract-specific validation for verifier reward rows. |
| ATIF | Agent Trajectory Interchange Format is a JSON format for recording an agent’s messages, tool calls and observations. |
| Authoritative | A `reward_transform` flag that must be `True` in the strict verifier reward models. |
| Cascade | The verifier’s sequence of checks runs from parsing through held-out scoring. |
| Daytona | A sandbox service named in capability values. |
| Evidence receipt | A typed acceptance-bundle record identifies evidence for a capability with its path and SHA-256 hash. |
| Harbor | An agent evaluation framework providing task, trial and trajectory formats; this package pins Harbor and imports its trajectory models. |
| Producer strings | A producer is the package that writes a reward row; the listed `ppl_synth_data` strings are literal `producer` field values. |
| Refusal density gate | A refusal diagnostic records whether the minimum predictive-scale check was assessed and passed, with its status and threshold. |
| Stock and numeric rows | Stock verifier rows record held-out observation counts; numeric verifier rows record counts of passed and total verification layers. |
| Strict row or model | A contract-specific model validates a reward row, which declares its transform in `reward_transform`. |
| TB and SubTB | TB means trajectory balance and SubTB means subtrajectory balance; capability values containing them are literal names. |
| Training receipt | A structured record of a training run's dataset, objective, policy, optimisation, metrics, artifacts and weight change. |
| Trusted runner | The execution component outside the submitted program establishes the origin of a timeout. |

### Added

| Name | Change | Action |
| --- | --- | --- |
| `ATTRIBUTION_TABLE` | A dict that maps each failure triple to its `Attribution` | |
| `Attribution` | `CANDIDATE` identifies a program defect; `INFRASTRUCTURE` identifies an infrastructure fault; `UNCHECKED` means the evidence does not distinguish them. | |
| `Attribution`, `VerdictStatus`, `ATTRIBUTION_TABLE`, `attribution_for` | Package-root exports. `TimeoutOrigin`, `CascadeLayer` and `POLICY_REFUSAL_KIND` import from `synthstats_schemas.verdicts` only | |
| `AuthoritativeTransform`, `ScoreTransform`, `RefusalTransform`, `CompilationTransform` | Reward transform models. `AuthoritativeTransform` is the base class and requires `authoritative` to be true | |
| `BundleCapability`, `TrainingReceiptCapability` | Package-root exports. The types of the bundle capability values and of the training receipt capability values | |
| `CUT_STOP_REASONS` | The frozenset `{StopReason.LENGTH, StopReason.UNKNOWN}` marks unfinished generations. The string `"length"` is also a member. | |
| `CascadeLayer` | Values `"L0_parse"`, `"L1_ast_safety"`, `"L2_model_build"`, `"L3_harbor_pymc_strict_profile"`, `"L4_exploit_heuristics"`, `"L5_data_perturbation"`, `"L5b_predictor_connectivity"` and `"L6_holdout_elpd"` | |
| `CompilationComponentDiagnostics`, `PolicyRefusedComponentDiagnostics`, `RefusalDensityGate` | Diagnostics and gate models | |
| `CompilationRefusedReward` | Row with `reward_source` `"validation.compilation"` and `provenance` of type `CompilationProvenance`. `kind` is `"compilation_refusal"`, the default. Records the cascade's `failed_layer` in its diagnostics | |
| `IndependentReviewEvidence` | A member of the evidence receipt union, with `kind` `'independent_review'`. Import it from `synthstats_schemas.acceptance_bundle` | |
| `NUMERIC_ADMISSION_CONTRACT`, `STOCK_ADMISSION_CONTRACT` | `"synthstats.harbor.numeric_admission.v1"` and `"synthstats.harbor.stock_admission.v1"`. A row names its contract in `provenance.admission_contract` | |
| `POLICY_REFUSAL_KIND` | `"policy_refusal"`. A policy refusal names no failed layer, so its table key is `(None, "policy_refusal", "policy_refusal")` | |
| `PolicyRefusedReward` | A `policy_refusal` row uses the compilation source and provenance, names no failed layer, and records passed and total layers. | |
| `PyMCProvenance`, `CompilationProvenance`, `StockProvenance` | Provenance models, with verifier `"harbor_pymc_l0_l6"`, `"harbor_pymc_l0_l5"` and `"pymc_verifier"` in that order | |
| `PyMCScoredReward`, `PyMCRefusedReward` | Numeric rows use `validation.pymc_elpd`, `ppl_synth_data.harbor.pymc_verifier` and `PyMCProvenance`, with `passed_layers` and `total_layers`. | |
| `RefusedCompilationReward`, `validate_refused_compilation_row` | Union of compilation refusal rows discriminated by `kind`. `validate_refused_compilation_row` defaults a missing `kind` to `"compilation_refusal"`. | |
| `ScoreComponent`, `StockScoreComponent`, `RefusalComponent`, `StockRefusalComponent`, `CompilationComponent`, `PolicyRefusedComponent` | Reward component models | |
| `ScoreMetrics`, `ScoreComponentMetrics`, `StockScoreMetrics`, `RefusalMetrics`, `StockRefusalMetrics`, `CompilationMetrics` | Metrics models | |
| `ScoreScalarization`, `RefusalScalarization`, `CompilationScalarization`, `CompilationScalarizationDiagnostics`, `ReferenceScores` | Scalarization and reference-score models | |
| `StockScoredReward`, `StockRefusedReward` | Stock rows: the same `reward_source` and producer, `provenance` of type `StockProvenance`. They carry `hidden_n_obs` and no layer counts | |
| `StopReason` | A string enum: `STOP`, `LENGTH`, `TOOL_CALLS`, `CONTENT_FILTER`, `FUNCTION_CALL`, `ABORT`, `ERROR`, `REPETITION`, `UNKNOWN`. Values are lowercase member names. | |
| `TimeoutOrigin` | Values `"candidate"`, `"runner_deadline"`, `"runner_kill"` and `"unresolved"`. The trusted runner establishes the origin, and `attribution_for` applies the value it receives | |
| `VerdictStatus` | Values `"scored"`, `"binding_refused"`, `"density_gate_failed"`, `"nonfinite"`, `"sampling_failed"`, `"holdout_failed"`, `"timeout"`, `"unjudged"` and `"isolation_failed"` | |
| `attribution_for` | `attribution_for(failed_layer, status, reason_code, *, timeout_origin=None)` returns the `Attribution` of a failure. A triple that `ATTRIBUTION_TABLE` does not list raises `ValueError` | |
| `failed_layer`, `status`, `reason_code` | The failure layer, outcome status and reason code identify a failure. Layer and status can be `None`. | |
| `recorded_stop_reason` | Returns non-blank strings unchanged; returns `"unknown"` for other values. | |
| `reward`, `training_log_reward` | Row models check score fields for consistency. Refused rows have `reward=0.0` and `training_log_reward=-20.0`. | |
| `synthstats_schemas.launch` | Checks the declared settings of a training run before the run starts | |
| `synthstats_schemas.launch`: `AxisGroup` | Base class of the axis models. It forbids unknown fields | |
| `synthstats_schemas.launch`: `BACKENDS` | Maps each backend name to its table: `"skyrl_0_3"`, `"skyrl_0_4"`, `"skyrl_0_4_grammar"`, `"trl"` and `"tinker"`. Any other `backend` is refused with `('backend', 'unregistered')` | |
| `synthstats_schemas.launch`: `ExperimentAxes` | Groups the model, objective, adapter, reward, optional judge, environment, tools, rollout, optimization, search, evaluation, cost and logging axes. | |
| `synthstats_schemas.launch`: `INAPPLICABLE_WORDS` | `"none"` and `"not_applicable"`. A field that does not apply takes `None`, `[]` or one of these words where its group allows it | |
| `synthstats_schemas.launch`: `LaunchRefused` | Raised by `check_launch` when a check fails. `problems` is a tuple of `(where, reason)` pairs, one for each problem found | |
| `synthstats_schemas.launch`: `None`, `""`, `"???"` | Axis models reject undeclared values, including these placeholders inside lists, except where null is accepted. | |
| `synthstats_schemas.launch`: `SKYRL_BINDINGS`, `SKYRL_0_4_BINDINGS`, `TRL_BINDINGS`, `TINKER_BINDINGS` | Tables that map each trainer setting to the value that the axes require | |
| `synthstats_schemas.launch`: `SKYRL_TARGET_MODULES` | Maps the `adapter.target_policy` value `"all_supported_linear"` to `"all-linear"` | |
| `synthstats_schemas.launch`: `WANDB_MODE` | After trimming and lowercasing, a non-empty value in `environ` must equal `logging.mode` | |
| `synthstats_schemas.launch`: `allowed_wandb_entities` | Required `check_launch` argument. Rejects disallowed `axes.logging.entity`; passing a string or bytes raises `TypeError`. | |
| `synthstats_schemas.launch`: `check_launch` | Validates `resolved["axes"]` as `ExperimentAxes`, checks trainer settings against the axes, and returns the axes. | |
| `synthstats_schemas.launch`: `env.observation_noise` | Accepts null when `env.mode` is `"pool"` or `"designed"`. `check_launch` refuses null on every backend except `"trl"`, with `('axes.env.observation_noise', 'numeric_noise_required_for_backend')` | |
| `synthstats_schemas.launch`: `env.query_budget` | A non-negative integer or a non-empty list of them | |
| `synthstats_schemas.launch`: `judge` | Required when a component has kind `llm_judge`. `None` otherwise | |
| `synthstats_schemas.launch`: `logging.mode` | Online logging requires a W&B credential, matching `WANDB_ENTITY`, and a `wandb_probe` callable. | |
| `synthstats_schemas.launch`: `logging.run_name`, `logging.sync_owner` | `run_name` differs from `SKYRL_DEFAULT_RUN_NAME`, which is `"test_run"`. Offline logging requires `sync_owner` | |
| `synthstats_schemas.launch`: `not_bound`, `mismatch` | Reasons reported for a trainer setting that is absent (`not_bound`) or has another value than the axes require (`mismatch`), for example `('lora_r', 'mismatch')` | |
| `synthstats_schemas.launch`: `reward.components`, `reward.weights` | Components have unique names. `weights` holds one weight for each name. Component kinds: `exact_posterior`, `heldout_fit`, `marginal_likelihood`, `llm_judge`, `qualified_checks`, `eig`, `discovery`, `format` and `budget_penalty` | |
| `synthstats_schemas.launch`: `reward.multiplicity_correction` | Required, as `True` or `False`, when a component is `exact_posterior` | |
| `synthstats_schemas.launch`: `reward.validity_gate`, `reward.integrity_categories` | `validity_gate` is `True` exactly when `integrity_categories` is non-empty. The `RefusalCategory` values in `integrity_categories` are unique | |
| `synthstats_schemas.launch`: `search.allocation_usd` | Accepts null | |
| `synthstats_schemas.launch`: `skyrl_0_4_grammar` | Adds the settings `native.grammar.objective.name`, `native.grammar.multiplicity_correction` and `trainer.max_training_steps` | |
| `synthstats_schemas.launch`: `synthstats_schemas.launch.RewardComponent` | A reward component of the axes, with a `name` and a `kind`. A different class from `synthstats_schemas.RewardComponent` | |
| `synthstats_schemas.launch`: `tools.named_control` | A run without exposed tools sets it to `True` or sets `env.mode` to `"fixed_transcript"` | |
| `synthstats_schemas.launch`: `tools` | Exposed tools require `"query"`, null or at least 2 `max_turns`, declared `call_format` and `sandbox`, and `named_control=False`. | |
| `synthstats_schemas.launch`: `trainer_root` | A dotted setting name such as `"trainer.policy.optimizer_config.lr"` addresses nested mappings under `trainer_root` in `resolved` | |
| `synthstats_schemas.launch`: `wandb_probe` | Callable `wandb_probe(entity, project)`. Called last, after all other checks pass; exceptions become problems named by exception class. | |
| `synthstats_schemas.numeric_reward` | Models for the reward rows that the PyMC verifiers emit. Every model is frozen and forbids unknown fields | |
| `synthstats_schemas.stop_reasons` | Names the reasons a generation stops. Holds `StopReason`, `CUT_STOP_REASONS` and `recorded_stop_reason` | |
| `synthstats_schemas.verdicts` | Attributes a verifier failure to the submitted program or to the infrastructure | |
| `timeout_origin` | For `verifier_timeout`, `candidate` attributes to the program; runner deadlines and kills to infrastructure; unresolved or missing to `UNCHECKED`. Invalid values raise `ValueError`. | |

### Changed

| Name | Change | Action |
| --- | --- | --- |
| `ATIFExportManifest`, `MixedATIFAssemblyManifest` | Each key of `trial_artifacts` must equal the `atif_id` inside its entry | Key every entry of `trial_artifacts` by its own `atif_id` |
| `BundleCapability` | Includes `independent_review`. A stored bundle fails validation when its `required_capabilities` or any `capabilities` list holds a string that `typing.get_args(BundleCapability)` does not list | Compare the capabilities in your bundles with `typing.get_args(BundleCapability)` |
| `BundleTrainingReceiptEntry.capabilities` | Accepts only the eight `TrainingReceiptCapability` values: `sft_warmstart`, `offline_current_policy_tb_rescore`, `offline_current_policy_subtb_rescore`, `live_harbor_skyrl_daytona_tb`, `live_harbor_skyrl_daytona_subtb`, `tinker_provider_sft`, `tinker_provider_tb` and `tinker_provider_subtb` | Declare the other `BundleCapability` values, such as `release_grade_mixed_atif`, in `BundleFile.capabilities` or in the `capabilities` of an evidence receipt |
| `CompilationRefusedReward`, `PolicyRefusedReward` | `RewardArtifact.model_validate` selects them by the numeric contract, `validation.compilation` and the disposition `refused`. It selects `PolicyRefusedReward` when `kind` is `'policy_refusal'` | |
| `ContractModel` | Rejects `NaN` and infinite values in float fields, in every model that subclasses it. The error is `ValidationError` with the Pydantic error type `finite_number` | Pass finite numbers to float fields |
| `JsonValue` | Pydantic's recursive JSON value type. In `ContractModel` fields, non-finite floats are rejected | Convert tuples and sets to lists, dates, paths and bytes to strings, and use string keys. |
| `PostTrainingAcceptanceBundle.required_capabilities` | Needs at least one entry. An empty list raises `ValidationError` with the Pydantic error type `too_short` | List at least one `BundleCapability` that the files or receipts of the bundle cover |
| `PyMCScoredReward`, `PyMCRefusedReward` | `RewardArtifact.model_validate` selects them by the numeric contract, `validation.pymc_elpd` and the disposition `scored` or `refused` | |
| `RewardArtifact.components` | An explicit empty mapping fails validation when `reward_components` is non-empty | Omit `components` to derive it, or map each component name to its value |
| `RewardArtifact.kind` | An optional `compilation_refusal` or `policy_refusal` value, omitted from dumps when `None` | Set the kind for a refusal when needed |
| `RewardArtifact.model_validate` with a declared contract | A `validation.pymc_elpd` row with a declared or expected stock or numeric admission contract needs `diagnostics['disposition']` | Set `diagnostics['disposition']` to `'scored'`, `'refused'`, `'unjudged'` or `'infrastructure'` |
| `RewardArtifact.model_validate` with a strict row | Contract, `reward_source` and disposition select a strict model. Other sources or dispositions retain general checks. | |
| `RewardArtifact.reward_components` | Matching components must copy artifact diagnostics for `validation.pymc_elpd` and metrics for `validation.compilation`, regardless of admission contract. | Copy the artifact's `diagnostics` or `metrics` into each matching component whenever you change either |
| `RewardComponent.confidence`, `RewardComponent.turn_index`, `RewardScalarization.temperature`, `RewardArtifact.rubric`, `RewardScalarization.diagnostics` | `model_dump()`, `model_dump(mode='json')` and `model_dump_json()` omit each of the first four keys when its value is `None`, and `RewardScalarization.diagnostics` when it is empty | Read these keys with `dump.get('rubric')` or `component.get('confidence')`. Indexing an absent key raises `KeyError` |
| `RewardSourceSpec.allowed_producers`, `RewardScalarization.component_names`, `RewardRubric.criteria`, `RewardArtifact.reward_components` | `model_dump()` returns lists. 0.22.0 returned tuples. Attribute access still returns tuples, and `model_dump(mode='json')` returned lists in 0.22.0 already | Compare dumped values with lists, or wrap them in `tuple(...)` before you compare them with tuples |
| `RuntimeDescriptor.metadata`, `TrialRecord.artifacts`, `TrialRecord.metadata`, `RewardArtifact.diagnostics`, `RewardArtifact.provenance` | Typed `dict[str, JsonValue]`. Their contents are checked at every depth | Convert values as listed for `JsonValue` |
| `StockScoredReward`, `StockRefusedReward` | `RewardArtifact.model_validate` selects them by the stock contract, `validation.pymc_elpd` and the disposition `scored` or `refused` | |
| `StockScoredReward`, `StockRefusedReward`, `PyMCScoredReward`, `PyMCRefusedReward`, `CompilationRefusedReward`, `PolicyRefusedReward` | Each row requires one reward component. Scored rewards follow metrics; refused rows require `reward=0.0` and `training_log_reward=-20.0`. | After editing a scored row, recompute its rewards and keep the component and scalarization values consistent with its metrics |
| `[build-system].requires` | `setuptools==84.2.1` and `setuptools-scm==10.3.4`, replacing `setuptools>=69` and `wheel` | |
| `expected_contract` | If supplied and either contract is `synthstats.harbor.stock_admission.v1`, the declared and expected contracts must match | If you supply `expected_contract`, match the declared contract |
| `get_capability_validation_requirements` | Takes a `TrainingReceiptCapability` and raises `ValueError` for an unrecognized capability string. 0.22.0 returned `{}` | Call it only with values from `typing.get_args(TrainingReceiptCapability)` |
| `[tool.uv].required-version` | The uv version requirement is removed. 0.22.0 required exactly 0.13.6 | Use a uv version that supports the lock file |
| `[tool.towncrier]` | Configures the changelog template, fragment directory and release-note categories | Use Towncrier to render release notes from fragments |
| `synthstats-schemas` | Version derives from Git tags via setuptools-scm. Source trees without Git metadata use `0.0.0+unknown`. | |
| `validation.compilation` | Allowed producers include `ppl_synth_data.harbor.pymc_verifier`, beside `ppl_synth_data.validation.compilation` and `ppl_synth_data.harbor.validator_template`. A `RewardComponent` with `source='validation.compilation'` and a `RewardArtifact` with `reward_source='validation.compilation'` accept it as `producer`. 0.22.0 raised `ValidationError` | |

### Fixed

| Name | Change | Action |
| --- | --- | --- |
| `IntegerDomain.accepts`, `RealDomain.accepts`, `NumericLevelsDomain.accepts` | Accept integers of any size within bounds or matching a `NumericLevelsDomain` level. 0.20.0 rejected integers larger than any float. | |

### Examples

#### `ATIFExportManifest`

  ```python
  # m is a valid two-entry manifest
  a, b = m['atif_ids']
  m['trial_artifacts'][a]['atif_id'], m['trial_artifacts'][b]['atif_id'] = b, a
  ATIFExportManifest.model_validate(m)
  # 0.22.0: returns a manifest
  # 0.22.1: ValidationError: 1 validation error for ATIFExportManifest
  #         Value error, trial_artifacts keys must match their embedded atif_id
  ```

#### `BundleCapability`

  ```python
  TypeAdapter(BundleCapability).validate_python('independent_review')
  # 0.22.0: ValidationError
  # 0.22.1: returns 'independent_review'
  ```

#### `BundleTrainingReceiptEntry.capabilities`

  ```python
  BundleTrainingReceiptEntry(name='x', path='x', sha256='a' * 64,
                             capabilities=['release_grade_mixed_atif'])
  # 0.22.0: validates
  # 0.22.1: ValidationError: 1 validation error for BundleTrainingReceiptEntry
  #         at capabilities.0
  ```

#### `ContractModel`

  ```python
  TrialRecord(trial_id='t', reward=float('nan'))
  # 0.22.0: validates
  # 0.22.1: ValidationError: 1 validation error for TrialRecord
  #         at reward, error type finite_number (float('inf') raises the same error)
  ```

#### `JsonValue`

  ```python
  RuntimeDescriptor(name='test', metadata={'value': (1, 2)})
  # 0.22.0: validates
  # 0.22.1: ValidationError: 1 validation error for RuntimeDescriptor
  #         input was not a valid JSON value
  ```

#### `PostTrainingAcceptanceBundle.required_capabilities`

  ```python
  TypeAdapter(PostTrainingAcceptanceBundle.model_fields['required_capabilities']
              .rebuild_annotation()).validate_python([])
  # 0.22.0: returns []
  # 0.22.1: ValidationError, error type too_short
  ```

#### `RewardArtifact.components`

  ```python
  # row is a valid artifact dict with reward_components and no admission contract
  RewardArtifact.model_validate({**row, 'components': {}})
  # 0.22.0: returns an artifact with components derived from reward_components
  # 0.22.1: ValidationError: 1 validation error for RewardArtifact
  #         Value error, components must exactly match reward_components names
  ```

#### `RewardArtifact.model_validate` with a declared contract

  ```python
  # row is a valid stock row
  del row['diagnostics']['disposition']
  RewardArtifact.model_validate(row)
  # 0.22.0: returns an artifact
  # 0.22.1: ValidationError: 1 validation error for RewardArtifact
  #         Value error, declared verifier contract requires diagnostics.disposition
  ```

#### `RewardArtifact.model_validate` with a strict row

  ```python
  # row is a valid scored stock row with its admission contract declared.
  row['reward'] = -123
  RewardArtifact.model_validate(row)
  # 0.22.0: validates
  # 0.22.1: ValidationError: scored reward does not follow the declared transform
  ```

#### `RewardArtifact.reward_components`

  ```python
  # row is a valid validation.pymc_elpd artifact dict.
  row['reward_components'][0]['diagnostics'] = {**row['diagnostics'], 'extra': True}
  RewardArtifact.model_validate(row)
  # 0.22.0: returns an artifact
  # 0.22.1: ValidationError: 1 validation error for RewardArtifact
  #         Value error, PyMC component diagnostics must match artifact diagnostics
  ```

#### `RewardArtifact.rubric`

  ```python
  # row is a valid artifact dict without a rubric
  'rubric' in RewardArtifact.model_validate(row).model_dump()
  # 0.22.0: True
  # 0.22.1: False
  ```

#### `RewardSourceSpec.allowed_producers`

  ```python
  type(get_reward_source('validation.compilation').model_dump()['allowed_producers']).__name__
  # 0.22.0: 'tuple'
  # 0.22.1: 'list'
  ```

#### `ScoreMetrics`

  ```python
  from pydantic import ValidationError
  from synthstats_schemas.numeric_reward import ScoreMetrics

  try:
      ScoreMetrics(avg_holdout_logp=-0.7, holdout_elpd=-7.0, train_elpd_loo=None,
                   pareto_k_max=None, baseline_score=-9.0, gold_score=None,
                   temperature=1.0, passed_layers=7, total_layers=6)
  except ValidationError as error:
      print(error.errors()[0]["msg"])
  # Value error, passed_layers cannot exceed total_layers
  ```

#### `attribution_for`

  ```python
  from synthstats_schemas.verdicts import attribution_for

  attribution_for("L2_model_build", None, "verifier_timeout", timeout_origin="runner_deadline")
  # <Attribution.INFRASTRUCTURE: 'infrastructure'>
  attribution_for("L2_model_build", None, "verifier_timeout")
  # <Attribution.UNCHECKED: 'UNCHECKED'>
  attribution_for("L0_parse", None, "L0_parse")
  # <Attribution.CANDIDATE: 'program_defect'>
  ```

#### `check_launch`

  ```python
  from synthstats_schemas.launch import LaunchRefused, check_launch

  try:
      check_launch({}, allowed_wandb_entities={"my-team"}, environ={},
                   netrc_path=None, backend="skyrl_0_4", wandb_probe=None)
  except LaunchRefused as refused:
      print(refused.problems[:2])
  # (('axes.model', 'missing'), ('axes.objective', 'missing'))
  ```

#### `expected_contract`

  ```python
  # row is a valid stock row
  RewardArtifact.model_validate(
      row, context={'expected_contract': 'synthstats.harbor.numeric_admission.v1'})
  # 0.22.0: returns an artifact
  # 0.22.1: ValidationError: 1 validation error for RewardArtifact
  #         Value error, the declared admission contract is not the expected one
  ```

#### `get_capability_validation_requirements`

  ```python
  get_capability_validation_requirements('release_grade_mixed_atif')
  # 0.22.0: {}
  # 0.22.1: ValueError. The first line of the message starts
  #         "no validation requirements are declared for capability 'release_grade_mixed_atif'"
  ```

#### `recorded_stop_reason`

  ```python
  from synthstats_schemas.stop_reasons import CUT_STOP_REASONS, recorded_stop_reason

  recorded_stop_reason(None)                          # 'unknown'
  recorded_stop_reason("length")                      # 'length'
  recorded_stop_reason("length") in CUT_STOP_REASONS  # True
  recorded_stop_reason("stop") in CUT_STOP_REASONS    # False
  ```



## 0.22.0 (2026-11-07)

### Added

| Name | Change | Action |
| --- | --- | --- |
| `validation.qualified_checks` | Registered reward source. Its value is the fraction of a task's qualified verifier checks that pass | |
| `canonical_json_bytes` | Package-root export. `canonical_json_bytes(value)` encodes `value` as UTF-8 JSON with compact separators, sorted keys, no ASCII escaping and no non-finite floats. Every digest the package computes over JSON content goes through it. A non-finite float raises `ValueError` | |
| `uv.lock`, `[tool.uv]` | Adds a dependency lock and an exact uv version requirement. 0.22.1 removes that requirement | Use a uv version that supports the lock file when reproducing the environment |

### Changed

| Name | Change | Action |
| --- | --- | --- |
| `ATIFTrialArtifact.data_ref`, `MixedATIFTrialArtifact.data_ref` | Take a canonical POSIX-relative path such as `data/run/train.csv`. Validation rejects a NUL byte, a backslash, an empty segment, a `.` segment, a `..` segment, a leading `/` and a trailing `/`. 0.20.0 rejected only `..` segments and a leading `/` | Write each `data_ref` in canonical form before you validate it |
| `atif_id`, `source_run_id` | Safe identifiers reject a backslash; 0.20.0 accepted it on POSIX systems | Use identifiers without backslashes |
| `BundleEvidenceReceiptEntry` | A Pydantic `TypeAdapter` over a union of evidence classes, selected by each entry's `kind`. `validate_python(entry)` and `validate_json(text)` return the class that `kind` selects, which is `FreshProviderGenerationEvidence` for `"fresh_provider_generation"`. `model_validate` and `model_validate_json` no longer exist and raise `AttributeError` | Rename each such call to `validate_python` or `validate_json` |
| `QualifiedImageReceiptV1.image_manifest_sha256`, `QualifiedImageReceiptV1.image_config_sha256` | An optional pair. A receipt with only `image_manifest_sha256` raises `ValidationError` (`Value error, image manifest and config SHA-256 must both be supplied and non-null`) | Supply both or neither |
| `QualifiedImageReceiptV1.runtime_wheel_sha256`, `QualifiedImageReceiptV1.runtime_wheel_package_version` | An optional pair. A receipt with only `runtime_wheel_sha256` raises `ValidationError` (`Value error, runtime wheel SHA-256 and package version must both be supplied and non-null`) | Supply both or neither |
| `QualifiedImageReceiptV1.image_digest` | When image identity fields are supplied, it must equal `"sha256:" + image_config_sha256` | Set the digest from the image configuration hash |
| `QualifiedImageReceiptV1.runtime_wheel_package_version` | When wheel identity fields are supplied, it must equal `runtime_packages['synthstats-schemas']` | Record the same package version in both fields |
| `QualifiedImageReceiptV1.runtime_registry_sha256` | Removed. A receipt that carries it raises `ValidationError` (`Extra inputs are not permitted`) | Delete the field from every receipt you write |
| `heuristic.llm_rubric` producer | Only `ppl_synth_data.reward.scalarization` is accepted; `ppl_synth_data.validation.llm_rubric` and `ppl_synth_data.eval.rubrics.llm` raise `ValidationError` | Use the accepted producer for this reward source |
| `heuristic.judge_exception` producer | Only `ppl_synth_data.harbor.validator_template` is accepted; `ppl_synth_data.eval.orchestrator` raises `ValidationError` | Use the accepted producer for this reward source |
| `synthstats_schemas.acceptance_bundle` | Defines the acceptance bundle contracts, among them `PostTrainingAcceptanceBundle`, `BundleFile`, `BundleTrainingReceiptEntry` and `BundleEvidenceReceiptEntry`. `synthstats_schemas.receipt` no longer defines them, so `from synthstats_schemas.receipt import PostTrainingAcceptanceBundle` raises `ImportError`. The package root exports the same names from this group as in 0.20.0 | Import them from `synthstats_schemas.acceptance_bundle`, or from the package root |
| `TaskContract` | The model for `synthstats.task_contract.v2` payloads. It no longer has `allowed_imports` or `scorer_rules`, and requires `required_predictor_groups`, `relationship` and `row_semantics`, which have no defaults. It forbids extra fields. `TaskContractV1` is unchanged | Delete `allowed_imports` and, where present, `scorer_rules` from each version 2 payload, and add the three new fields |

### Removed

| Name | Change | Action |
| --- | --- | --- |
| `heuristic.no_submission`, `validation.expected_information_gain` | No longer registered reward sources. `get_reward_source` raises `RewardSourceNotRegisteredError` for either name. A `RewardArtifact` whose `reward_source` is one of them raises `ValidationError` | |
| `ScorerRules` | Removed from `synthstats_schemas.task_contract` and from the package root. `from synthstats_schemas import ScorerRules` raises `ImportError` | |
| `synthstats_schemas.atif_rl_sidecar` | Removed with its package-root exports `ATIFRLCall`, `ATIFRLSidecar` and `ATIFServingIdentity`. `import synthstats_schemas.atif_rl_sidecar` raises `ModuleNotFoundError` | |
| `synthstats_schemas.predictive` | Removed with its package-root exports `F64Bits`, `NormalJointDrawV1`, `NormalJointSampleBlockV1`, `NormalPredictiveRunSpecV1`, `NormalPredictiveWorkerAttemptV1`, `NormalPredictiveWorkerFailureV1`, `NormalPredictiveWorkerSuccessV1`, `PredictiveWireError`, `decode_f64_bits`, `decode_normal_joint_sample_block`, `decode_normal_predictive_run_spec`, `decode_normal_predictive_worker_attempt`, `encode_f64_bits`, `encode_normal_joint_sample_block`, `encode_normal_predictive_run_spec`, `encode_normal_predictive_worker_attempt`, `validate_normal_joint_sample_blocks` and `validate_normal_predictive_worker_attempts`. `import synthstats_schemas.predictive` raises `ModuleNotFoundError` | |
| `wandb` extra | Removed. It installed `wandb>=0.30`, which `synthstats_schemas.wandb_privacy.private_project_probe` imports when you call it without `execute` | Install `wandb>=0.30` yourself if you call that function |

### Examples

#### `canonical_json_bytes`

```python
canonical_json_bytes({"b": 1, "a": "é"})
# b'{"a":"\xc3\xa9","b":1}'
canonical_json_bytes({"x": float("nan")})
# ValueError
```

#### `ATIFTrialArtifact.data_ref`

```python
# An otherwise valid ATIFTrialArtifact
data_ref="data//x.csv"
# ValidationError: Value error, data_ref must be a canonical POSIX-relative path
# "../out.csv", "data/./x.csv", "\data\x.csv" and "data/run/" raise the same error
```

#### `BundleEvidenceReceiptEntry`

```python
BundleEvidenceReceiptEntry.validate_python(entry)  # returns the class that entry["kind"] selects
BundleEvidenceReceiptEntry.model_validate(entry)   # AttributeError
```

#### `TaskContract`

```python
parse_task_contract(json.dumps(payload))
# payload is a version 2 payload written for 0.20.0
# ValidationError: Field required, for each of required_predictor_groups, relationship and row_semantics
#                  Extra inputs are not permitted, for allowed_imports
```

## 0.20.0 (2026-11-03)

- Task contracts validate data columns and interaction requirements, with canonical JSON and SHA-256 identities.
- Schemas cover trajectories, reward artifacts, training receipts, release bundles and predictive samples.
- `independent_review` evidence requires review metadata, a `GO` verdict and zero unresolved P0/P1 findings.
- `oracle_review` is a supported evidence kind.

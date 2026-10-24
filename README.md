# synthstats-schemas

This package defines Pydantic models for SynthStats tasks, traces and rewards.

## Install

Run these commands from the repository root.

```sh
uv sync
```

## Example

```sh
uv run python - <<'PY'
from synthstats_schemas import TaskManifest

task = TaskManifest(task_id="coin-toss", name="Coin toss", version="1")
print(task.model_dump_json())
PY
```

## Modules

| Module | Contents |
| --- | --- |
| `synthstats_schemas.task` | Task manifests |
| `synthstats_schemas.trace` | Trace records |
| `synthstats_schemas.reward` | Reward records |

[SynthStats](https://synthstats.org)

The [MIT Licence](LICENSE) applies.

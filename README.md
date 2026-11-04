<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/mark.svg">
    <img src="assets/mark.svg" alt="SynthStats: synthstats-schemas" width="140">
  </picture>
</p>

<p align="center">
  <a href="https://synthstats.org"><img src="https://img.shields.io/badge/website-synthstats.org-5C56A5?labelColor=5A6078" alt="Website: synthstats.org"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-MIT-5C56A5?labelColor=5A6078" alt="Licence: MIT"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%E2%89%A53.11-5C56A5?labelColor=5A6078" alt="Python: ≥3.11"></a>
</p>

# synthstats-schemas

Pydantic schemas for SynthStats tasks, trajectories, rewards and receipts.

## Install

```bash
uv venv --python 3.12
uv pip install .
```

Run these commands from this repository's checkout.

## Example

```bash
.venv/bin/python - <<'PY'
from synthstats_schemas import canonical_json, parse_task_contract

contract = parse_task_contract('''{
  "schema_version": "synthstats.task_contract.v2",
  "language": "pymc",
  "entry_point": {"function": "model", "parameter": "data", "returns": "pymc.Model"},
  "data": {
    "container": "dict", "column_type": "numpy.ndarray",
    "row_count_key": "n", "keys": ["x", "y"], "target_key": "y"
  },
  "provided_names": {"pm": "pymc", "np": "numpy"},
  "allowed_imports": [],
  "conventions": [],
  "environment": {
    "image_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "python": "3.12", "packages": {}
  },
  "interaction": {"mode": "pool", "query_budget": 3, "observation_key": "y"}
}''')
encoded = canonical_json(contract)
assert parse_task_contract(encoded) == contract
print(contract.interaction_mode)  # pool
PY
```

## Optional dependencies

The read-only W&B project probe needs `uv pip install '.[wandb]'`.
Tests need `uv pip install '.[test]'`, then `.venv/bin/pytest`.

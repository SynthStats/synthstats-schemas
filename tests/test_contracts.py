import json

import pytest
from pydantic import ValidationError

from synthstats_schemas import (
    BundleEvidenceReceiptEntry,
    canonical_json,
    parse_task_contract,
)


def contract_payload(version="synthstats.task_contract.v2"):
    payload = {
        "schema_version": version,
        "language": "pymc",
        "entry_point": {"function": "model", "parameter": "data", "returns": "pymc.Model"},
        "data": {
            "container": "dict", "column_type": "numpy.ndarray",
            "row_count_key": "n", "keys": ["x", "y"], "target_key": "y",
        },
        "provided_names": {"pm": "pymc", "np": "numpy"},
        "allowed_imports": [],
        "conventions": [],
        "environment": {"image_digest": "sha256:" + "0" * 64, "python": "3.12", "packages": {}},
    }
    if version == "synthstats.task_contract.v2":
        payload["interaction"] = {"mode": "pool", "query_budget": 3, "observation_key": "y"}
    return payload


@pytest.mark.parametrize("version", ["synthstats.task_contract.v1", "synthstats.task_contract.v2"])
def test_contract_json_round_trip_preserves_declared_version(version):
    contract = parse_task_contract(json.dumps(contract_payload(version)))
    assert contract.schema_version == version
    assert contract.interaction_mode == "pool"
    encoded = canonical_json(contract)
    assert json.loads(encoded)["schema_version"] == version
    assert parse_task_contract(encoded) == contract


def test_contract_rejects_observation_column_outside_declared_target():
    payload = contract_payload()
    payload["interaction"]["observation_key"] = "x"
    with pytest.raises(ValidationError, match="observation_key must equal data target_key"):
        parse_task_contract(json.dumps(payload))


def review_payload():
    return {
        "name": "review", "kind": "independent_review", "path": "review.json",
        "sha256": "0" * 64, "model_id": "example-review-model", "route": "example-provider",
        "output_sha256": "1" * 64, "heuristic_only": True, "classification": "review",
        "review_verdict": "GO", "unresolved_p0_count": 0, "unresolved_p1_count": 0,
    }


def test_complete_independent_review_is_accepted():
    receipt = BundleEvidenceReceiptEntry.model_validate(review_payload())
    assert receipt.kind == "independent_review"
    assert receipt.review_verdict == "GO"


@pytest.mark.parametrize("field", [
    "model_id", "route", "output_sha256", "heuristic_only", "classification",
    "review_verdict", "unresolved_p0_count", "unresolved_p1_count",
])
def test_independent_review_rejects_missing_required_metadata(field):
    payload = review_payload()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        BundleEvidenceReceiptEntry.model_validate(payload)


@pytest.mark.parametrize("field,value", [
    ("review_verdict", "NO-GO"), ("unresolved_p0_count", 1), ("unresolved_p1_count", 1),
])
def test_independent_review_rejects_unresolved_findings_or_rejection(field, value):
    payload = review_payload()
    payload[field] = value
    with pytest.raises(ValidationError, match="independent_review evidence requires"):
        BundleEvidenceReceiptEntry.model_validate(payload)

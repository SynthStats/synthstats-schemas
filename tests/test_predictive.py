import hashlib
import json

import pytest

from synthstats_schemas import (
    NormalPredictiveRunSpecV1,
    PredictiveWireError,
    decode_normal_joint_sample_block,
    encode_normal_predictive_run_spec,
)


def run_spec():
    return NormalPredictiveRunSpecV1(
        candidate_sha256="0" * 64, draws_per_block=1, independent_block_count=2,
        public_task_sha256="1" * 64, query_count=1, query_set_sha256="2" * 64,
        schema_version="synthstats.predictive.normal-run.v1", visible_data_sha256="3" * 64,
    )


def block_payload(spec):
    return {
        "block_index": 0,
        "draws": [{"draw_index": 0, "targets_by_query": ["3ff0000000000000"]}],
        "run_spec_sha256": hashlib.sha256(encode_normal_predictive_run_spec(spec)).hexdigest(),
        "schema_version": "synthstats.predictive.normal-joint-block.v1",
    }


def wire_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def test_canonical_predictive_block_preserves_sample():
    spec = run_spec()
    block = decode_normal_joint_sample_block(wire_bytes(block_payload(spec)), run_spec=spec)
    assert block.block_index == 0
    assert block.draws[0].targets_by_query == ("3ff0000000000000",)


@pytest.mark.parametrize("change,label", [
    ("extra", "wire.extra_field:extra"),
    ("negative_zero", "wire.negative_zero:block.draws[0].targets_by_query[0]"),
    ("count", "wire.count_mismatch:targets_by_query"),
    ("context", "wire.context_mismatch:run_spec_sha256"),
])
def test_predictive_block_rejects_invalid_wire_data(change, label):
    spec = run_spec()
    payload = block_payload(spec)
    if change == "extra":
        payload["extra"] = 1
    elif change == "negative_zero":
        payload["draws"][0]["targets_by_query"] = ["8000000000000000"]
    elif change == "count":
        payload["draws"][0]["targets_by_query"] = []
    else:
        payload["run_spec_sha256"] = "4" * 64
    with pytest.raises(PredictiveWireError) as exc:
        decode_normal_joint_sample_block(wire_bytes(payload), run_spec=spec)
    assert exc.value.label == label

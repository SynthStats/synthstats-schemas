"""Check reward compatibility with the recorded ppl-synth-data source tree."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from synthstats_schemas import REWARD_SOURCE_CATALOG, RewardArtifact, get_reward_source


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures" / "ppl_reward_contract.json").read_text()
)


@pytest.mark.parametrize("source", CONTRACT["referenced_sources"])
def test_ppl_reward_source_is_accepted(source):
    spec = get_reward_source(source)
    owner = "verifier" if source.startswith("validation.") else "heuristic"
    artifact = RewardArtifact.model_validate(
        {
            "reward_source": source,
            "evidence_kind": owner,
            "truth_owner": owner,
            "producer": spec.allowed_producers[0],
            "training_log_reward": -1.0,
            "reward_transform": {"kind": "manual"},
        }
    )
    assert artifact.reward_source == source


def test_every_catalogue_producer_is_a_ppl_source_module():
    modules = {**CONTRACT["src_modules"], **CONTRACT["script_producers"]}
    absent = {
        source: [producer for producer in spec.allowed_producers if producer not in modules]
        for source, spec in REWARD_SOURCE_CATALOG.items()
        if any(producer not in modules for producer in spec.allowed_producers)
    }
    assert not absent, f"Catalogue producers absent from ppl-synth-data: {absent}"


def test_contract_export_reads_sources_and_module_paths(tmp_path):
    tree = tmp_path / "ppl"
    module = tree / "src" / "ppl_synth_data" / "scores.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        'SOURCE = "validation.example"\n'
        'TEXT = "validation.example is a source"\n'
    )
    scorer = tree / "docker" / "runtime" / "score.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text(
        '_PRODUCER = "ppl_synth_data.docker.runtime.score"\n'
        'payload = {"reward_source": "heuristic.example"}\n'
    )
    (tree / "scripts").mkdir()
    output = tmp_path / "contract.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_ppl_reward_contract.py"),
            str(tree),
            "--output",
            str(output),
        ],
        check=True,
    )

    contract = json.loads(output.read_text())
    assert contract["referenced_sources"] == ["heuristic.example", "validation.example"]
    assert contract["src_modules"] == {
        "ppl_synth_data.scores": "src/ppl_synth_data/scores.py"
    }
    assert contract["source_locations"]["heuristic.example"] == [
        "docker/runtime/score.py:2"
    ]
    assert contract["script_producers"] == {
        "ppl_synth_data.docker.runtime.score": "docker/runtime/score.py"
    }

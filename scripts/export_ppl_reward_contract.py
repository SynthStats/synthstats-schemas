"""Snapshot literal reward-source references and producer paths from ppl-synth-data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    locations: dict[str, set[str]] = {}
    modules: dict[str, str] = {}
    script_producers: dict[str, str] = {}
    hashes: dict[str, str] = {}

    for folder in ("src", "scripts", "docker"):
        directory = args.tree / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(args.tree)
            if len(relative.parts) > 12:
                raise ValueError(f"Python source exceeds the traversal depth: {relative}")
            raw = path.read_bytes()
            hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
            if folder == "src":
                parts = list(relative.with_suffix("").parts[1:])
                if parts[-1] == "__init__":
                    parts.pop()
                modules[".".join(parts)] = str(relative)
            tree = ast.parse(raw, filename=str(relative))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if value.startswith(("validation.", "heuristic.")) and all(
                        part.isidentifier() for part in value.split(".")
                    ):
                        locations.setdefault(value, set()).add(f"{relative}:{node.lineno}")
                if (
                    folder == "docker"
                    and isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_PRODUCER" for t in node.targets)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    script_producers[node.value.value] = str(relative)

    contract = {
        "referenced_sources": sorted(locations),
        "source_locations": {key: sorted(value) for key, value in sorted(locations.items())},
        "src_modules": dict(sorted(modules.items())),
        "script_producers": dict(sorted(script_producers.items())),
        "python_files_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    }
    args.output.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

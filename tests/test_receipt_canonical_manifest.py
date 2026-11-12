"""The training receipt's manifest digest follows the package's one canonical JSON rule.

The package canonicalises JSON in exactly one place: compact separators, sorted keys,
UTF-8 and no ASCII escaping. A manifest whose path carries a non-ASCII character is
where a second, ASCII-escaping copy of that rule shows itself: the same record then has
two different digests, and a receipt written against one is refused by the other.

The expected digest below is a pinned literal, computed once from the canonical rule and
recorded here. It is not recomputed from the code under test, so this file is a
verifier: it fails on code that hashes the manifest a second way.
"""

import hashlib
import json

from synthstats_schemas.receipt import TrainingReceiptArtifacts

_HASH = "0" * 64
_NON_ASCII_PATH = "données/poids.safetensors"
# The digest of the manifest below through the production canonical path, computed once
# and pinned: a change to the canonical encoding (key order, separators, unicode
# handling) changes this digest and fails here.
_CANONICAL_DIGEST = "a866f94c5458433de5dc37dacbf2793c1c230e54121459da29ef7cc29c5e8961"


def _manifest() -> dict[str, dict[str, object]]:
    return {_NON_ASCII_PATH: {"sha256": _HASH, "size_bytes": 256}}


def test_artifacts_accept_a_non_ascii_manifest_path_under_the_canonical_rule():
    artifacts = TrainingReceiptArtifacts(
        checkpoints=[_NON_ASCII_PATH],
        final_checkpoint=_NON_ASCII_PATH,
        final_manifest=_manifest(),
        final_manifest_sha256=_CANONICAL_DIGEST,
    )
    assert artifacts.final_manifest_sha256 == _CANONICAL_DIGEST


def test_ascii_escaping_would_hash_the_same_manifest_differently():
    """The two rules disagree, so a receipt written against the wrong one is refused."""
    escaped = json.dumps(
        _manifest(),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert hashlib.sha256(escaped).hexdigest() != _CANONICAL_DIGEST

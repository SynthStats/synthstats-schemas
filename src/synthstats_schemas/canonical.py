"""The package's one canonical JSON encoding.

Every digest the package computes over JSON content goes through
:func:`canonical_json_bytes`, so a record has exactly one canonical encoding
and therefore one digest. A second copy of this rule differs from this one by
one flag, and a receipt written against the other copy is refused here.
"""

from __future__ import annotations

import json


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a value as the package's one canonical JSON form.

    Compact separators, sorted keys, UTF-8, no ASCII escaping and no
    non-finite floats.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

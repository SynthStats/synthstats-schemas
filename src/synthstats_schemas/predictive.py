"""Strict Normal predictive wire contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Annotated, Final, Literal, TypeAlias, TypedDict

from pydantic import (
    AfterValidator,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from synthstats_schemas.base import ContractModel

_MAX_DOCUMENT_BYTES: Final = 67_108_864
_FIXED_ENVELOPE_BYTES: Final = 4_096
_UINT16_MAX: Final = 65_535
_UINT32_MAX: Final = 4_294_967_295
_U128_MAX: Final = (1 << 128) - 1

_RUN_SCHEMA_VERSION: Final = "synthstats.predictive.normal-run.v1"
_BLOCK_SCHEMA_VERSION: Final = "synthstats.predictive.normal-joint-block.v1"
_ATTEMPT_SCHEMA_VERSION: Final = "synthstats.predictive.normal-worker-attempt.v1"

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_F64_BITS_PATTERN: Final = re.compile(r"^[0-9a-f]{16}$")


def _validate_f64_bits_type(value: str) -> str:
    bits = int(value, 16)
    if (bits >> 52) & 0x7FF == 0x7FF:
        raise ValueError("F64Bits must encode a finite binary64 value")
    if bits == 0x8000000000000000:
        raise ValueError("F64Bits must not encode negative zero")
    return value


Sha256: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
F64Bits: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{16}$"),
    AfterValidator(_validate_f64_bits_type),
]
_UInt16: TypeAlias = Annotated[int, Field(strict=True, ge=0, le=_UINT16_MAX)]
_UInt32: TypeAlias = Annotated[int, Field(strict=True, ge=0, le=_UINT32_MAX)]
_PositiveUInt32: TypeAlias = Annotated[
    int,
    Field(strict=True, ge=1, le=_UINT32_MAX),
]
_BlockCount: TypeAlias = Annotated[
    int,
    Field(strict=True, ge=2, le=_UINT16_MAX),
]
_FailureDisposition: TypeAlias = Literal[
    "candidate_policy_rejected",
    "candidate_build_failed",
    "candidate_execution_failed",
    "candidate_resource_exhausted",
    "payload_rejected",
    "infrastructure_failed",
]

_FAILURE_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "candidate_policy_rejected",
        "candidate_build_failed",
        "candidate_execution_failed",
        "candidate_resource_exhausted",
        "payload_rejected",
        "infrastructure_failed",
    }
)


class PredictiveWireError(ValueError):
    """A stable rejection from the Normal predictive wire."""

    def __init__(self, label: str) -> None:
        self.label = label
        super().__init__(label)


def _wire_error(label: str) -> PredictiveWireError:
    return PredictiveWireError(label)


def _checked_add(left: int, right: int) -> int:
    result = left + right
    if result > _U128_MAX:
        raise _wire_error("wire.derived_size_overflow")
    return result


def _checked_mul(left: int, right: int) -> int:
    result = left * right
    if result > _U128_MAX:
        raise _wire_error("wire.derived_size_overflow")
    return result


def _decimal_digits(value: int) -> int:
    return len(str(value))


def _draw_index_digit_sum(draw_count: int) -> int:
    total = 1
    lower = 1
    for digits in range(1, 11):
        upper = 10**digits
        term_count = max(0, min(draw_count, upper) - lower)
        total = _checked_add(total, _checked_mul(digits, term_count))
        lower = upper
    return total


def _exact_block_document_length(
    *,
    draws_per_block: int,
    query_count: int,
    block_index: int,
) -> int:
    target_bytes = _checked_mul(19, query_count)
    draw_bytes = _checked_add(37, target_bytes)
    all_draw_bytes = _checked_mul(draws_per_block, draw_bytes)
    return _checked_add(
        _checked_add(
            _checked_add(174, _decimal_digits(block_index)),
            all_draw_bytes,
        ),
        _draw_index_digit_sum(draws_per_block),
    )


def _derived_document_lengths(
    *,
    draws_per_block: int,
    independent_block_count: int,
    query_count: int,
) -> tuple[int, int]:
    target_bytes = _checked_mul(19, query_count)
    block_length = _exact_block_document_length(
        draws_per_block=draws_per_block,
        query_count=query_count,
        block_index=independent_block_count - 1,
    )
    hidden_length = _checked_add(
        _checked_add(270, _decimal_digits(query_count)),
        target_bytes,
    )
    return block_length, hidden_length


def _require_derived_documents_fit(
    *,
    draws_per_block: int,
    independent_block_count: int,
    query_count: int,
) -> tuple[int, int]:
    block_length, hidden_length = _derived_document_lengths(
        draws_per_block=draws_per_block,
        independent_block_count=independent_block_count,
        query_count=query_count,
    )
    if block_length > _MAX_DOCUMENT_BYTES or hidden_length > _MAX_DOCUMENT_BYTES:
        raise _wire_error("wire.byte_limit_exceeded")
    return block_length, hidden_length


def _validate_f64_bits(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _F64_BITS_PATTERN.fullmatch(value) is None:
        raise _wire_error(f"wire.f64_format:{path}")
    bits = int(value, 16)
    if (bits >> 52) & 0x7FF == 0x7FF:
        raise _wire_error(f"wire.non_finite:{path}")
    if bits == 0x8000000000000000:
        raise _wire_error(f"wire.negative_zero:{path}")
    return value


class NormalPredictiveRunSpecV1(ContractModel):
    candidate_sha256: Sha256
    draws_per_block: _PositiveUInt32
    independent_block_count: _BlockCount
    public_task_sha256: Sha256
    query_count: _PositiveUInt32
    query_set_sha256: Sha256
    schema_version: Literal["synthstats.predictive.normal-run.v1"]
    visible_data_sha256: Sha256

    @model_validator(mode="after")
    def derived_documents_fit(self) -> NormalPredictiveRunSpecV1:
        _require_derived_documents_fit(
            draws_per_block=self.draws_per_block,
            independent_block_count=self.independent_block_count,
            query_count=self.query_count,
        )
        return self


class NormalJointDrawV1(ContractModel):
    draw_index: _UInt32
    targets_by_query: tuple[F64Bits, ...]

    @field_validator("targets_by_query")
    @classmethod
    def targets_are_finite_binary64(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for index, target in enumerate(value):
            _validate_f64_bits(target, path=f"targets_by_query[{index}]")
        return value


class NormalJointSampleBlockV1(ContractModel):
    block_index: _UInt16
    draws: tuple[NormalJointDrawV1, ...]
    run_spec_sha256: Sha256
    schema_version: Literal["synthstats.predictive.normal-joint-block.v1"]

    @model_validator(mode="after")
    def draw_indexes_match_positions(self) -> NormalJointSampleBlockV1:
        seen: set[int] = set()
        for expected, draw in enumerate(self.draws):
            if draw.draw_index in seen:
                raise ValueError(f"duplicate draw_index {draw.draw_index}")
            seen.add(draw.draw_index)
            if draw.draw_index != expected:
                raise ValueError(
                    f"draw_index at position {expected} must equal {expected}"
                )
        return self


class NormalPredictiveWorkerSuccessV1(ContractModel):
    block_index: _UInt16
    kind: Literal["success"]
    run_spec_sha256: Sha256
    sample_block_sha256: Sha256
    schema_version: Literal["synthstats.predictive.normal-worker-attempt.v1"]


class NormalPredictiveWorkerFailureV1(ContractModel):
    block_index: _UInt16
    disposition: _FailureDisposition
    kind: Literal["failure"]
    run_spec_sha256: Sha256
    schema_version: Literal["synthstats.predictive.normal-worker-attempt.v1"]


NormalPredictiveWorkerAttemptV1: TypeAlias = (
    NormalPredictiveWorkerSuccessV1 | NormalPredictiveWorkerFailureV1
)


@dataclass(frozen=True, slots=True)
class _JsonSpan:
    start: int
    end: int


class _MaterializedDraw(TypedDict):
    draw_index: int
    targets_by_query: list[str]


@dataclass(slots=True)
class _JsonContainerFrame:
    kind: Literal["object", "array"]
    state: str
    keys: set[str] | None = None


class _DuplicateKey(Exception):
    def __init__(self, name: str) -> None:
        self.name = name


class _InvalidJson(Exception):
    def __init__(
        self,
        position: int | None = None,
        *,
        recovery_position: int | None = None,
    ) -> None:
        self.position = position
        self.recovery_position = recovery_position


@dataclass(slots=True)
class _DuplicateKeyState:
    name: str | None = None
    sort_key: bytes | None = None

    def observe(self, name: str) -> None:
        sort_key = _utf16_sort_key(name)
        if self.sort_key is None or sort_key < self.sort_key:
            self.name = name
            self.sort_key = sort_key


def _json_whitespace_end(text: str, start: int) -> int:
    while start < len(text) and text[start] in " \t\r\n":
        start += 1
    return start


def _json_string_recovery_position(text: str, start: int) -> int:
    position = start + 1
    while position < len(text):
        if text[position] == '"':
            return position + 1
        if text[position] == "\\":
            position += 2
        else:
            position += 1
    return len(text)


def _scan_json_string(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != '"':
        raise _InvalidJson(start)
    position = start + 1
    while position < len(text):
        character = text[position]
        if character == '"':
            end = position + 1
            try:
                value = json.loads(text[start:end])
            except JSONDecodeError as exc:
                raise _InvalidJson(
                    start + exc.pos,
                    recovery_position=end,
                ) from exc
            if not isinstance(value, str):
                raise _InvalidJson(start, recovery_position=end)
            if any(0xD800 <= ord(member) <= 0xDFFF for member in value):
                raise _InvalidJson(position, recovery_position=end)
            return value, end
        if ord(character) < 0x20:
            raise _InvalidJson(
                position,
                recovery_position=_json_string_recovery_position(text, start),
            )
        if character != "\\":
            position += 1
            continue
        if position + 1 >= len(text):
            raise _InvalidJson(len(text), recovery_position=len(text))
        escape = text[position + 1]
        if escape == "u":
            end = position + 6
            if end > len(text) or any(
                digit not in "0123456789abcdefABCDEF"
                for digit in text[position + 2 : end]
            ):
                raise _InvalidJson(
                    min(end, len(text)),
                    recovery_position=_json_string_recovery_position(text, start),
                )
            position = end
            continue
        if escape not in '"\\/bfnrt':
            raise _InvalidJson(
                position + 1,
                recovery_position=_json_string_recovery_position(text, start),
            )
        position += 2
    raise _InvalidJson(len(text), recovery_position=len(text))


def _scan_json_number(text: str, start: int) -> int:
    position = start
    if text[position] == "-":
        position += 1
        if position == len(text):
            raise _InvalidJson(position)
    if text[position] == "0":
        position += 1
        if position < len(text) and "0" <= text[position] <= "9":
            raise _InvalidJson(position)
    elif "1" <= text[position] <= "9":
        position += 1
        while position < len(text) and "0" <= text[position] <= "9":
            position += 1
    else:
        raise _InvalidJson(position)
    if position < len(text) and text[position] == ".":
        position += 1
        if position == len(text) or not "0" <= text[position] <= "9":
            raise _InvalidJson(position)
        while position < len(text) and "0" <= text[position] <= "9":
            position += 1
    if position < len(text) and text[position] in "eE":
        position += 1
        if position < len(text) and text[position] in "+-":
            position += 1
        if position == len(text) or not "0" <= text[position] <= "9":
            raise _InvalidJson(position)
        while position < len(text) and "0" <= text[position] <= "9":
            position += 1
    return position


def _finish_json_value(stack: list[_JsonContainerFrame]) -> bool:
    if not stack:
        return True
    parent = stack[-1]
    if parent.state == "object_value":
        parent.state = "object_comma_or_end"
    elif parent.state == "array_value":
        parent.state = "array_comma_or_end"
    else:
        raise _InvalidJson
    return False


def _scan_json_value(
    text: str,
    start: int,
    *,
    duplicate_state: _DuplicateKeyState | None = None,
) -> int:
    position = _json_whitespace_end(text, start)
    stack: list[_JsonContainerFrame] = []
    root_started = False

    while True:
        needs_value = (not root_started and not stack) or (
            bool(stack) and stack[-1].state in {"object_value", "array_value"}
        )
        if needs_value:
            position = _json_whitespace_end(text, position)
            if position >= len(text):
                raise _InvalidJson(position)
            character = text[position]
            if character == "{":
                stack.append(
                    _JsonContainerFrame(
                        kind="object",
                        state="object_first_key",
                        keys=set(),
                    )
                )
                root_started = True
                position += 1
                continue
            if character == "[":
                stack.append(
                    _JsonContainerFrame(kind="array", state="array_first_value")
                )
                root_started = True
                position += 1
                continue
            if character == '"':
                _, position = _scan_json_string(text, position)
            elif character == "-" or "0" <= character <= "9":
                position = _scan_json_number(text, position)
            elif text.startswith("true", position):
                position += 4
            elif text.startswith("false", position):
                position += 5
            elif text.startswith("null", position):
                position += 4
            else:
                raise _InvalidJson(position)
            root_started = True
            if _finish_json_value(stack):
                return position
            continue

        frame = stack[-1]
        position = _json_whitespace_end(text, position)
        if position >= len(text):
            raise _InvalidJson(position)

        if frame.kind == "object":
            if frame.state in {"object_first_key", "object_key"}:
                if frame.state == "object_first_key" and text[position] == "}":
                    position += 1
                    stack.pop()
                    if _finish_json_value(stack):
                        return position
                    continue
                key, position = _scan_json_string(text, position)
                if frame.keys is None:
                    raise _InvalidJson(position)
                if key in frame.keys:
                    if duplicate_state is None:
                        raise _DuplicateKey(key)
                    duplicate_state.observe(key)
                else:
                    frame.keys.add(key)
                frame.state = "object_colon"
                continue
            if frame.state == "object_colon":
                if text[position] != ":":
                    raise _InvalidJson(position)
                frame.state = "object_value"
                position += 1
                continue
            if frame.state == "object_comma_or_end":
                if text[position] == ",":
                    frame.state = "object_key"
                    position += 1
                    continue
                if text[position] == "}":
                    position += 1
                    stack.pop()
                    if _finish_json_value(stack):
                        return position
                    continue
                raise _InvalidJson(position)
            raise _InvalidJson(position)

        if frame.state == "array_first_value":
            if text[position] == "]":
                position += 1
                stack.pop()
                if _finish_json_value(stack):
                    return position
                continue
            frame.state = "array_value"
            continue
        if frame.state == "array_comma_or_end":
            if text[position] == ",":
                frame.state = "array_value"
                position += 1
                continue
            if text[position] == "]":
                position += 1
                stack.pop()
                if _finish_json_value(stack):
                    return position
                continue
            raise _InvalidJson(position)
        raise _InvalidJson(position)


def _scan_payload_duplicate_key(text: str) -> str | None:
    duplicate_state = _DuplicateKeyState()
    position = 0
    while position < len(text):
        start = _json_whitespace_end(text, position)
        if start == len(text):
            break
        try:
            end = _scan_json_value(
                text,
                start,
                duplicate_state=duplicate_state,
            )
        except _InvalidJson as exc:
            failure = exc.position
            if exc.recovery_position is not None:
                position = exc.recovery_position
            else:
                position = start + 1 if failure is None or failure <= start else failure
            continue
        if end <= start:
            raise RuntimeError("JSON scanner did not advance")
        position = end
    return duplicate_state.name


def _scan_document(data: bytes, *, envelope: int) -> tuple[str, _JsonSpan]:
    if not isinstance(data, bytes):
        raise TypeError("predictive wire input must be bytes")
    if len(data) > envelope:
        raise _wire_error("wire.byte_limit_exceeded")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _wire_error("wire.malformed_utf8") from exc

    duplicate_name = _scan_payload_duplicate_key(text)
    if duplicate_name is not None:
        raise _wire_error(f"wire.duplicate_key:{duplicate_name}")

    start = _json_whitespace_end(text, 0)
    duplicate_state = _DuplicateKeyState()
    try:
        end = _scan_json_value(text, start, duplicate_state=duplicate_state)
    except _DuplicateKey as exc:
        raise _wire_error(f"wire.duplicate_key:{exc.name}") from exc
    except _InvalidJson as exc:
        if duplicate_state.name is not None:
            raise _wire_error(f"wire.duplicate_key:{duplicate_state.name}") from exc
        raise _wire_error("wire.invalid_json") from exc

    if duplicate_state.name is not None:
        raise _wire_error(f"wire.duplicate_key:{duplicate_state.name}")

    if end == len(text):
        return text, _JsonSpan(start=start, end=end)

    next_start = _json_whitespace_end(text, end)
    if next_start == len(text):
        raise _wire_error("wire.trailing_bytes")
    next_duplicate_state = _DuplicateKeyState()
    try:
        _scan_json_value(
            text,
            next_start,
            duplicate_state=next_duplicate_state,
        )
    except _DuplicateKey as exc:
        raise _wire_error(f"wire.duplicate_key:{exc.name}") from exc
    except _InvalidJson:
        if next_duplicate_state.name is not None:
            raise _wire_error(
                f"wire.duplicate_key:{next_duplicate_state.name}"
            ) from None
        raise _wire_error("wire.trailing_bytes") from None
    if next_duplicate_state.name is not None:
        raise _wire_error(f"wire.duplicate_key:{next_duplicate_state.name}")
    raise _wire_error("wire.multiple_documents")


def _json_span_kind(text: str, span: _JsonSpan) -> str:
    character = text[span.start]
    if character == "{":
        return "object"
    if character == "[":
        return "array"
    if character == '"':
        return "string"
    if character in "-0123456789":
        token = text[span.start : span.end]
        return "fraction" if any(marker in token for marker in ".eE") else "integer"
    if character in "tf":
        return "boolean"
    return "null"


def _utf16_sort_key(value: str) -> bytes:
    return value.encode("utf-16-be", errors="surrogatepass")


def _scan_object_semantic_members(
    text: str,
    span: _JsonSpan,
    *,
    expected_fields: tuple[str, ...],
) -> tuple[dict[str, _JsonSpan], str | None]:
    if _json_span_kind(text, span) != "object":
        raise _InvalidJson
    position = _json_whitespace_end(text, span.start + 1)
    if text[position] == "}":
        return {}, None
    expected = frozenset(expected_fields)
    known_members: dict[str, _JsonSpan] = {}
    extra_name: str | None = None
    extra_sort_key: bytes | None = None
    while True:
        name, position = _scan_json_string(text, position)
        position = _json_whitespace_end(text, position)
        if text[position] != ":":
            raise _InvalidJson
        value_start = _json_whitespace_end(text, position + 1)
        value_end = _scan_json_value(text, value_start)
        if name in expected:
            known_members[name] = _JsonSpan(start=value_start, end=value_end)
        else:
            sort_key = _utf16_sort_key(name)
            if extra_sort_key is None or sort_key < extra_sort_key:
                extra_name = name
                extra_sort_key = sort_key
        position = _json_whitespace_end(text, value_end)
        if text[position] == "}":
            return known_members, extra_name
        if text[position] != ",":
            raise _InvalidJson
        position = _json_whitespace_end(text, position + 1)


def _logical_shape_field_order(
    expected_fields: tuple[str, ...],
    extra_name: str | None,
) -> tuple[str, ...]:
    names = expected_fields if extra_name is None else (*expected_fields, extra_name)
    return tuple(sorted(names, key=_utf16_sort_key))


def _iter_array_items(text: str, span: _JsonSpan) -> Iterator[_JsonSpan]:
    if _json_span_kind(text, span) != "array":
        raise _InvalidJson
    position = _json_whitespace_end(text, span.start + 1)
    if text[position] == "]":
        return
    while True:
        item_end = _scan_json_value(text, position)
        yield _JsonSpan(start=position, end=item_end)
        position = _json_whitespace_end(text, item_end)
        if text[position] == "]":
            return
        if text[position] != ",":
            raise _InvalidJson
        position = _json_whitespace_end(text, position + 1)


_NON_STRING_JSON_VALUE: Final = object()


def _string_from_span(text: str, span: _JsonSpan) -> object:
    if _json_span_kind(text, span) != "string":
        return _NON_STRING_JSON_VALUE
    value, end = _scan_json_string(text, span.start)
    if end != span.end:
        raise _InvalidJson
    return value


def _strict_integer_from_span(text: str, span: _JsonSpan, *, field: str) -> int:
    if _json_span_kind(text, span) != "integer":
        raise _wire_error(f"wire.strict_integer:{field}")
    token = text[span.start : span.end]
    if token == "-0":
        raise _wire_error(f"wire.strict_integer:{field}")
    digits = token[1:] if token.startswith("-") else token
    if len(digits) > 10:
        return -1 if token.startswith("-") else _UINT32_MAX + 1
    return int(token)


def _scan_object_document(
    data: bytes,
    *,
    envelope: int,
    expected_fields: tuple[str, ...],
) -> tuple[str, dict[str, _JsonSpan], str | None] | None:
    text, span = _scan_document(data, envelope=envelope)
    if _json_span_kind(text, span) != "object":
        return None
    try:
        members, extra_name = _scan_object_semantic_members(
            text,
            span,
            expected_fields=expected_fields,
        )
    except _InvalidJson as exc:
        raise _wire_error("wire.invalid_json") from exc
    return text, members, extra_name


def _check_scanned_fields(
    members: dict[str, _JsonSpan],
    expected_fields: tuple[str, ...],
    document_extra_name: str | None,
) -> None:
    for name in expected_fields:
        if name not in members:
            raise _wire_error(f"wire.missing_field:{name}")
    expected = frozenset(expected_fields)
    extra_name = min(
        (
            name
            for name in (*members, document_extra_name)
            if name is not None and name not in expected
        ),
        key=_utf16_sort_key,
        default=None,
    )
    if extra_name is not None:
        raise _wire_error(f"wire.extra_field:{extra_name}")


def _check_schema_version(value: object, expected: str) -> str:
    if value != expected or not isinstance(value, str):
        raise _wire_error("wire.schema_version")
    return value


def _integer_range(value: int, *, field: str, minimum: int, maximum: int) -> int:
    if value < minimum or value > maximum:
        raise _wire_error(f"wire.integer_range:{field}")
    return value


def _index_range(value: int, *, field: str, maximum_exclusive: int) -> None:
    if value < 0 or value >= maximum_exclusive:
        raise _wire_error(f"wire.index_range:{field}")


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise _wire_error(f"wire.sha256_format:{field}")
    return value


def _literal(value: object, *, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise _wire_error(f"wire.literal_value:{field}")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_canonical(data: bytes, value: object) -> None:
    if _canonical_json_bytes(value) != data:
        raise _wire_error("wire.noncanonical_json")


def _run_spec_payload(value: NormalPredictiveRunSpecV1) -> dict[str, object]:
    return {
        "candidate_sha256": value.candidate_sha256,
        "draws_per_block": value.draws_per_block,
        "independent_block_count": value.independent_block_count,
        "public_task_sha256": value.public_task_sha256,
        "query_count": value.query_count,
        "query_set_sha256": value.query_set_sha256,
        "schema_version": value.schema_version,
        "visible_data_sha256": value.visible_data_sha256,
    }


def encode_normal_predictive_run_spec(value: NormalPredictiveRunSpecV1) -> bytes:
    if not isinstance(value, NormalPredictiveRunSpecV1):
        raise TypeError("value must be NormalPredictiveRunSpecV1")
    encoded = _canonical_json_bytes(_run_spec_payload(value))
    decode_normal_predictive_run_spec(encoded)
    return encoded


def decode_normal_predictive_run_spec(data: bytes) -> NormalPredictiveRunSpecV1:
    fields = (
        "candidate_sha256",
        "draws_per_block",
        "independent_block_count",
        "public_task_sha256",
        "query_count",
        "query_set_sha256",
        "schema_version",
        "visible_data_sha256",
    )
    scanned = _scan_object_document(
        data,
        envelope=_FIXED_ENVELOPE_BYTES,
        expected_fields=fields,
    )
    if scanned is None:
        raise _wire_error(f"wire.missing_field:{fields[0]}")
    text, members, extra_name = scanned
    field_set = frozenset(fields)
    for name in _logical_shape_field_order(fields, extra_name):
        if name not in field_set:
            raise _wire_error(f"wire.extra_field:{name}")
        if name not in members:
            raise _wire_error(f"wire.missing_field:{name}")
    schema_version = _check_schema_version(
        _string_from_span(text, members["schema_version"]),
        _RUN_SCHEMA_VERSION,
    )
    candidate_sha256 = _sha256(
        _string_from_span(text, members["candidate_sha256"]),
        field="candidate_sha256",
    )
    draws_per_block = _integer_range(
        _strict_integer_from_span(
            text,
            members["draws_per_block"],
            field="draws_per_block",
        ),
        field="draws_per_block",
        minimum=1,
        maximum=_UINT32_MAX,
    )
    independent_block_count = _integer_range(
        _strict_integer_from_span(
            text,
            members["independent_block_count"],
            field="independent_block_count",
        ),
        field="independent_block_count",
        minimum=2,
        maximum=_UINT16_MAX,
    )
    public_task_sha256 = _sha256(
        _string_from_span(text, members["public_task_sha256"]),
        field="public_task_sha256",
    )
    query_count = _integer_range(
        _strict_integer_from_span(
            text,
            members["query_count"],
            field="query_count",
        ),
        field="query_count",
        minimum=1,
        maximum=_UINT32_MAX,
    )
    query_set_sha256 = _sha256(
        _string_from_span(text, members["query_set_sha256"]),
        field="query_set_sha256",
    )
    visible_data_sha256 = _sha256(
        _string_from_span(text, members["visible_data_sha256"]),
        field="visible_data_sha256",
    )
    _require_derived_documents_fit(
        draws_per_block=draws_per_block,
        independent_block_count=independent_block_count,
        query_count=query_count,
    )
    logical = {
        "candidate_sha256": candidate_sha256,
        "draws_per_block": draws_per_block,
        "independent_block_count": independent_block_count,
        "public_task_sha256": public_task_sha256,
        "query_count": query_count,
        "query_set_sha256": query_set_sha256,
        "schema_version": schema_version,
        "visible_data_sha256": visible_data_sha256,
    }
    _require_canonical(data, logical)
    return NormalPredictiveRunSpecV1(**logical)


def encode_f64_bits(value: float) -> F64Bits:
    if not isinstance(value, float):
        raise TypeError("value must be a binary64 float")
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if value == 0.0:
        return "0000000000000000"
    return struct.pack(">d", value).hex()


def decode_f64_bits(value: F64Bits) -> float:
    accepted = _validate_f64_bits(value, path="f64_bits")
    return struct.unpack(">d", bytes.fromhex(accepted))[0]


def _block_envelope(run_spec: NormalPredictiveRunSpecV1) -> int:
    block_length, _ = _require_derived_documents_fit(
        draws_per_block=run_spec.draws_per_block,
        independent_block_count=run_spec.independent_block_count,
        query_count=run_spec.query_count,
    )
    return min(
        _MAX_DOCUMENT_BYTES,
        _checked_add(block_length, _FIXED_ENVELOPE_BYTES),
    )


def _block_payload(value: NormalJointSampleBlockV1) -> dict[str, object]:
    return {
        "block_index": value.block_index,
        "draws": [
            {
                "draw_index": draw.draw_index,
                "targets_by_query": list(draw.targets_by_query),
            }
            for draw in value.draws
        ],
        "run_spec_sha256": value.run_spec_sha256,
        "schema_version": value.schema_version,
    }


def encode_normal_joint_sample_block(
    value: NormalJointSampleBlockV1,
    *,
    run_spec: NormalPredictiveRunSpecV1,
) -> bytes:
    if not isinstance(value, NormalJointSampleBlockV1):
        raise TypeError("value must be NormalJointSampleBlockV1")
    encoded = _canonical_json_bytes(_block_payload(value))
    decode_normal_joint_sample_block(encoded, run_spec=run_spec)
    return encoded


def decode_normal_joint_sample_block(
    data: bytes,
    *,
    run_spec: NormalPredictiveRunSpecV1,
) -> NormalJointSampleBlockV1:
    block_fields = ("block_index", "draws", "run_spec_sha256", "schema_version")
    draw_fields = ("draw_index", "targets_by_query")
    try:
        text, root_span = _scan_document(data, envelope=_block_envelope(run_spec))
        if _json_span_kind(text, root_span) != "object":
            raise _wire_error("wire.missing_field:block_index")

        block_members, block_extra_name = _scan_object_semantic_members(
            text,
            root_span,
            expected_fields=block_fields,
        )

        for name in _logical_shape_field_order(block_fields, block_extra_name):
            if name not in block_fields:
                raise _wire_error(f"wire.extra_field:{name}")
            if name not in block_members:
                raise _wire_error(f"wire.missing_field:{name}")
            if name != "draws":
                continue
            draws_span = block_members[name]
            if _json_span_kind(text, draws_span) != "array":
                continue
            for draw_span in _iter_array_items(text, draws_span):
                if _json_span_kind(text, draw_span) != "object":
                    raise _wire_error("wire.missing_field:draw_index")
                draw_members, draw_extra_name = _scan_object_semantic_members(
                    text,
                    draw_span,
                    expected_fields=draw_fields,
                )
                for draw_name in _logical_shape_field_order(
                    draw_fields,
                    draw_extra_name,
                ):
                    if draw_name not in draw_fields:
                        raise _wire_error(f"wire.extra_field:{draw_name}")
                    if draw_name not in draw_members:
                        raise _wire_error(f"wire.missing_field:{draw_name}")

        draws_span = block_members["draws"]

        schema_version = _check_schema_version(
            _string_from_span(text, block_members["schema_version"]),
            _BLOCK_SCHEMA_VERSION,
        )

        block_index = _integer_range(
            _strict_integer_from_span(
                text,
                block_members["block_index"],
                field="block_index",
            ),
            field="block_index",
            minimum=0,
            maximum=_UINT16_MAX,
        )
        if _json_span_kind(text, draws_span) == "array":
            for draw_span in _iter_array_items(text, draws_span):
                draw_members, _ = _scan_object_semantic_members(
                    text,
                    draw_span,
                    expected_fields=draw_fields,
                )
                _integer_range(
                    _strict_integer_from_span(
                        text,
                        draw_members["draw_index"],
                        field="draw_index",
                    ),
                    field="draw_index",
                    minimum=0,
                    maximum=_UINT32_MAX,
                )
        run_spec_sha256 = _sha256(
            _string_from_span(text, block_members["run_spec_sha256"]),
            field="run_spec_sha256",
        )

        exact_length = _exact_block_document_length(
            draws_per_block=run_spec.draws_per_block,
            query_count=run_spec.query_count,
            block_index=block_index,
        )
        if exact_length > _MAX_DOCUMENT_BYTES:
            raise _wire_error("wire.byte_limit_exceeded")

        if _json_span_kind(text, draws_span) == "array":
            for draw_position, draw_span in enumerate(
                _iter_array_items(text, draws_span)
            ):
                draw_members, _ = _scan_object_semantic_members(
                    text,
                    draw_span,
                    expected_fields=draw_fields,
                )
                target_span = draw_members["targets_by_query"]
                if _json_span_kind(text, target_span) != "array":
                    continue
                for query_position, target in enumerate(
                    _iter_array_items(text, target_span)
                ):
                    _validate_f64_bits(
                        _string_from_span(text, target),
                        path=(
                            f"block.draws[{draw_position}]"
                            f".targets_by_query[{query_position}]"
                        ),
                    )

        _index_range(
            block_index,
            field="block_index",
            maximum_exclusive=run_spec.independent_block_count,
        )
        if _json_span_kind(text, draws_span) != "array":
            raise _wire_error("wire.count_mismatch:draws")

        draw_count = 0
        for _ in _iter_array_items(text, draws_span):
            draw_count += 1
            if draw_count > run_spec.draws_per_block:
                raise _wire_error("wire.count_mismatch:draws")
        if draw_count != run_spec.draws_per_block:
            raise _wire_error("wire.count_mismatch:draws")

        seen_draw_indexes: set[int] = set()
        for expected_index, draw_span in enumerate(_iter_array_items(text, draws_span)):
            draw_members, _ = _scan_object_semantic_members(
                text,
                draw_span,
                expected_fields=draw_fields,
            )
            actual_index = _strict_integer_from_span(
                text,
                draw_members["draw_index"],
                field="draw_index",
            )
            _index_range(
                actual_index,
                field="draw_index",
                maximum_exclusive=run_spec.draws_per_block,
            )
            if actual_index in seen_draw_indexes:
                raise _wire_error(f"wire.duplicate_index:draw_index:{actual_index}")
            seen_draw_indexes.add(actual_index)
            if actual_index != expected_index:
                raise _wire_error(
                    "wire.index_order:draw_index:"
                    f"expected:{expected_index}:actual:{actual_index}"
                )

            target_span = draw_members["targets_by_query"]
            if _json_span_kind(text, target_span) != "array":
                raise _wire_error("wire.count_mismatch:targets_by_query")
            target_count = 0
            for _ in _iter_array_items(text, target_span):
                target_count += 1
                if target_count > run_spec.query_count:
                    raise _wire_error("wire.count_mismatch:targets_by_query")
            if target_count != run_spec.query_count:
                raise _wire_error("wire.count_mismatch:targets_by_query")

        if len(data) != exact_length:
            raise _wire_error("wire.noncanonical_json")

        materialized_draws: list[_MaterializedDraw] = []
        for draw_position, draw_span in enumerate(_iter_array_items(text, draws_span)):
            draw_members, _ = _scan_object_semantic_members(
                text,
                draw_span,
                expected_fields=draw_fields,
            )
            draw_index = _strict_integer_from_span(
                text,
                draw_members["draw_index"],
                field="draw_index",
            )
            target_values: list[str] = []
            for query_position, target_span in enumerate(
                _iter_array_items(text, draw_members["targets_by_query"])
            ):
                target_values.append(
                    _validate_f64_bits(
                        _string_from_span(text, target_span),
                        path=(
                            f"block.draws[{draw_position}]"
                            f".targets_by_query[{query_position}]"
                        ),
                    )
                )
            materialized_draws.append(
                {
                    "draw_index": draw_index,
                    "targets_by_query": target_values,
                }
            )

        logical = {
            "block_index": block_index,
            "draws": materialized_draws,
            "run_spec_sha256": run_spec_sha256,
            "schema_version": schema_version,
        }
        _require_canonical(data, logical)

        expected_run_hash = hashlib.sha256(
            encode_normal_predictive_run_spec(run_spec)
        ).hexdigest()
        if run_spec_sha256 != expected_run_hash:
            raise _wire_error("wire.context_mismatch:run_spec_sha256")

        return NormalJointSampleBlockV1(
            block_index=block_index,
            draws=tuple(
                NormalJointDrawV1(
                    draw_index=draw["draw_index"],
                    targets_by_query=tuple(draw["targets_by_query"]),
                )
                for draw in materialized_draws
            ),
            run_spec_sha256=run_spec_sha256,
            schema_version=schema_version,
        )
    except _InvalidJson as exc:
        raise _wire_error("wire.invalid_json") from exc


def _attempt_payload(value: NormalPredictiveWorkerAttemptV1) -> dict[str, object]:
    if isinstance(value, NormalPredictiveWorkerSuccessV1):
        return {
            "block_index": value.block_index,
            "kind": value.kind,
            "run_spec_sha256": value.run_spec_sha256,
            "sample_block_sha256": value.sample_block_sha256,
            "schema_version": value.schema_version,
        }
    return {
        "block_index": value.block_index,
        "disposition": value.disposition,
        "kind": value.kind,
        "run_spec_sha256": value.run_spec_sha256,
        "schema_version": value.schema_version,
    }


def encode_normal_predictive_worker_attempt(
    value: NormalPredictiveWorkerAttemptV1,
    *,
    run_spec: NormalPredictiveRunSpecV1,
) -> bytes:
    if not isinstance(
        value,
        NormalPredictiveWorkerSuccessV1 | NormalPredictiveWorkerFailureV1,
    ):
        raise TypeError("value must be a NormalPredictiveWorkerAttemptV1")
    encoded = _canonical_json_bytes(_attempt_payload(value))
    decode_normal_predictive_worker_attempt(encoded, run_spec=run_spec)
    return encoded


def decode_normal_predictive_worker_attempt(
    data: bytes,
    *,
    run_spec: NormalPredictiveRunSpecV1,
) -> NormalPredictiveWorkerAttemptV1:
    common_fields = ("block_index", "kind", "run_spec_sha256", "schema_version")
    union_fields = (
        "block_index",
        "disposition",
        "kind",
        "run_spec_sha256",
        "sample_block_sha256",
        "schema_version",
    )
    scanned = _scan_object_document(
        data,
        envelope=_FIXED_ENVELOPE_BYTES,
        expected_fields=union_fields,
    )
    if scanned is None:
        raise _wire_error(f"wire.missing_field:{common_fields[0]}")
    text, members, extra_name = scanned
    for name in common_fields:
        if name not in members:
            raise _wire_error(f"wire.missing_field:{name}")
    if extra_name is not None:
        raise _wire_error(f"wire.extra_field:{extra_name}")

    schema_version = _check_schema_version(
        _string_from_span(text, members["schema_version"]),
        _ATTEMPT_SCHEMA_VERSION,
    )
    kind = _literal(
        _string_from_span(text, members["kind"]),
        field="kind",
        allowed=frozenset({"success", "failure"}),
    )
    if kind == "success":
        variant_fields = (
            "block_index",
            "kind",
            "run_spec_sha256",
            "sample_block_sha256",
            "schema_version",
        )
    else:
        variant_fields = (
            "block_index",
            "disposition",
            "kind",
            "run_spec_sha256",
            "schema_version",
        )
    _check_scanned_fields(members, variant_fields, extra_name)

    disposition: str | None = None
    if kind == "failure":
        disposition = _literal(
            _string_from_span(text, members["disposition"]),
            field="disposition",
            allowed=_FAILURE_DISPOSITIONS,
        )

    block_index = _integer_range(
        _strict_integer_from_span(
            text,
            members["block_index"],
            field="block_index",
        ),
        field="block_index",
        minimum=0,
        maximum=_UINT16_MAX,
    )
    _index_range(
        block_index,
        field="block_index",
        maximum_exclusive=run_spec.independent_block_count,
    )
    run_spec_sha256 = _sha256(
        _string_from_span(text, members["run_spec_sha256"]),
        field="run_spec_sha256",
    )
    sample_block_sha256: str | None = None
    if kind == "success":
        sample_block_sha256 = _sha256(
            _string_from_span(text, members["sample_block_sha256"]),
            field="sample_block_sha256",
        )

    if kind == "success":
        logical = {
            "block_index": block_index,
            "kind": kind,
            "run_spec_sha256": run_spec_sha256,
            "sample_block_sha256": sample_block_sha256,
            "schema_version": schema_version,
        }
    else:
        logical = {
            "block_index": block_index,
            "disposition": disposition,
            "kind": kind,
            "run_spec_sha256": run_spec_sha256,
            "schema_version": schema_version,
        }
    _require_canonical(data, logical)
    expected_run_hash = hashlib.sha256(
        encode_normal_predictive_run_spec(run_spec)
    ).hexdigest()
    if run_spec_sha256 != expected_run_hash:
        raise _wire_error("wire.context_mismatch:run_spec_sha256")

    if kind == "success":
        assert sample_block_sha256 is not None
        return NormalPredictiveWorkerSuccessV1(
            block_index=block_index,
            kind="success",
            run_spec_sha256=run_spec_sha256,
            sample_block_sha256=sample_block_sha256,
            schema_version=schema_version,
        )
    assert disposition is not None
    return NormalPredictiveWorkerFailureV1(
        block_index=block_index,
        disposition=disposition,
        kind="failure",
        run_spec_sha256=run_spec_sha256,
        schema_version=schema_version,
    )


def _validate_ordered_block_indexes(
    values: Sequence[NormalJointSampleBlockV1],
    *,
    maximum_exclusive: int,
    expected_indexes: Sequence[int] | None = None,
) -> None:
    seen: set[int] = set()
    for position, value in enumerate(values):
        actual = value.block_index
        _index_range(
            actual,
            field="block_index",
            maximum_exclusive=maximum_exclusive,
        )
        if actual in seen:
            raise _wire_error(f"wire.duplicate_index:block_index:{actual}")
        seen.add(actual)
        expected = position if expected_indexes is None else expected_indexes[position]
        if actual != expected:
            raise _wire_error(
                f"wire.index_order:block_index:expected:{expected}:actual:{actual}"
            )


def validate_normal_joint_sample_blocks(
    blocks: Sequence[NormalJointSampleBlockV1],
    *,
    run_spec: NormalPredictiveRunSpecV1,
) -> None:
    if len(blocks) != run_spec.independent_block_count:
        raise _wire_error("wire.count_mismatch:blocks")
    _validate_ordered_block_indexes(
        blocks,
        maximum_exclusive=run_spec.independent_block_count,
    )
    _checked_mul(run_spec.independent_block_count, run_spec.draws_per_block)
    for block in blocks:
        encode_normal_joint_sample_block(block, run_spec=run_spec)


def validate_normal_predictive_worker_attempts(
    attempts: Sequence[NormalPredictiveWorkerAttemptV1],
    *,
    run_spec: NormalPredictiveRunSpecV1,
    sample_blocks: Sequence[NormalJointSampleBlockV1],
) -> None:
    if len(attempts) != run_spec.independent_block_count:
        raise _wire_error("wire.count_mismatch:attempts")

    seen_attempt_indexes: set[int] = set()
    for expected_index, attempt in enumerate(attempts):
        actual_index = attempt.block_index
        _index_range(
            actual_index,
            field="block_index",
            maximum_exclusive=run_spec.independent_block_count,
        )
        if actual_index in seen_attempt_indexes:
            raise _wire_error(f"wire.duplicate_index:block_index:{actual_index}")
        seen_attempt_indexes.add(actual_index)
        if actual_index != expected_index:
            raise _wire_error(
                "wire.index_order:block_index:"
                f"expected:{expected_index}:actual:{actual_index}"
            )

    successes = tuple(
        attempt
        for attempt in attempts
        if isinstance(attempt, NormalPredictiveWorkerSuccessV1)
    )
    if len(sample_blocks) != len(successes):
        raise _wire_error("wire.count_mismatch:blocks")

    expected_block_indexes = tuple(success.block_index for success in successes)
    _validate_ordered_block_indexes(
        sample_blocks,
        maximum_exclusive=run_spec.independent_block_count,
        expected_indexes=expected_block_indexes,
    )
    for attempt in attempts:
        encode_normal_predictive_worker_attempt(attempt, run_spec=run_spec)
    for success, block in zip(successes, sample_blocks, strict=True):
        encoded_block = encode_normal_joint_sample_block(block, run_spec=run_spec)
        if success.sample_block_sha256 != hashlib.sha256(encoded_block).hexdigest():
            raise _wire_error("wire.context_mismatch:sample_block_sha256")


__all__ = [
    "F64Bits",
    "NormalJointDrawV1",
    "NormalJointSampleBlockV1",
    "NormalPredictiveRunSpecV1",
    "NormalPredictiveWorkerAttemptV1",
    "NormalPredictiveWorkerFailureV1",
    "NormalPredictiveWorkerSuccessV1",
    "PredictiveWireError",
    "decode_f64_bits",
    "decode_normal_joint_sample_block",
    "decode_normal_predictive_run_spec",
    "decode_normal_predictive_worker_attempt",
    "encode_f64_bits",
    "encode_normal_joint_sample_block",
    "encode_normal_predictive_run_spec",
    "encode_normal_predictive_worker_attempt",
    "validate_normal_joint_sample_blocks",
    "validate_normal_predictive_worker_attempts",
]

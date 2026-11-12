"""The stop-reason vocabulary: upstream's finish reasons, plus "unknown".

checked vllm==0.30.0 (vllm/v1/engine/__init__.py, FINISH_REASON_STRINGS),
openai==2.54.0 (openai/types/chat/chat_completion.py, Choice.finish_reason) and
harbor==0.25.0 (harbor/models/trajectories/*): the first two define the finish
reasons a generation can carry; Harbor's ATIF trajectory model defines none
(its only place for one is the free-form Step.extra). None of them has a value
for a reason that was never recorded. The members below re-use the upstream
strings as they are and add only UNKNOWN.

A generation whose stop reason is missing, null or empty reads UNKNOWN, never a
finished generation: it may have been cut by a length limit. A generation cut by
a length limit and one whose stop reason was never recorded are cut generations:
interrupted rollouts that a batch builder leaves out, never a verdict on the
program.
"""
from __future__ import annotations

from enum import Enum


class StopReason(str, Enum):
    """The recorded reason a generation stopped.

    A str-valued Enum, so a member equals the string a server returned and the
    string stored in a record.
    """

    # The OpenAI-style chat API's finish_reason values (also what vLLM serves).
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"
    # vLLM's own finish reasons beyond that set.
    ABORT = "abort"
    ERROR = "error"
    REPETITION = "repetition"
    # Ours: the reason was never recorded (missing, null or empty).
    UNKNOWN = "unknown"


# The stop reasons of a generation that did not finish. Such a generation is
# masked where the batch is built: no row, no reward, counted.
CUT_STOP_REASONS: frozenset[StopReason] = frozenset({StopReason.LENGTH, StopReason.UNKNOWN})


def recorded_stop_reason(value: object) -> str:
    """The stop reason a generation recorded, or "unknown" when it recorded none.

    A missing, null, blank or non-string stop reason is "unknown", never a
    finished generation. A recorded string is returned as recorded.
    """
    if not isinstance(value, str) or not value.strip():
        return StopReason.UNKNOWN.value
    return value

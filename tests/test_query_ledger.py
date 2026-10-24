import json


def test_query_ledger_event_round_trips_without_reconstructing_environment_facts():
    from synthstats_schemas import QueryLedgerEvent

    payload = {
        "request_id": "episode-1/request-1",
        "tool": "measurement-view",
        "arguments_digest": "a" * 64,
        "charge": {
            "acquisition_credits": 1,
            "assistant_turns": 1,
            "execution_calls": 0,
            "execution_time_ms": 0,
            "generation_tokens": 12,
        },
        "remaining": {
            "acquisition_credits": 7,
            "assistant_turns": 15,
            "execution_calls": 2,
            "execution_time_ms": 1000,
            "generation_tokens": 8180,
        },
        "outcome": "resolved",
        "observation": {"count": 7},
        "refusal_reason": None,
    }
    event = QueryLedgerEvent.model_validate(payload)
    assert json.loads(event.model_dump_json()) == payload
    assert QueryLedgerEvent.model_validate_json(event.model_dump_json()) == event

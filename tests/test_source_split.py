"""The shared TRAIN/EVAL source-split contract rejects duplicate and shared task ids."""

import pytest
from pydantic import ValidationError

from synthstats_schemas import SourceTaskSplit

VALID_FULL = {
    "training": ["task-a", "task-b"],
    "evaluation": ["task-c"],
    "training_groups": ["group-a", "group-b"],
    "evaluation_groups": ["group-c"],
    "package_group_sides": [
        {"task": "task-a", "group": "group-a", "side": "training"},
        {"task": "task-b", "group": "group-b", "side": "training"},
        {"task": "task-c", "group": "group-c", "side": "evaluation"},
    ],
}

VALID_TRAIN_ONLY = {
    "training": ["task-a"],
    "evaluation": [],
    "training_groups": ["group-a"],
    "evaluation_groups": [],
    "package_group_sides": [
        {"task": "task-a", "group": "group-a", "side": "training"},
    ],
}


def variant(**overrides):
    payload = {**VALID_FULL, **overrides}
    return payload


def test_valid_split_round_trips():
    split = SourceTaskSplit.model_validate(VALID_FULL)
    assert split.model_dump() == VALID_FULL


def test_train_only_split_round_trips():
    split = SourceTaskSplit.model_validate(VALID_TRAIN_ONLY)
    assert split.model_dump() == VALID_TRAIN_ONLY


def test_repeated_training_task_id_is_refused():
    with pytest.raises(ValidationError, match="(?i)(repeat|duplicat).*training"):
        SourceTaskSplit.model_validate(variant(training=["task-a", "task-a"]))


def test_repeated_evaluation_task_id_is_refused():
    with pytest.raises(ValidationError, match="(?i)(repeat|duplicat).*evaluation"):
        SourceTaskSplit.model_validate(variant(evaluation=["task-c", "task-c"]))


def test_task_in_both_sides_is_refused():
    with pytest.raises(ValidationError, match="(?i)(both|training.*evaluation|evaluation.*training)"):
        SourceTaskSplit.model_validate(variant(evaluation=["task-a"]))


def test_repeated_group_id_is_refused():
    with pytest.raises(ValidationError, match="(?i)(repeat|duplicat).*group"):
        SourceTaskSplit.model_validate(variant(training_groups=["group-a", "group-a"]))


def test_repeated_package_group_side_task_is_refused():
    sides = VALID_FULL["package_group_sides"] + [
        {"task": "task-a", "group": "group-a", "side": "training"},
    ]
    with pytest.raises(ValidationError, match="(?i)(repeat|duplicat|twice).*side"):
        SourceTaskSplit.model_validate(variant(package_group_sides=sides))


def test_side_entry_missing_from_task_lists_is_refused():
    sides = [row for row in VALID_FULL["package_group_sides"] if row["task"] != "task-c"]
    with pytest.raises(ValidationError, match="(?i)(cover|match|missing).*"):
        SourceTaskSplit.model_validate(variant(package_group_sides=sides))


def test_side_entry_group_outside_its_side_is_refused():
    sides = [
        {"task": "task-a", "group": "group-c", "side": "training"},
        *[row for row in VALID_FULL["package_group_sides"] if row["task"] != "task-a"],
    ]
    with pytest.raises(ValidationError, match="(?i)group"):
        SourceTaskSplit.model_validate(variant(package_group_sides=sides))


def test_side_entry_disagreeing_with_task_side_is_refused():
    sides = [
        {"task": "task-a", "group": "group-a", "side": "evaluation"},
        *[row for row in VALID_FULL["package_group_sides"] if row["task"] != "task-a"],
    ]
    with pytest.raises(ValidationError, match="(?i)(side|train)"):
        SourceTaskSplit.model_validate(variant(package_group_sides=sides))


def test_group_without_any_task_is_refused():
    with pytest.raises(ValidationError, match="(?i)unused"):
        SourceTaskSplit.model_validate(variant(training_groups=["group-a", "group-b", "group-d"]))

"""The shared TRAIN/EVAL source-task split contract."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from synthstats_schemas.base import ContractModel

TaskId = Annotated[str, Field(min_length=1)]
GroupId = Annotated[str, Field(min_length=1)]


class PackageGroupSide(ContractModel):
    """One task's source group and its side of the TRAIN/EVAL split."""

    task: TaskId
    group: GroupId
    side: Literal["training", "evaluation"]


class SourceTaskSplit(ContractModel):
    """The five-key source-task split contract.

    Producers and consumers parse the authenticated split through this model
    so that repeated task or group ids and ids shared between TRAIN and EVAL
    are refused before any set conversion can erase them.
    """

    training: list[TaskId]
    evaluation: list[TaskId]
    training_groups: list[GroupId]
    evaluation_groups: list[GroupId]
    package_group_sides: list[PackageGroupSide]

    @model_validator(mode="after")
    def tasks_and_groups_are_unique_disjoint_and_mapped(self) -> "SourceTaskSplit":
        for field_name, values in (
            ("training", self.training),
            ("evaluation", self.evaluation),
            ("training_groups", self.training_groups),
            ("evaluation_groups", self.evaluation_groups),
        ):
            seen: set[str] = set()
            for value in values:
                if value in seen:
                    raise ValueError(f"repeated id in {field_name}: {value}")
                seen.add(value)
        shared_tasks = set(self.training) & set(self.evaluation)
        if shared_tasks:
            raise ValueError(f"task id in both training and evaluation: {sorted(shared_tasks)}")
        shared_groups = set(self.training_groups) & set(self.evaluation_groups)
        if shared_groups:
            raise ValueError(
                f"group id in both training_groups and evaluation_groups: "
                f"{sorted(shared_groups)}"
            )
        sides: dict[str, PackageGroupSide] = {}
        for row in self.package_group_sides:
            if row.task in sides:
                raise ValueError(f"repeated task in package_group_sides: {row.task}")
            sides[row.task] = row
        covered = set(sides)
        split_tasks = set(self.training) | set(self.evaluation)
        if covered != split_tasks:
            raise ValueError(
                "package_group_sides must cover exactly the split's tasks; "
                f"missing: {sorted(split_tasks - covered)}, "
                f"unknown: {sorted(covered - split_tasks)}"
            )
        for side_name, tasks in (("training", self.training), ("evaluation", self.evaluation)):
            side_groups = set(
                self.training_groups if side_name == "training" else self.evaluation_groups
            )
            used_groups: set[str] = set()
            for task in tasks:
                row = sides[task]
                if row.side != side_name:
                    raise ValueError(
                        f"task {task} is in {side_name} but its package_group_side "
                        f"says {row.side}"
                    )
                if row.group not in side_groups:
                    raise ValueError(
                        f"group {row.group} for task {task} is not in "
                        f"{side_name}_groups"
                    )
                used_groups.add(row.group)
            if used_groups != side_groups:
                raise ValueError(
                    f"{side_name}_groups must be exactly the groups used by its tasks; "
                    f"unused: {sorted(side_groups - used_groups)}"
                )
        return self

"""The Weights & Biases privacy check that ``check_launch`` calls before a paid launch.

"""

from __future__ import annotations

import pytest

from synthstats_schemas import wandb_privacy as probe


def _api(viewer="example-user", project=None):
    """A W&B GraphQL endpoint that answers the probe's query with the given viewer and project."""
    calls = []

    def execute(query, variables):
        calls.append((query, dict(variables)))
        return {"viewer": {"username": viewer, "entity": "example-team"}, "project": project}

    return execute, calls


def test_accepts_a_private_project_of_the_keys_own_user():
    execute, calls = _api(project={"access": "PRIVATE"})

    probe.private_project_probe("example-user", "example-project", execute=execute)

    assert calls[0][1] == {"entity": "example-user", "project": "example-project"}


@pytest.mark.parametrize("access", ["USER_READ", "USER_WRITE", "ENTITY_READ", "RESTRICTED", None])
def test_refuses_a_project_that_is_not_private(access):
    execute, _ = _api(project={"access": access})

    with pytest.raises(probe.WandbProjectNotPrivate):
        probe.private_project_probe("example-user", "example-project", execute=execute)


def test_refuses_a_project_that_does_not_exist():
    # W&B would create it on first wandb.init with the entity's default visibility, which is not checked.
    execute, _ = _api(project=None)

    with pytest.raises(probe.WandbProjectNotPrivate):
        probe.private_project_probe("example-user", "new-project", execute=execute)


def test_refuses_a_key_whose_user_is_not_the_declared_entity():
    # A key for another user must not authorise the declared account.
    execute, _ = _api(viewer="someone-else", project={"access": "PRIVATE"})

    with pytest.raises(probe.WandbProjectNotPrivate):
        probe.private_project_probe("example-user", "example-project", execute=execute)


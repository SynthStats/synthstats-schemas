"""The Weights & Biases (W&B) privacy check that launchers pass to
``check_launch`` as ``wandb_probe``.

A run may log only to an existing project of the key's own user whose access is "PRIVATE". A project that W&B creates on first use takes the entity's default visibility. The probe only reads; it never creates or changes
a project.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

PRIVATE_ACCESS = "PRIVATE"

_QUERY = """
query LaunchGateProbe($entity: String!, $project: String!) {
  viewer { username }
  project(name: $project, entityName: $entity) { access }
}
"""


class WandbProjectNotPrivate(RuntimeError):
    """The project is missing, not private, or not the key's own user's."""


def private_project_probe(
    entity: str,
    project: str,
    *,
    execute: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> None:
    """Raise WandbProjectNotPrivate unless ``entity/project`` exists, is private and belongs to the key's user.

    ``execute(query, variables)`` runs a GraphQL query and returns its data; by default it is the W&B SDK's.
    """
    data = (execute or _execute_with_wandb)(_QUERY, {"entity": entity, "project": project})
    if (data.get("viewer") or {}).get("username") != entity:
        raise WandbProjectNotPrivate("the W&B key does not belong to the declared entity")
    found = data.get("project")
    if found is None:
        raise WandbProjectNotPrivate("the W&B project does not exist; create it as a private project first")
    if found.get("access") != PRIVATE_ACCESS:
        raise WandbProjectNotPrivate("the W&B project is not private")


def _execute_with_wandb(query: str, variables: Mapping[str, Any]) -> Mapping[str, Any]:
    import wandb  # only launchers that call the probe need the W&B SDK

    api = wandb.Api(timeout=30)
    # W&B 0.30 runs GraphQL through wandb-core; its public Api has no other query method.
    service = getattr(api, "_service_api", None)
    if service is None:
        raise RuntimeError("this probe needs the W&B 0.30 SDK, which runs GraphQL through Api._service_api")
    return service.execute_graphql(query, dict(variables))

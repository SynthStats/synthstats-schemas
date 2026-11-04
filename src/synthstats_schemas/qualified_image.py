"""The qualified scorer-image receipt shared by its writer and consumers."""

from typing import Annotated, Literal

from pydantic import Field

from synthstats_schemas.base import ContractModel


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
ImageDigest = Annotated[
    str, Field(pattern=r"^sha256:[0-9a-f]{64}$", min_length=71, max_length=71),
]


class QualifiedImageReceiptV1(ContractModel):
    schema_version: Literal["synthstats.qualified_image.v1"]
    status: Literal["qualified"]
    controller_image: ImageDigest
    environment_manifest_sha256: Sha256
    image_digest: ImageDigest
    metadata_return_sha256: Sha256
    packages: dict[str, str]
    python: str
    runtime_packages: dict[str, str]
    runtime_registry_sha256: Sha256
    validator_sha256: Sha256

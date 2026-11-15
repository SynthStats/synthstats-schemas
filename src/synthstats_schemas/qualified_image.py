"""The qualified scorer-image receipt shared by its writer and consumers."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from synthstats_schemas.base import ContractModel, Sha256

ImageDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class QualifiedImageReceiptV1(ContractModel):
    """A qualified image with optional image and runtime wheel identity pairs.

    Supply both image identity hashes or omit both. Explicit nulls are rejected.
    When the image pair is present, image_digest identifies the config hash.
    Supply both runtime wheel fields or omit both. Explicit nulls are rejected.
    The wheel package version must match runtime_packages['synthstats-schemas'].
    """

    schema_version: Literal["synthstats.qualified_image.v1"]
    status: Literal["qualified"]
    controller_image: ImageDigest
    environment_manifest_sha256: Sha256
    image_digest: ImageDigest
    image_manifest_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    image_config_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    metadata_return_sha256: Sha256
    packages: dict[str, str]
    python: str
    runtime_packages: dict[str, str]
    runtime_wheel_sha256: Sha256 | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    runtime_wheel_package_version: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None,
    )
    validator_sha256: Sha256

    @model_validator(mode="after")
    def validate_image_identity(self) -> Self:
        identity_fields = {"image_manifest_sha256", "image_config_sha256"}
        if not identity_fields.intersection(self.model_fields_set):
            return self
        if self.image_manifest_sha256 is None or self.image_config_sha256 is None:
            raise ValueError("image manifest and config SHA-256 must both be supplied and non-null")
        if self.image_digest != "sha256:" + self.image_config_sha256:
            raise ValueError("image_digest must identify image_config_sha256")
        return self

    @model_validator(mode="after")
    def validate_runtime_wheel_identity(self) -> Self:
        wheel_fields = {"runtime_wheel_sha256", "runtime_wheel_package_version"}
        if not wheel_fields.intersection(self.model_fields_set):
            return self
        if self.runtime_wheel_sha256 is None or self.runtime_wheel_package_version is None:
            raise ValueError("runtime wheel SHA-256 and package version must both be supplied and non-null")
        if self.runtime_wheel_package_version != self.runtime_packages.get("synthstats-schemas"):
            raise ValueError("runtime_wheel_package_version must match runtime_packages['synthstats-schemas']")
        return self

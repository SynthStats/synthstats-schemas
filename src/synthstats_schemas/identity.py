"""Installed implementation identity for synthstats-schemas."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from importlib import import_module, metadata
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, Field, StringConstraints

from synthstats_schemas.base import ContractModel

_DISTRIBUTION = "synthstats-schemas"
_READ_SIZE = 1024 * 1024

_StatFingerprint: TypeAlias = tuple[int, int, int, int, int]
_SourceInventoryEntry: TypeAlias = tuple[str, Path, _StatFingerprint]


def _require_nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("version must not be blank")
    return value


_NonBlankStr: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True),
    AfterValidator(_require_nonblank),
]
_Sha256: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
_PositiveInt: TypeAlias = Annotated[int, Field(strict=True, gt=0)]


class SynthStatsSchemasIdentity(ContractModel):
    """Identity of the installed synthstats-schemas implementation."""

    schema_version: Literal["synthstats.schemas_identity.v1"]
    distribution: Literal["synthstats-schemas"]
    version: _NonBlankStr
    implementation_sha256: _Sha256
    source_file_count: _PositiveInt


def synthstats_schemas_identity() -> SynthStatsSchemasIdentity:
    """Capture the installed distribution metadata and Python source identity."""

    package_root = _installed_package_root()
    distribution_before = _installed_distribution_metadata()
    source_inventory = _python_source_inventory(package_root)

    manifest: dict[str, dict[str, object]] = {}
    for relative_path, source_path, expected_stat in source_inventory:
        if relative_path in manifest:
            raise RuntimeError(
                f"duplicate synthstats-schemas source path: {relative_path}"
            )
        manifest[relative_path] = _stable_source_record(
            source_path,
            expected_stat=expected_stat,
        )

    if source_inventory != _python_source_inventory(package_root):
        raise RuntimeError(
            "synthstats-schemas source inventory changed during identity capture"
        )

    distribution_after = _installed_distribution_metadata()
    if distribution_after != distribution_before:
        raise RuntimeError(
            "synthstats-schemas distribution metadata changed during identity capture"
        )

    canonical_manifest = _canonical_json(manifest).encode("utf-8")
    distribution_name, version = distribution_after
    return SynthStatsSchemasIdentity(
        schema_version="synthstats.schemas_identity.v1",
        distribution=distribution_name,
        version=version,
        implementation_sha256=hashlib.sha256(canonical_manifest).hexdigest(),
        source_file_count=len(source_inventory),
    )


def require_synthstats_schemas_identity(
    expected: SynthStatsSchemasIdentity,
) -> SynthStatsSchemasIdentity:
    """Return the current identity or reject an exact identity mismatch."""

    actual = synthstats_schemas_identity()
    if actual != expected:
        expected_json = _canonical_json(expected.model_dump(mode="json"))
        actual_json = _canonical_json(actual.model_dump(mode="json"))
        raise RuntimeError(
            "synthstats-schemas identity mismatch: "
            f"expected={expected_json}; actual={actual_json}"
        )
    return actual


def _installed_package_root() -> Path:
    package = import_module("synthstats_schemas")
    module_file = getattr(package, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(
            "installed synthstats_schemas package has no implementation path"
        )

    unresolved_root = Path(module_file).parent
    root_stat = _lstat(unresolved_root, subject="package root")
    if stat.S_ISLNK(root_stat.st_mode):
        raise RuntimeError(
            f"refusing symlinked synthstats_schemas package root: {unresolved_root}"
        )
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(
            f"synthstats_schemas package root is not a directory: {unresolved_root}"
        )

    try:
        return unresolved_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"cannot resolve synthstats_schemas package root: {unresolved_root}"
        ) from exc


def _installed_distribution_metadata() -> tuple[Literal["synthstats-schemas"], str]:
    try:
        installed_distribution = metadata.distribution(_DISTRIBUTION)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "installed synthstats-schemas distribution metadata is missing"
        ) from exc

    raw_name = installed_distribution.metadata.get("Name")
    raw_version = installed_distribution.version
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise RuntimeError(
            "installed synthstats-schemas distribution name is missing or blank"
        )
    if not isinstance(raw_version, str) or not raw_version.strip():
        raise RuntimeError(
            "installed synthstats-schemas distribution version is missing or blank"
        )

    name = raw_name.strip()
    if name != _DISTRIBUTION:
        raise RuntimeError(
            "installed distribution name mismatch: "
            f"expected={_DISTRIBUTION}; actual={name}"
        )
    return _DISTRIBUTION, raw_version.strip()


def _python_source_inventory(package_root: Path) -> tuple[_SourceInventoryEntry, ...]:
    entries: list[_SourceInventoryEntry] = []
    seen: set[str] = set()
    try:
        for (
            directory_path,
            directory_names,
            file_names,
            directory_descriptor,
        ) in os.fwalk(package_root, topdown=True, follow_symlinks=False):
            directory_names.sort()
            file_names.sort()
            relative_directory = Path(directory_path).relative_to(package_root)

            for directory_name in directory_names:
                relative = relative_directory / directory_name
                source_directory = package_root / relative
                directory_stat = os.stat(
                    directory_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(directory_stat.st_mode):
                    raise RuntimeError(
                        "refusing symlinked synthstats_schemas source directory: "
                        f"{source_directory}"
                    )
                if not stat.S_ISDIR(directory_stat.st_mode):
                    raise RuntimeError(
                        "synthstats_schemas source parent is not a directory: "
                        f"{source_directory}"
                    )

            for file_name in file_names:
                relative = relative_directory / file_name
                source_path = package_root / relative
                source_stat = os.stat(
                    file_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(source_stat.st_mode):
                    raise RuntimeError(
                        f"refusing symlinked synthstats_schemas source: {source_path}"
                    )
                if source_path.suffix != ".py":
                    continue
                if not stat.S_ISREG(source_stat.st_mode):
                    raise RuntimeError(
                        "synthstats_schemas source is not a regular file: "
                        f"{source_path}"
                    )

                relative_path = relative.as_posix()
                if relative_path in seen:
                    raise RuntimeError(
                        f"duplicate synthstats-schemas source path: {relative_path}"
                    )
                seen.add(relative_path)
                entries.append(
                    (
                        relative_path,
                        source_path,
                        _stat_fingerprint(source_stat),
                    )
                )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot enumerate synthstats_schemas sources under {package_root}"
        ) from exc

    if not entries:
        raise RuntimeError(
            f"synthstats_schemas has no Python sources under {package_root}"
        )

    return tuple(sorted(entries, key=lambda entry: entry[0]))


def _stable_source_record(
    source_path: Path,
    *,
    expected_stat: _StatFingerprint,
) -> dict[str, object]:
    path_before = _lstat(source_path, subject="package source")
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise RuntimeError(
            f"refusing non-regular synthstats_schemas source: {source_path}"
        )
    if _stat_fingerprint(path_before) != expected_stat:
        raise RuntimeError(
            f"synthstats_schemas source changed before read: {source_path}"
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source_path, flags)
    except OSError as exc:
        raise RuntimeError(
            f"cannot open synthstats_schemas source without following links: {source_path}"
        ) from exc

    digest = hashlib.sha256()
    size_bytes = 0
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise RuntimeError(
                f"opened synthstats_schemas source is not regular: {source_path}"
            )
        if _stat_fingerprint(descriptor_before) != expected_stat:
            raise RuntimeError(
                f"synthstats_schemas source changed while opening: {source_path}"
            )

        while True:
            chunk = os.read(descriptor, _READ_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)

        descriptor_after = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError(
            f"cannot read synthstats_schemas source: {source_path}"
        ) from exc
    finally:
        os.close(descriptor)

    path_after = _lstat(source_path, subject="package source")
    observed_stats = (
        _stat_fingerprint(descriptor_before),
        _stat_fingerprint(descriptor_after),
        _stat_fingerprint(path_after),
    )
    if any(observed != expected_stat for observed in observed_stats):
        raise RuntimeError(
            f"synthstats_schemas source changed during read: {source_path}"
        )
    if size_bytes != descriptor_after.st_size:
        raise RuntimeError(
            f"synthstats_schemas source size changed during read: {source_path}"
        )

    return {"sha256": digest.hexdigest(), "size_bytes": size_bytes}


def _lstat(path: Path, *, subject: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"cannot inspect synthstats_schemas {subject}: {path}"
        ) from exc


def _stat_fingerprint(value: os.stat_result) -> _StatFingerprint:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

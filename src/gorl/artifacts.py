from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from .config import RunConfig

if TYPE_CHECKING:
    from .tracking import Tracker, WandbConfig


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RUNTIME_PACKAGES = (
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "jax-dataclasses",
    "brax",
    "flax",
    "optax",
    "mujoco",
    "mujoco-mjx",
    "playground",
    "numpy",
    "scipy",
    "ml-dtypes",
    "chex",
    "orbax-checkpoint",
    "tensorstore",
    "nvidia-cublas-cu12",
    "nvidia-cuda-cccl-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvshmem-cu12",
    "mediapy",
    "tyro",
    "wandb",
)
_MAX_UNTRACKED_FILES = 1_024
_MAX_UNTRACKED_BYTES = 64 * 1024 * 1024
_MAX_PATCH_BYTES = 128 * 1024 * 1024
_MAX_OMITTED_PATH_EXAMPLES = 16
_DISTRIBUTION_NAME = "gorl"
_DISTRIBUTION_RECORD_FORMAT = "gorl-normalized-installed-record-v1"
_DISTRIBUTION_METADATA_EXCLUSIONS = {
    "INSTALLER",
    "RECORD",
    "REQUESTED",
    "direct_url.json",
}


class ArtifactError(RuntimeError):
    pass


def benchmark_schedule_provenance(config: RunConfig) -> dict[str, Any]:
    """Describe the nominal benchmark axis separately from latent stages."""

    training = config.section("training")
    anchor = int(training.get("benchmark_start_env_steps", 0))
    latent_steps = tuple(int(value) for value in config.stage_steps)
    continuation = sum(latent_steps)
    nominal_steps = ((anchor,) + latent_steps) if anchor > 0 else latent_steps
    cumulative = 0
    nominal_coordinates = []
    for value in nominal_steps:
        cumulative += value
        nominal_coordinates.append(cumulative)
    return {
        "benchmark_anchor_env_steps": anchor,
        "benchmark_continuation_env_steps": continuation,
        "benchmark_end_env_steps": anchor + continuation,
        "latent_stage_env_steps": list(latent_steps),
        "nominal_stage_env_steps": list(nominal_steps),
        "nominal_stage_end_env_steps": nominal_coordinates,
    }


def trainer_backend(config: RunConfig) -> str:
    """Return the concrete trainer selected by the public method and profile."""

    if config.method == "ppo":
        if config.profile == "humanoid":
            return "brax_official_ppo"
        return "upstream_fpo_ppo"
    if config.method == "fpo":
        return "upstream_fpo"
    if config.method == "dppo":
        return "upstream_fpo_denoising_mdp"
    if config.profile == "humanoid":
        return "native_humanoid_brax_gorl"
    return "native_dm_control_gorl"


def _actual_action_semantics(
    resolved: Mapping[str, Any],
) -> str | None:
    environment = resolved.get("environment")
    if not isinstance(environment, Mapping):
        return None
    value = environment.get("action_semantics")
    return value if isinstance(value, str) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_git(path: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _run_git_bytes(
    path: Path,
    arguments: Sequence[str],
    *,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    if completed.returncode not in allowed_returncodes:
        return None
    return completed.stdout


def _display_git_path(value: bytes) -> str:
    return value.decode("utf-8", errors="backslashreplace")


def _git_pathspecs(
    root: Path, excluded_paths: Sequence[Path]
) -> tuple[list[str], list[str]]:
    pathspecs = ["."]
    excluded: list[str] = []
    for path in excluded_paths:
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue
        if relative == Path("."):
            continue
        display = relative.as_posix()
        pathspecs.append(f":(exclude,top,literal){display}")
        excluded.append(display)
    return pathspecs, excluded


def _status_has_unmerged_entries(status: bytes) -> bool:
    for record in status.split(b"\0"):
        if len(record) >= 3 and b"U" in record[:2]:
            return True
    return False


def _append_patch_chunk(
    chunks: list[bytes], chunk: bytes, *, current_size: int
) -> tuple[int, bool]:
    if not chunk:
        return current_size, True
    separator_size = int(bool(chunks and not chunks[-1].endswith(b"\n")))
    if current_size + separator_size + len(chunk) > _MAX_PATCH_BYTES:
        return current_size, False
    if separator_size:
        chunks.append(b"\n")
        current_size += 1
    chunks.append(chunk)
    return current_size + len(chunk), True


def _capture_git_patch(
    root: Path,
    *,
    destination: Path,
    reference: str,
    status_before: bytes,
    pathspecs: Sequence[str],
    excluded_generated_paths: Sequence[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    chunks: list[bytes] = []
    patch_size = 0

    tracked_patch = _run_git_bytes(
        root,
        [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            "HEAD",
            "--",
            *pathspecs,
        ],
    )
    tracked_included = tracked_patch is not None
    if tracked_patch is None:
        reasons.append("could not create the tracked-file Git diff")
    elif b"Subproject commit " in tracked_patch:
        reasons.append("submodule worktree or gitlink changes are not self-contained")
        patch_size, tracked_included = _append_patch_chunk(
            chunks, tracked_patch, current_size=patch_size
        )
    else:
        patch_size, tracked_included = _append_patch_chunk(
            chunks, tracked_patch, current_size=patch_size
        )
    if tracked_patch is not None and not tracked_included:
        reasons.append(
            f"tracked-file diff exceeds the {_MAX_PATCH_BYTES}-byte patch limit"
        )

    untracked_output = _run_git_bytes(
        root,
        [
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *pathspecs,
        ],
    )
    raw_paths = (
        sorted(path for path in untracked_output.split(b"\0") if path)
        if untracked_output is not None
        else []
    )
    paths_sha256 = hashlib.sha256(
        b"\0".join(raw_paths) + (b"\0" if raw_paths else b"")
    ).hexdigest()
    if untracked_output is None:
        reasons.append("could not enumerate untracked files")

    included_paths: list[str] = []
    omitted_paths: list[dict[str, str]] = []
    inspected_bytes = 0
    included_bytes = 0
    all_sizes_known = untracked_output is not None
    eligible_paths: list[tuple[bytes, Path, int]] = []

    if len(raw_paths) > _MAX_UNTRACKED_FILES:
        reasons.append(
            f"{len(raw_paths)} untracked files exceed the "
            f"{_MAX_UNTRACKED_FILES}-file capture limit"
        )

    for index, raw_path in enumerate(raw_paths):
        display = _display_git_path(raw_path)
        if index >= _MAX_UNTRACKED_FILES:
            if len(omitted_paths) < _MAX_OMITTED_PATH_EXAMPLES:
                omitted_paths.append({"path": display, "reason": "file-count limit"})
            all_sizes_known = False
            continue

        filesystem_path = root / os.fsdecode(raw_path)
        try:
            file_stat = filesystem_path.lstat()
        except OSError as error:
            reasons.append(f"cannot inspect untracked path {display!r}: {error}")
            if len(omitted_paths) < _MAX_OMITTED_PATH_EXAMPLES:
                omitted_paths.append({"path": display, "reason": "stat failed"})
            all_sizes_known = False
            continue

        if stat.S_ISREG(file_stat.st_mode):
            size = file_stat.st_size
        elif stat.S_ISLNK(file_stat.st_mode):
            try:
                size = len(os.fsencode(os.readlink(filesystem_path)))
            except OSError as error:
                reasons.append(f"cannot inspect untracked symlink {display!r}: {error}")
                if len(omitted_paths) < _MAX_OMITTED_PATH_EXAMPLES:
                    omitted_paths.append({"path": display, "reason": "readlink failed"})
                all_sizes_known = False
                continue
        else:
            reasons.append(
                f"untracked path {display!r} is not a regular file or symlink"
            )
            if len(omitted_paths) < _MAX_OMITTED_PATH_EXAMPLES:
                omitted_paths.append(
                    {"path": display, "reason": "unsupported file type"}
                )
            all_sizes_known = False
            continue

        inspected_bytes += size
        if inspected_bytes > _MAX_UNTRACKED_BYTES:
            if not any("untracked content exceeds" in reason for reason in reasons):
                reasons.append(
                    "untracked content exceeds the "
                    f"{_MAX_UNTRACKED_BYTES}-byte capture limit"
                )
            if len(omitted_paths) < _MAX_OMITTED_PATH_EXAMPLES:
                omitted_paths.append({"path": display, "reason": "content-size limit"})
            continue
        eligible_paths.append((raw_path, filesystem_path, size))

    for raw_path, filesystem_path, size in eligible_paths:
        display = _display_git_path(raw_path)
        before = filesystem_path.lstat()
        untracked_patch = _run_git_bytes(
            root,
            [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--binary",
                "--full-index",
                "--no-index",
                "--",
                "/dev/null",
                os.fsdecode(raw_path),
            ],
            allowed_returncodes=(0, 1),
        )
        try:
            after = filesystem_path.lstat()
        except OSError:
            after = None
        stable = (
            after is not None
            and before.st_mode == after.st_mode
            and before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
        )
        if untracked_patch is None:
            reasons.append(f"could not create a Git diff for {display!r}")
            if len(omitted_paths) < _MAX_OMITTED_PATH_EXAMPLES:
                omitted_paths.append({"path": display, "reason": "git diff failed"})
            continue
        if not stable:
            reasons.append(f"untracked path {display!r} changed while it was captured")
        patch_size, included = _append_patch_chunk(
            chunks, untracked_patch, current_size=patch_size
        )
        if not included:
            reasons.append(
                f"patch content for {display!r} exceeds the "
                f"{_MAX_PATCH_BYTES}-byte patch limit"
            )
            if len(omitted_paths) < _MAX_OMITTED_PATH_EXAMPLES:
                omitted_paths.append({"path": display, "reason": "patch-size limit"})
            continue
        included_paths.append(display)
        included_bytes += size

    if _status_has_unmerged_entries(status_before):
        reasons.append("unmerged index entries cannot be represented reproducibly")

    status_after = _run_git_bytes(
        root,
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
            "--",
            *pathspecs,
        ],
    )
    status_stable = status_after == status_before
    if status_after is None:
        reasons.append("could not verify worktree status after patch capture")
    elif not status_stable:
        reasons.append("worktree status changed while provenance was captured")

    patch = b"".join(chunks)
    _atomic_write_bytes(destination, patch)
    patch_sha256 = hashlib.sha256(patch).hexdigest()
    complete = (
        not reasons
        and tracked_included
        and untracked_output is not None
        and len(included_paths) == len(raw_paths)
        and status_stable
    )
    return {
        "path": reference,
        "sha256": patch_sha256,
        "size_bytes": len(patch),
        "format": "git-diff-binary",
        "complete": complete,
        "tracked_changes_included": tracked_included,
        "untracked": {
            "count": len(raw_paths),
            "included_count": len(included_paths),
            "included_bytes": included_bytes,
            "inspected_bytes": inspected_bytes if all_sizes_known else None,
            "paths_sha256": paths_sha256,
            "included_paths": included_paths,
            "omitted_count": len(raw_paths) - len(included_paths),
            "omitted_examples": omitted_paths,
        },
        "status": {
            "before_sha256": hashlib.sha256(status_before).hexdigest(),
            "after_sha256": (
                hashlib.sha256(status_after).hexdigest()
                if status_after is not None
                else None
            ),
            "stable": status_stable,
        },
        "excluded_generated_paths": list(excluded_generated_paths),
        "incomplete_reasons": reasons,
    }


def _git_snapshot(
    path: Path,
    *,
    require_exact_root: bool,
    patch_destination: Path | None = None,
    patch_reference: str | None = None,
    excluded_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    requested = path.resolve()
    root_text = _run_git(requested, "rev-parse", "--show-toplevel")
    if root_text is None:
        return {
            "path": str(requested),
            "available": False,
            "source_kind": "unavailable",
            "commit": None,
            "dirty": None,
            "patch": None,
        }

    root = Path(root_text).resolve()
    if require_exact_root and root != requested:
        return {
            "path": str(requested),
            "available": False,
            "source_kind": "unavailable",
            "commit": None,
            "dirty": None,
            "patch": None,
            "reason": f"not an independent git repository (parent is {root})",
        }

    commit = _run_git(root, "rev-parse", "HEAD")
    pathspecs, excluded_generated_paths = _git_pathspecs(root, excluded_paths)
    status = _run_git_bytes(
        root,
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
            "--",
            *pathspecs,
        ],
    )
    snapshot: dict[str, Any] = {
        "path": str(root),
        "available": commit is not None and status is not None,
        "source_kind": "git",
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "patch": None,
    }
    if excluded_generated_paths:
        snapshot["excluded_generated_paths"] = excluded_generated_paths
    if commit is None:
        snapshot["reason"] = "repository has no resolvable HEAD commit"
        return snapshot
    if status is None or not status:
        return snapshot
    if patch_destination is None or patch_reference is None:
        snapshot["reason"] = "dirty worktree patch destination was not provided"
        return snapshot
    snapshot["patch"] = _capture_git_patch(
        root,
        destination=patch_destination,
        reference=patch_reference,
        status_before=status,
        pathspecs=pathspecs,
        excluded_generated_paths=excluded_generated_paths,
    )
    return snapshot


def _is_generated_console_script(record_path: str) -> bool:
    remaining = record_path
    traversed_parent = False
    while remaining.startswith("../"):
        traversed_parent = True
        remaining = remaining[3:]
    first_component = remaining.split("/", maxsplit=1)[0].lower()
    return traversed_parent and first_component in {"bin", "scripts"}


def _decode_record_sha256(value: str) -> bytes:
    algorithm, separator, encoded = value.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        raise ValueError(f"unsupported RECORD hash: {value!r}")
    try:
        digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (binascii.Error, ValueError, TypeError) as error:
        raise ValueError(f"invalid RECORD sha256: {value!r}") from error
    if len(digest) != hashlib.sha256().digest_size:
        raise ValueError(f"invalid RECORD sha256: {value!r}")
    return digest


def _normalized_distribution_record(
    distribution: importlib.metadata.Distribution,
) -> dict[str, Any]:
    record_text = distribution.read_text("RECORD")
    files = distribution.files
    if record_text is None or files is None:
        raise ValueError("installed distribution has no readable RECORD")

    record_candidates = [
        str(path).replace("\\", "/")
        for path in files
        if str(path).replace("\\", "/").endswith(".dist-info/RECORD")
    ]
    if len(record_candidates) != 1:
        raise ValueError("installed distribution does not identify exactly one RECORD")
    record_relative_path = record_candidates[0]
    dist_info_directory = record_relative_path.rsplit("/", maxsplit=1)[0]
    record_path = Path(distribution.locate_file(record_relative_path)).resolve()

    included: list[tuple[str, str, int]] = []
    excluded_counts: dict[str, int] = {}
    verification_errors: list[str] = []
    seen: set[str] = set()
    owns_runtime_code = False
    reader = csv.reader(io.StringIO(record_text, newline=""))
    for line_number, row in enumerate(reader, start=1):
        if len(row) != 3:
            verification_errors.append(
                f"RECORD row {line_number} has {len(row)} fields"
            )
            continue
        raw_path, declared_hash, declared_size = row
        normalized_path = raw_path.replace("\\", "/")
        if not normalized_path or normalized_path in seen:
            verification_errors.append(
                f"RECORD row {line_number} has an empty or duplicate path"
            )
            continue
        seen.add(normalized_path)

        relative_to_dist_info = (
            normalized_path.removeprefix(f"{dist_info_directory}/")
            if normalized_path.startswith(f"{dist_info_directory}/")
            else None
        )
        exclusion: str | None = None
        verify_before_excluding = False
        if "__pycache__/" in normalized_path and normalized_path.endswith(".pyc"):
            exclusion = "generated_bytecode"
        elif _is_generated_console_script(normalized_path):
            exclusion = "generated_console_script"
            verify_before_excluding = True
        elif relative_to_dist_info in _DISTRIBUTION_METADATA_EXCLUSIONS:
            exclusion = "installer_metadata"
        if exclusion is not None:
            excluded_counts[exclusion] = excluded_counts.get(exclusion, 0) + 1
            if not verify_before_excluding:
                continue
        if not declared_hash or not declared_size:
            verification_errors.append(
                f"RECORD entry {normalized_path!r} has no hash or size"
            )
            continue

        try:
            declared_digest = _decode_record_sha256(declared_hash)
            size = int(declared_size)
        except ValueError as error:
            verification_errors.append(f"RECORD entry {normalized_path!r}: {error}")
            continue
        if size < 0:
            verification_errors.append(
                f"RECORD entry {normalized_path!r} has a negative size"
            )
            continue

        installed_path = Path(distribution.locate_file(normalized_path)).resolve()
        if normalized_path == "gorl/artifacts.py":
            owns_runtime_code = installed_path == Path(__file__).resolve()
        try:
            payload = installed_path.read_bytes()
        except OSError as error:
            verification_errors.append(
                f"cannot read installed RECORD entry {normalized_path!r}: {error}"
            )
            continue
        actual_digest = hashlib.sha256(payload).digest()
        if len(payload) != size or actual_digest != declared_digest:
            verification_errors.append(
                f"installed RECORD entry {normalized_path!r} does not match "
                "its declared hash and size"
            )
        if exclusion is None:
            included.append((normalized_path, declared_digest.hex(), size))

    if not owns_runtime_code:
        verification_errors.append(
            "distribution RECORD does not own the imported gorl/artifacts.py"
        )
    if not included:
        verification_errors.append("distribution RECORD has no verifiable files")

    canonical = hashlib.sha256()
    canonical.update(f"{_DISTRIBUTION_RECORD_FORMAT}\n".encode())
    for path, digest, size in sorted(included):
        canonical.update(path.encode("utf-8"))
        canonical.update(b"\0")
        canonical.update(digest.encode("ascii"))
        canonical.update(b"\0")
        canonical.update(str(size).encode("ascii"))
        canonical.update(b"\n")

    return {
        "record_path": str(record_path),
        "record_sha256": canonical.hexdigest(),
        "record_digest_format": _DISTRIBUTION_RECORD_FORMAT,
        "record_file_count": len(included),
        "record_excluded_count": sum(excluded_counts.values()),
        "record_excluded_by_reason": dict(sorted(excluded_counts.items())),
        "record_verified": not verification_errors,
        "verification_errors": verification_errors[:_MAX_OMITTED_PATH_EXAMPLES],
    }


def _distribution_snapshot() -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        return {
            "path": str(Path(__file__).resolve().parent),
            "available": False,
            "source_kind": "unavailable",
            "commit": None,
            "dirty": None,
            "patch": None,
            "reason": "gorl distribution metadata is not installed",
        }

    name = distribution.metadata.get("Name")
    version = distribution.version
    try:
        record = _normalized_distribution_record(distribution)
    except (OSError, ValueError) as error:
        record = {
            "record_path": None,
            "record_sha256": None,
            "record_digest_format": _DISTRIBUTION_RECORD_FORMAT,
            "record_file_count": 0,
            "record_excluded_count": 0,
            "record_excluded_by_reason": {},
            "record_verified": False,
            "verification_errors": [str(error)],
        }
    return {
        "path": str(Path(__file__).resolve().parent),
        "available": isinstance(name, str) and bool(name) and bool(version),
        "source_kind": "distribution",
        "commit": None,
        "dirty": None,
        "patch": None,
        "distribution": {
            "name": name,
            "version": version,
            **record,
        },
    }


def _code_snapshot(
    project_root: Path,
    *,
    patch_destination: Path,
    patch_reference: str,
    excluded_paths: Sequence[Path],
) -> dict[str, Any]:
    git_snapshot = _git_snapshot(
        project_root,
        require_exact_root=True,
        patch_destination=patch_destination,
        patch_reference=patch_reference,
        excluded_paths=excluded_paths,
    )
    if git_snapshot.get("source_kind") == "git":
        return git_snapshot

    distribution_snapshot = _distribution_snapshot()
    if distribution_snapshot.get("source_kind") == "distribution":
        return distribution_snapshot
    git_reason = git_snapshot.get("reason")
    distribution_reason = distribution_snapshot.get("reason")
    reasons = [
        reason
        for reason in (git_reason, distribution_reason)
        if isinstance(reason, str) and reason
    ]
    if reasons:
        git_snapshot["reason"] = "; ".join(reasons)
    return git_snapshot


def _package_status() -> dict[str, dict[str, str | None]]:
    packages: dict[str, dict[str, str | None]] = {}
    for name in _RUNTIME_PACKAGES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"status": "missing", "version": None}
        else:
            packages[name] = {"status": "installed", "version": version}
    return packages


def _package_versions(
    package_status: Mapping[str, Mapping[str, str | None]] | None = None,
) -> dict[str, str]:
    status = _package_status() if package_status is None else package_status
    return {
        name: version
        for name, record in status.items()
        if record.get("status") == "installed"
        and isinstance((version := record.get("version")), str)
    }


def _runtime_environment() -> dict[str, Any]:
    package_status = _package_status()
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": _package_versions(package_status),
        "package_status": package_status,
        "variables": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING",
                "GORL_DPPO_ROOT",
                "GORL_FPO_ROOT",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "JAX_PLATFORM_NAME",
                "JAX_PLATFORMS",
                "MUJOCO_GL",
                "MUJOCO_PY_MUJOCO_PATH",
                "PYTHONPATH",
                "PYTHONNOUSERSITE",
                "TF_XLA_FLAGS",
                "TORCH_USE_CUDA_DSA",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
    }


def _safe_artifact_name(value: Any, *, fallback: str) -> str:
    if value is None:
        return fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return (name or fallback)[:80]


def _dependency_environment_variable(
    dependency: Mapping[str, Any],
) -> str | None:
    name = dependency.get("name")
    if name == "fpo_dppo_fix":
        return "GORL_DPPO_ROOT"
    if name == "fpo":
        return "GORL_FPO_ROOT"
    return None


def _dependency_candidate_matches(path: Path, expected_commit: Any) -> bool:
    if not isinstance(expected_commit, str):
        return False
    snapshot = _git_snapshot(path, require_exact_root=True)
    return (
        snapshot.get("available") is True
        and snapshot.get("commit") == expected_commit
        and snapshot.get("dirty") is False
    )


def _dependency_snapshots(
    config: RunConfig,
    project_root: Path,
    runtime_root: Path,
    run_dir: Path,
) -> list[dict[str, Any]]:
    declared = config.values.get("upstream_dependencies", [])
    if not isinstance(declared, list):
        raise ArtifactError("upstream_dependencies must be an array of tables")

    snapshots: list[dict[str, Any]] = []
    for index, item in enumerate(declared):
        if not isinstance(item, Mapping):
            raise ArtifactError(f"upstream_dependencies[{index}] must be a table")
        dependency = dict(item)
        relative_path = dependency.get("relative_path")
        dependency_name = _safe_artifact_name(
            dependency.get("name"), fallback=f"dependency-{index}"
        )
        patch_reference = (
            Path("provenance") / f"dependency-{index:02d}-{dependency_name}.patch"
        )
        dependency_path: Path | None = None
        if isinstance(relative_path, str):
            environment_variable = _dependency_environment_variable(dependency)
            configured_root = (
                os.environ.get(environment_variable)
                if environment_variable is not None
                else None
            )
            if configured_root:
                dependency_path = Path(configured_root).expanduser()
                dependency["root_environment_variable"] = environment_variable
            else:
                candidates = [
                    project_root / relative_path,
                    runtime_root / relative_path,
                ]
                expected_commit = dependency.get("commit")
                dependency_path = next(
                    (
                        candidate
                        for candidate in candidates
                        if _dependency_candidate_matches(candidate, expected_commit)
                    ),
                    next(
                        (candidate for candidate in candidates if candidate.exists()),
                        candidates[0],
                    ),
                )
        actual = (
            _git_snapshot(
                dependency_path,
                require_exact_root=True,
                patch_destination=run_dir / patch_reference,
                patch_reference=patch_reference.as_posix(),
                excluded_paths=(run_dir,),
            )
            if dependency_path is not None
            else {
                "available": False,
                "commit": None,
                "dirty": None,
                "patch": None,
                "reason": "relative_path is missing or is not a string",
            }
        )
        expected_commit = dependency.get("commit")
        dependency["actual"] = actual
        dependency["matches_expected_commit"] = (
            actual["commit"] == expected_commit
            if expected_commit is not None and actual["commit"] is not None
            else False
        )
        snapshots.append(dependency)
    return snapshots


def _provenance_is_reproducible(
    code: Mapping[str, Any], dependencies: Sequence[Mapping[str, Any]]
) -> bool:
    def clean_or_complete_patch(snapshot: Mapping[str, Any]) -> bool:
        if snapshot.get("dirty") is False:
            return True
        patch = snapshot.get("patch")
        return (
            snapshot.get("dirty") is True
            and isinstance(patch, Mapping)
            and patch.get("complete") is True
            and isinstance(patch.get("path"), str)
            and bool(patch.get("path"))
            and isinstance(patch.get("sha256"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", patch.get("sha256", "")))
        )

    source_kind = code.get("source_kind")
    if source_kind == "distribution":
        distribution = code.get("distribution")
        code_ok = (
            code.get("available") is True
            and code.get("commit") is None
            and isinstance(distribution, Mapping)
            and isinstance(distribution.get("name"), str)
            and bool(distribution.get("name"))
            and isinstance(distribution.get("version"), str)
            and bool(distribution.get("version"))
            and distribution.get("record_verified") is True
            and distribution.get("record_digest_format") == _DISTRIBUTION_RECORD_FORMAT
            and isinstance(distribution.get("record_sha256"), str)
            and bool(
                re.fullmatch(r"[0-9a-f]{64}", distribution.get("record_sha256", ""))
            )
        )
    else:
        # source_kind was added as a backwards-compatible schema-1 extension.
        # A missing value therefore retains the original Git interpretation.
        code_ok = (
            source_kind in {None, "git"}
            and code.get("available") is True
            and code.get("commit") is not None
            and clean_or_complete_patch(code)
        )
    dependencies_ok = all(
        dependency.get("matches_expected_commit") is True
        and isinstance(dependency.get("actual"), Mapping)
        and dependency["actual"].get("source_kind") in {None, "git"}
        and clean_or_complete_patch(dependency["actual"])
        for dependency in dependencies
    )
    return code_ok and dependencies_ok


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    config: RunConfig
    _tracker: Tracker | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _tracking_finished: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.run_dir / "resolved_config.json"

    @property
    def stages_dir(self) -> Path:
        return self.run_dir / "stages"

    @classmethod
    def create(
        cls,
        config: RunConfig,
        *,
        argv: Sequence[str],
        cwd: str | Path,
        run_id: str | None = None,
    ) -> "RunArtifacts":
        selected_id = run_id or _default_run_id()
        if not _RUN_ID.fullmatch(selected_id):
            raise ArtifactError(
                "run_id must start with an alphanumeric character and contain "
                "only letters, numbers, '.', '_' or '-' (maximum 128 characters)"
            )
        run_dir = (
            config.output_root
            / config.task
            / config.method
            / f"seed-{config.seed}"
            / selected_id
        )
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ArtifactError(f"run directory already exists: {run_dir}") from error
        artifacts = cls(run_dir=run_dir, config=config)
        artifacts._initialize(
            run_id=selected_id,
            argv=list(argv),
            cwd=Path(cwd).resolve(),
        )
        return artifacts

    def _initialize(self, *, run_id: str, argv: list[str], cwd: Path) -> None:
        resolved = self.config.to_dict()
        atomic_write_json(self.resolved_config_path, resolved)
        config_sha256 = hashlib.sha256(
            self.resolved_config_path.read_bytes()
        ).hexdigest()

        project_root = self.config.config_root.parent
        code_patch_reference = Path("provenance") / "code.patch"
        code = _code_snapshot(
            project_root,
            patch_destination=self.run_dir / code_patch_reference,
            patch_reference=code_patch_reference.as_posix(),
            excluded_paths=(self.run_dir,),
        )
        dependencies = _dependency_snapshots(
            self.config,
            project_root,
            cwd,
            self.run_dir,
        )
        algorithm = {
            "name": self.config.method,
            "family": resolved.get("family"),
            "decoder": resolved.get("decoder", {}).get("type")
            if isinstance(resolved.get("decoder"), Mapping)
            else None,
            "ppo_profile": self.config.profile,
            "profile_ppo_backend": self.config.ppo_backend,
            "ppo_backend": (
                None
                if self.config.method in {"fpo", "dppo"}
                else self.config.ppo_backend
            ),
            "trainer_backend": trainer_backend(self.config),
            "action_semantics": _actual_action_semantics(resolved),
        }
        training = resolved.get("training", {})
        configured_total = (
            training.get("total_env_steps") if isinstance(training, Mapping) else None
        )
        schedule = benchmark_schedule_provenance(self.config)
        effective_stage_steps = list(self.config.stage_steps)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "initialized",
            "created_at": _utc_now(),
            "command": {
                "argv": argv,
                "shell": shlex.join(argv),
                "working_directory": str(cwd),
            },
            "task": self.config.task,
            "algorithm": algorithm,
            "seed": self.config.seed,
            "seeds": {
                "teacher": self.config.teacher_seed,
                "decoder": self.config.decoder_seed,
                "encoder": self.config.encoder_seed,
                "evaluation": self.config.eval_seed,
            },
            "environment": {
                "task": resolved.get("environment", {}),
                "runtime": _runtime_environment(),
            },
            "code": code,
            "upstream_dependencies": dependencies,
            "reproducible": _provenance_is_reproducible(code, dependencies),
            "config": {
                "path": self.resolved_config_path.name,
                "sha256": config_sha256,
                "values": resolved,
            },
            "timestep_budget": {
                "stage_env_steps": effective_stage_steps,
                "benchmark_total_env_steps": schedule["benchmark_end_env_steps"],
                **schedule,
                "training_total_env_steps": configured_total,
            },
            "final_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
        }
        atomic_write_json(self.manifest_path, manifest)

    def read_manifest(self) -> dict[str, Any]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ArtifactError(
                f"cannot read run manifest: {self.manifest_path}"
            ) from error
        if not isinstance(value, dict):
            raise ArtifactError(f"manifest root is not an object: {self.manifest_path}")
        return value

    def update_manifest(self, values: Mapping[str, Any]) -> None:
        manifest = self.read_manifest()
        manifest.update(dict(values))
        atomic_write_json(self.manifest_path, manifest)

    def mark_running(self) -> None:
        self.update_manifest({"status": "running", "started_at": _utc_now()})

    def mark_dry_run(self) -> None:
        self.update_manifest({"status": "dry_run", "completed_at": _utc_now()})

    def start_tracking(
        self,
        *,
        wandb: WandbConfig,
        wandb_loader: Callable[[], Any] | None = None,
    ) -> Tracker:
        """Create the run tracker while leaving terminal ownership to the CLI."""

        if self._tracker is not None:
            raise ArtifactError("tracking has already started for this run")
        if self._tracking_finished:
            raise ArtifactError("tracking has already finished for this run")
        from .tracking import Tracker

        tracker = Tracker(
            self.run_dir / "tracking",
            wandb=wandb,
            wandb_loader=wandb_loader,
        )
        object.__setattr__(self, "_tracker", tracker)
        return tracker

    def finish_tracking(self, *, exit_code: int) -> None:
        """Finish tracking after the terminal local manifest is durable."""

        if exit_code not in {0, 1}:
            raise ArtifactError("tracking exit_code must be 0 or 1")
        if self._tracking_finished:
            return
        tracker = self._tracker
        if tracker is None:
            object.__setattr__(self, "_tracking_finished", True)
            return
        tracker.finish(exit_code=exit_code)
        object.__setattr__(self, "_tracking_finished", True)

    def mark_failed(self, error: BaseException) -> None:
        error_values = {
            "type": type(error).__name__,
            "message": str(error) or type(error).__name__,
        }
        formal_metrics = {
            "final_return": None,
            "stage_best_return": None,
            "overall_best_return": None,
            "performance_target_met": None,
        }
        self.update_manifest(
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "error": error_values,
                **formal_metrics,
            }
        )
        summary_update_error: dict[str, str] | None = None
        summary_path = self.run_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(summary, dict):
                    raise ArtifactError("summary root is not an object")
                summary.update(
                    {
                        "status": "failed",
                        **formal_metrics,
                    }
                )
                if "deterministic_final_return" in summary:
                    summary["deterministic_final_return"] = None
                summary.setdefault(
                    "failure",
                    {
                        "type": "terminal_run_failure",
                        "error_type": error_values["type"],
                        "message": error_values["message"],
                    },
                )
                atomic_write_json(summary_path, summary)
            except Exception as summary_error:
                summary_update_error = {
                    "type": type(summary_error).__name__,
                    "message": str(summary_error) or type(summary_error).__name__,
                }
        if summary_update_error is not None:
            self.update_manifest({"summary_update_error": summary_update_error})

    def complete(self, result: Mapping[str, Any]) -> None:
        required = ("final_return", "stage_best_return", "overall_best_return")
        missing = [name for name in required if name not in result]
        if missing:
            raise ArtifactError(
                "pipeline result is missing metric(s): " + ", ".join(missing)
            )
        metrics: dict[str, float] = {}
        for name in required:
            value = result[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ArtifactError(f"{name} must be a finite number")
            metrics[name] = float(value)
        self.update_manifest(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                **metrics,
                "target_return": self.config.target_return,
                "performance_target_met": (
                    metrics["final_return"] >= self.config.target_return
                ),
            }
        )

    def write_stage_manifest(self, stage: int, values: Mapping[str, Any]) -> Path:
        if stage < 0 or stage >= len(self.config.stage_steps):
            raise ArtifactError(f"stage must be in [0, {len(self.config.stage_steps)})")
        path = self.stages_dir / f"stage-{stage}" / "manifest.json"
        atomic_write_json(path, dict(values))
        return path

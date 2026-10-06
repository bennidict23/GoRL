"""Lazy discovery of local training scripts without importing JAX stacks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class BackendUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EntrypointCandidate:
    path: Path
    working_directory: Path = Path(".")

    def __init__(
        self,
        path: str | Path,
        working_directory: str | Path = ".",
    ) -> None:
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(self, "working_directory", Path(working_directory))


@dataclass(frozen=True, slots=True)
class ResolvedEntrypoint:
    script: Path
    working_directory: Path
    source_root: Path


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    backend: str
    available: bool
    entrypoint: ResolvedEntrypoint | None
    reason: str
    checked_paths: tuple[Path, ...]

    def require(self) -> ResolvedEntrypoint:
        if self.entrypoint is None:
            raise BackendUnavailableError(self.reason)
        return self.entrypoint


class EntrypointResolver:
    def __init__(
        self,
        backend: str,
        candidates: Iterable[EntrypointCandidate],
        *,
        search_roots: Iterable[str | Path] | None = None,
    ) -> None:
        self.backend = backend
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise ValueError("at least one entrypoint candidate is required")
        roots = default_search_roots() if search_roots is None else search_roots
        self.search_roots = _unique_paths(Path(root) for root in roots)
        if not self.search_roots:
            raise ValueError("at least one search root is required")

    def availability(self) -> BackendAvailability:
        checked: list[Path] = []
        for root in self.search_roots:
            for candidate in self.candidates:
                script = _under(root, candidate.path)
                checked.append(script)
                if not script.is_file():
                    continue
                working_directory = _under(root, candidate.working_directory)
                if not working_directory.is_dir():
                    continue
                resolved = ResolvedEntrypoint(
                    script=script.resolve(),
                    working_directory=working_directory.resolve(),
                    source_root=root.resolve(),
                )
                return BackendAvailability(
                    backend=self.backend,
                    available=True,
                    entrypoint=resolved,
                    reason=f"{self.backend} entrypoint found at {resolved.script}",
                    checked_paths=tuple(checked),
                )
        rendered = "\n  - ".join(str(path) for path in checked)
        return BackendAvailability(
            backend=self.backend,
            available=False,
            entrypoint=None,
            reason=(
                f"{self.backend} is unavailable; no verified entrypoint exists. "
                f"Checked:\n  - {rendered}"
            ),
            checked_paths=tuple(checked),
        )


def default_search_roots() -> tuple[Path, ...]:
    project_root = Path(__file__).resolve().parents[3]
    package_root = Path(__file__).resolve().parents[2]
    installed_data_root = Path(sys.prefix) / "share" / "gorl"
    configured = [os.environ.get("GORL_REPOSITORY_ROOT")]
    roots = [Path(value).expanduser() for value in configured if value]
    roots.extend(
        (
            project_root,
            package_root,
            installed_data_root,
            Path.cwd(),
        )
    )
    return _unique_paths(roots)


def _under(root: Path, child: Path) -> Path:
    if child.is_absolute():
        return child
    return root / child


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.expanduser().resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique)

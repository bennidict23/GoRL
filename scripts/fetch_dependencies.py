#!/usr/bin/env python3
"""Fetch the exact FPO revisions used by the baseline workers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FPO_REPOSITORY = "https://github.com/akanazawa/fpo.git"
UPSTREAM_FPO_COMMIT = "418c2554f7cd22d52e14c07d951280929d73bf2f"
DPPO_FIX_COMMIT = "964dd78c6fb64de8c52eeb7ce80561c43455128f"
DPPO_FIX_AUTHOR_NAME = "GoRL Authors"
DPPO_FIX_AUTHOR_EMAIL = "gorl-authors@users.noreply.github.com"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DependencyError(RuntimeError):
    pass


class DirtyCheckoutError(DependencyError):
    pass


@dataclass(frozen=True, slots=True)
class Dependency:
    name: str
    directory_name: str
    commit: str
    description: str
    patch: "PatchRecipe | None" = None


@dataclass(frozen=True, slots=True)
class PatchRecipe:
    base_commit: str
    patch_path: Path
    author_name: str
    author_email: str
    timestamp: int
    timezone: str
    message: str


@dataclass(frozen=True, slots=True)
class FetchResult:
    name: str
    path: Path
    commit: str
    action: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": str(self.path),
            "commit": self.commit,
            "action": self.action,
        }


DEPENDENCIES = {
    "upstream": Dependency(
        name="upstream",
        directory_name="fpo",
        commit=UPSTREAM_FPO_COMMIT,
        description="Clean upstream PPO/FPO baseline revision.",
    ),
    "dppo": Dependency(
        name="dppo",
        directory_name="fpo_dppo_fix",
        commit=DPPO_FIX_COMMIT,
        description=(
            "Upstream revision plus the scalar-sigma denoising-MDP likelihood fix."
        ),
        patch=PatchRecipe(
            base_commit=UPSTREAM_FPO_COMMIT,
            patch_path=Path(__file__).resolve().parent
            / "patches"
            / "dppo_scalar_sigma.patch",
            author_name=DPPO_FIX_AUTHOR_NAME,
            author_email=DPPO_FIX_AUTHOR_EMAIL,
            timestamp=1779982084,
            timezone="+0800",
            message="fix scalar sigma in denoising MDP likelihood",
        ),
    ),
}


def default_destination_root() -> Path:
    """Use the clone when invoked from source, otherwise the current workspace."""

    if (PROJECT_ROOT / "pyproject.toml").is_file():
        return PROJECT_ROOT / "external"
    return Path.cwd() / "external"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone the exact clean FPO revisions required by GoRL baselines. "
            "Existing dirty checkouts are never modified."
        )
    )
    parser.add_argument(
        "--dependency",
        choices=("upstream", "dppo", "all"),
        default="all",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=default_destination_root(),
    )
    parser.add_argument(
        "--repository-url",
        default=FPO_REPOSITORY,
        help="FPO mirror URL; exact commit checks remain mandatory.",
    )
    return parser.parse_args(argv)


def git(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    input_text: str | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        input=input_text,
        env=environment,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def git_output(path: Path, *arguments: str) -> str:
    return git(("-C", str(path), *arguments), capture=True)


def checkout_status(path: Path) -> tuple[str, bool]:
    try:
        inside = git_output(path, "rev-parse", "--is-inside-work-tree")
    except subprocess.CalledProcessError as error:
        raise DependencyError(f"not a Git checkout: {path}") from error
    if inside != "true":
        raise DependencyError(f"not a Git worktree: {path}")
    checkout_root = Path(git_output(path, "rev-parse", "--show-toplevel")).resolve()
    if checkout_root != path.resolve():
        raise DependencyError(
            f"not an independent Git checkout: {path}; "
            f"belongs to {checkout_root}. Choose another destination directory."
        )
    commit = git_output(path, "rev-parse", "HEAD")
    clean = git_output(path, "status", "--porcelain") == ""
    return commit, clean


def ensure_checkout(
    dependency: Dependency,
    destination_root: Path,
    *,
    repository_url: str = FPO_REPOSITORY,
) -> FetchResult:
    destination_root = destination_root.expanduser().resolve()
    destination = destination_root / dependency.directory_name
    destination_root.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        commit, clean = checkout_status(destination)
        if not clean:
            raise DirtyCheckoutError(
                f"refusing to modify dirty dependency checkout: {destination}"
            )
        if commit == dependency.commit:
            return FetchResult(
                dependency.name,
                destination,
                dependency.commit,
                "reused",
            )
        _ensure_dependency_commit(destination, dependency)
        git(("-C", str(destination), "checkout", "--detach", dependency.commit))
        resolved_commit, resolved_clean = checkout_status(destination)
        if resolved_commit != dependency.commit or not resolved_clean:
            raise DependencyError(
                f"failed to checkout {dependency.commit} in {destination}"
            )
        return FetchResult(
            dependency.name,
            destination,
            dependency.commit,
            "updated",
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{dependency.directory_name}.",
            dir=destination_root,
        )
    )
    temporary_checkout = temporary_root / "checkout"
    try:
        git(("clone", "--no-checkout", repository_url, str(temporary_checkout)))
        _ensure_dependency_commit(temporary_checkout, dependency)
        git(
            (
                "-C",
                str(temporary_checkout),
                "checkout",
                "--detach",
                dependency.commit,
            )
        )
        resolved_commit, resolved_clean = checkout_status(temporary_checkout)
        if resolved_commit != dependency.commit or not resolved_clean:
            raise DependencyError(
                f"cloned checkout did not resolve cleanly to {dependency.commit}"
            )
        temporary_checkout.rename(destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    return FetchResult(
        dependency.name,
        destination,
        dependency.commit,
        "cloned",
    )


def _ensure_commit_available(path: Path, commit: str) -> None:
    try:
        git_output(path, "cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        git(("-C", str(path), "fetch", "origin", commit))
        try:
            git_output(path, "cat-file", "-e", f"{commit}^{{commit}}")
        except subprocess.CalledProcessError as error:
            raise DependencyError(
                f"repository does not contain required commit {commit}: {path}"
            ) from error


def _ensure_dependency_commit(path: Path, dependency: Dependency) -> None:
    if _commit_available(path, dependency.commit):
        return
    if dependency.patch is None:
        _ensure_commit_available(path, dependency.commit)
        return

    recipe = dependency.patch
    if not recipe.patch_path.is_file():
        raise DependencyError(f"missing dependency patch: {recipe.patch_path}")
    _ensure_commit_available(path, recipe.base_commit)
    environment = os.environ.copy()
    raw_date = f"@{recipe.timestamp} {recipe.timezone}"
    environment.update(
        {
            "GIT_AUTHOR_NAME": recipe.author_name,
            "GIT_AUTHOR_EMAIL": recipe.author_email,
            "GIT_AUTHOR_DATE": raw_date,
            "GIT_COMMITTER_NAME": recipe.author_name,
            "GIT_COMMITTER_EMAIL": recipe.author_email,
            "GIT_COMMITTER_DATE": raw_date,
        }
    )
    descriptor, index_name = tempfile.mkstemp(prefix=".gorl-dppo-index.")
    os.close(descriptor)
    index_path = Path(index_name)
    index_path.unlink()
    environment["GIT_INDEX_FILE"] = str(index_path)
    try:
        git(
            ("-C", str(path), "read-tree", recipe.base_commit),
            environment=environment,
        )
        git(
            (
                "-C",
                str(path),
                "apply",
                "--cached",
                str(recipe.patch_path.resolve()),
            ),
            environment=environment,
        )
        tree = git(
            ("-C", str(path), "write-tree"),
            capture=True,
            environment=environment,
        )
        generated = git(
            (
                "-C",
                str(path),
                "commit-tree",
                tree,
                "-p",
                recipe.base_commit,
            ),
            capture=True,
            input_text=recipe.message + "\n",
            environment=environment,
        )
    finally:
        index_path.unlink(missing_ok=True)
        index_path.with_suffix(index_path.suffix + ".lock").unlink(missing_ok=True)
    if generated != dependency.commit:
        raise DependencyError(
            "reconstructed patch commit mismatch: "
            f"expected {dependency.commit}, generated {generated}"
        )


def _commit_available(path: Path, commit: str) -> bool:
    try:
        git_output(path, "cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected = (
        tuple(DEPENDENCIES.values())
        if args.dependency == "all"
        else (DEPENDENCIES[args.dependency],)
    )
    try:
        results = [
            ensure_checkout(
                dependency,
                args.destination_root,
                repository_url=args.repository_url,
            )
            for dependency in selected
        ]
    except (DependencyError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            [result.to_dict() for result in results],
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the attested Diffusion consumer for a fresh Humanoid pair.

This private entry point accepts only a parent-owned request inside the pair
directory. It verifies the command, request, pair manifest, sealed prefix, and
resolved configuration before creating the branch run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gorl import cli
from gorl.humanoid.shared_prefix import load_shared_prefix


_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "pair_id",
        "task",
        "seed",
        "teacher_seed",
        "decoder_seed",
        "encoder_seed",
        "eval_seed",
        "profile",
        "config",
        "config_root",
        "output_root",
        "run_id",
        "wandb_mode",
        "wandb_project",
        "wandb_entity",
        "smoke",
        "pair_manifest",
        "shared_prefix_manifest",
        "expected_contract_sha256",
        "expected_shared_prefix_id",
    }
)


class SharedBranchWorkerError(RuntimeError):
    """Raised when a private pair request fails its fresh-run attestation."""


@dataclass(frozen=True, slots=True)
class AttestedRequest:
    request_path: Path
    request: Mapping[str, Any]
    pair_root: Path
    pair_manifest_path: Path
    pair_manifest: Mapping[str, Any]
    resolved_config_path: Path
    resolved_config: Mapping[str, Any]
    shared_prefix_manifest: Path
    shared_prefix_id: str
    contract_sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SharedBranchWorkerError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise SharedBranchWorkerError(f"{label} root must be an object")
    return value


def _request_mapping(path: Path) -> Mapping[str, Any]:
    value = _json_mapping(path, label="branch request")
    actual = set(value)
    if actual != _REQUEST_KEYS:
        missing = sorted(_REQUEST_KEYS.difference(actual))
        unknown = sorted(actual.difference(_REQUEST_KEYS))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise SharedBranchWorkerError("invalid branch request: " + "; ".join(details))
    if value["schema_version"] != 1:
        raise SharedBranchWorkerError("unsupported branch request schema")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SharedBranchWorkerError(f"{label} must be an object")
    return value


def _require(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise SharedBranchWorkerError(f"fresh-pair attestation mismatch: {label}")


def _resolve_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SharedBranchWorkerError(f"{label} must be a non-empty path")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SharedBranchWorkerError(f"cannot resolve {label}: {value}") from error
    if not path.is_file():
        raise SharedBranchWorkerError(f"{label} is not a file: {path}")
    return path


def _owned_file(value: Any, *, root: Path, expected: Path, label: str) -> Path:
    path = _resolve_file(value, label=label)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SharedBranchWorkerError(f"{label} escapes the fresh pair") from error
    if path != expected:
        raise SharedBranchWorkerError(f"{label} is not the canonical pair path")
    return path


def _namespace(request: Mapping[str, Any]) -> argparse.Namespace:
    task = request["task"]
    if task not in {"HumanoidStand", "HumanoidRun"}:
        raise SharedBranchWorkerError("private branch supports only Humanoid tasks")
    integer_fields = (
        "seed",
        "teacher_seed",
        "decoder_seed",
        "encoder_seed",
        "eval_seed",
    )
    for field in integer_fields:
        value = request[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SharedBranchWorkerError(
                f"branch request {field} must be non-negative"
            )
    if not isinstance(request["smoke"], bool):
        raise SharedBranchWorkerError("branch request smoke must be a boolean")
    return argparse.Namespace(
        command="train",
        task=task,
        method="gorl_diffusion",
        seed=request["seed"],
        teacher_seed=request["teacher_seed"],
        decoder_seed=request["decoder_seed"],
        encoder_seed=request["encoder_seed"],
        eval_seed=request["eval_seed"],
        profile=request["profile"],
        config=None if request["config"] is None else Path(request["config"]),
        config_root=Path(request["config_root"]),
        output_root=Path(request["output_root"]),
        run_id=request["run_id"],
        wandb_mode=request["wandb_mode"],
        wandb_project=request["wandb_project"],
        wandb_entity=request["wandb_entity"],
        smoke=request["smoke"],
        dry_run=False,
    )


def _attest_request(path: Path, *, command: Sequence[str]) -> AttestedRequest:
    request_path = path.expanduser().resolve(strict=True)
    request = _request_mapping(request_path)
    pair_manifest_path = _resolve_file(request["pair_manifest"], label="pair manifest")
    pair_root = pair_manifest_path.parent
    if pair_manifest_path != pair_root / "manifest.json":
        raise SharedBranchWorkerError("pair manifest is not at its canonical path")
    if (
        request_path.parent != pair_root
        or request_path.name != "diffusion_worker_request.json"
    ):
        raise SharedBranchWorkerError(
            "branch request is not owned by its pair directory"
        )
    pair_manifest = _json_mapping(pair_manifest_path, label="pair manifest")
    _require(pair_manifest.get("schema_version"), 1, label="pair schema")
    _require(
        pair_manifest.get("kind"),
        "fresh_humanoid_gorl_pair",
        label="pair kind",
    )
    _require(pair_manifest.get("status"), "running", label="pair status")
    _require(pair_manifest.get("fresh_from_scratch"), True, label="fresh declaration")
    _require(
        pair_manifest.get("accepts_external_shared_prefix"),
        False,
        label="external-prefix declaration",
    )
    _require(pair_manifest.get("pair_id"), request["pair_id"], label="pair ID")
    _require(pair_root.name, request["pair_id"], label="pair directory ID")

    request_record = _mapping(
        pair_manifest.get("diffusion_request"), label="diffusion request record"
    )
    _owned_file(
        request_record.get("path"),
        root=pair_root,
        expected=request_path,
        label="recorded branch request",
    )
    _require(request_record.get("sha256"), _sha256(request_path), label="request hash")

    _require(pair_manifest.get("task"), request["task"], label="task")
    _require(pair_manifest.get("profile"), request["profile"], label="profile")
    seeds = _mapping(pair_manifest.get("seeds"), label="pair seeds")
    _require(seeds.get("teacher"), request["teacher_seed"], label="teacher seed")
    _require(seeds.get("diffusion"), request["seed"], label="diffusion seed")
    for role in ("decoder_seed", "encoder_seed", "eval_seed"):
        _require(request[role], request["seed"], label=role)

    inputs = _mapping(pair_manifest.get("inputs"), label="pair inputs")
    _require(inputs.get("smoke"), request["smoke"], label="smoke mode")
    _require(inputs.get("config_root"), request["config_root"], label="config root")
    _require(inputs.get("overlay"), request["config"], label="config overlay")
    expected_output_root = pair_root / "branches"
    _require(
        Path(request["output_root"]).resolve(),
        expected_output_root,
        label="branch output root",
    )
    _require(
        inputs.get("branch_output_root"),
        str(expected_output_root),
        label="recorded branch output root",
    )
    _require(request["run_id"], "diffusion", label="diffusion run ID")
    wandb = _mapping(inputs.get("wandb"), label="pair W&B inputs")
    _require(wandb.get("mode"), request["wandb_mode"], label="W&B mode")
    _require(wandb.get("project"), request["wandb_project"], label="W&B project")
    _require(wandb.get("entity"), request["wandb_entity"], label="W&B entity")

    contract = _mapping(pair_manifest.get("teacher_contract"), label="teacher contract")
    contract_values = _mapping(contract.get("values"), label="teacher contract values")
    contract_sha256 = _sha256_json(contract_values)
    _require(contract.get("sha256"), contract_sha256, label="teacher contract hash")
    _require(
        request["expected_contract_sha256"],
        contract_sha256,
        label="requested teacher contract hash",
    )

    shared = _mapping(pair_manifest.get("shared_prefix"), label="shared prefix")
    _require(shared.get("status"), "sealed", label="shared-prefix status")
    prefix_root = pair_root / "shared_prefix"
    _require(shared.get("root"), str(prefix_root), label="shared-prefix root")
    prefix_manifest = _owned_file(
        request["shared_prefix_manifest"],
        root=pair_root,
        expected=prefix_root / "shared_prefix.json",
        label="shared-prefix manifest",
    )
    _require(
        shared.get("manifest"), str(prefix_manifest), label="recorded prefix manifest"
    )
    shared_prefix_id = request["expected_shared_prefix_id"]
    if not isinstance(shared_prefix_id, str) or len(shared_prefix_id) != 64:
        raise SharedBranchWorkerError("expected shared-prefix ID is not sealed")
    _require(shared.get("shared_prefix_id"), shared_prefix_id, label="shared-prefix ID")
    _require(
        shared.get("expected_shared_prefix_id"),
        shared_prefix_id,
        label="expected shared-prefix ID",
    )
    _require(
        shared.get("contract_sha256"), contract_sha256, label="prefix contract hash"
    )

    branches = _mapping(pair_manifest.get("branches"), label="pair branches")
    fm = _mapping(branches.get("fm"), label="FM branch")
    diffusion = _mapping(branches.get("diffusion"), label="diffusion branch")
    _require(fm.get("role"), "fresh_prefix_producer", label="FM branch role")
    _require(fm.get("method"), "gorl_fm", label="FM branch method")
    _require(
        diffusion.get("role"),
        "private_shared_prefix_consumer",
        label="diffusion branch role",
    )
    _require(diffusion.get("method"), "gorl_diffusion", label="diffusion branch method")
    diffusion_command = _mapping(
        diffusion.get("command"), label="diffusion branch command"
    )
    _require(diffusion_command.get("argv"), list(command), label="worker invocation")
    expected_run_dir = (
        expected_output_root
        / request["task"]
        / "gorl_diffusion"
        / f"seed-{request['seed']}"
        / "diffusion"
    )
    _require(diffusion.get("run_dir"), str(expected_run_dir), label="diffusion run dir")

    resolved_configs = _mapping(
        pair_manifest.get("resolved_configs"), label="resolved configs"
    )
    resolved_record = _mapping(
        resolved_configs.get("diffusion"), label="diffusion resolved config"
    )
    resolved_config_path = _owned_file(
        str(pair_root / str(resolved_record.get("path"))),
        root=pair_root,
        expected=pair_root / "resolved_diffusion_config.json",
        label="diffusion resolved config",
    )
    _require(
        resolved_record.get("sha256"),
        _sha256(resolved_config_path),
        label="resolved config hash",
    )
    resolved_config = _json_mapping(
        resolved_config_path, label="diffusion resolved config"
    )
    return AttestedRequest(
        request_path=request_path,
        request=request,
        pair_root=pair_root,
        pair_manifest_path=pair_manifest_path,
        pair_manifest=pair_manifest,
        resolved_config_path=resolved_config_path,
        resolved_config=resolved_config,
        shared_prefix_manifest=prefix_manifest,
        shared_prefix_id=shared_prefix_id,
        contract_sha256=contract_sha256,
    )


def _verify_loaded_config(config: Any, attested: AttestedRequest) -> None:
    _require(config.to_dict(), attested.resolved_config, label="loaded resolved config")
    request = attested.request
    expected = {
        "task": request["task"],
        "method": "gorl_diffusion",
        "seed": request["seed"],
        "teacher_seed": request["teacher_seed"],
        "decoder_seed": request["decoder_seed"],
        "encoder_seed": request["encoder_seed"],
        "eval_seed": request["eval_seed"],
        "profile": request["profile"],
        "smoke": request["smoke"],
    }
    for field, value in expected.items():
        _require(getattr(config, field), value, label=f"loaded config {field}")


def execute_request(path: Path, *, command: Sequence[str]) -> int:
    attested = _attest_request(path, command=command)
    request = attested.request
    args = _namespace(request)

    def run(config: Any, artifacts: Any) -> Mapping[str, Any] | None:
        _verify_loaded_config(config, attested)
        bundle = load_shared_prefix(attested.shared_prefix_manifest)
        _require(
            bundle.shared_prefix_id,
            attested.shared_prefix_id,
            label="loaded shared-prefix ID",
        )
        _require(
            bundle.contract_sha256,
            attested.contract_sha256,
            label="loaded shared-prefix contract",
        )
        artifacts.update_manifest(
            {
                "branch_execution": {
                    "mode": "private_shared_prefix_consumer",
                    "ordinary_fresh_train": False,
                    "pair_fresh_from_scratch": True,
                    "attestation": "verified",
                    "pair_id": request["pair_id"],
                    "pair_manifest": str(attested.pair_manifest_path),
                    "request_path": str(attested.request_path),
                    "request_sha256": _sha256(attested.request_path),
                    "shared_prefix_manifest": str(bundle.manifest_path),
                    "shared_prefix_id": bundle.shared_prefix_id,
                    "contract_sha256": bundle.contract_sha256,
                }
            }
        )
        from gorl.humanoid.pipeline import run as run_humanoid

        return run_humanoid(config, artifacts, shared_prefix=bundle)

    return cli._train(
        args,
        argv=("train",),
        pipeline_runner=run,
        command_argv=command,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    command = [
        sys.executable,
        "-m",
        "gorl._workers.humanoid_shared_branch",
        *arguments,
    ]
    return execute_request(args.request, command=command)


if __name__ == "__main__":
    raise SystemExit(main())

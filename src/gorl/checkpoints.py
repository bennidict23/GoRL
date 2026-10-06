"""Atomic, hash-addressed checkpoints for in-process GoRL training."""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_pickle_checkpoint(path: str | Path, payload: Any) -> dict[str, Any]:
    """Write one trusted local checkpoint atomically and return its metadata."""

    import jax

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(
                jax.device_get(payload),
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output.resolve()),
        "format": "pickle",
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
    }


def load_pickle_checkpoint(path: str | Path) -> Any:
    """Load a checkpoint created by this package.

    Pickle is executable data. Only load checkpoints from a trusted source.
    """

    with Path(path).open("rb") as stream:
        return pickle.load(stream)

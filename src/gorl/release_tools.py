"""Console dispatch for release tools shipped as wheel data files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class ReleaseToolError(RuntimeError):
    pass


def _tool_path(name: str) -> Path:
    source_root = Path(__file__).resolve().parents[2]
    candidates = (
        source_root / "scripts" / f"{name}.py",
        Path(sys.prefix) / "share" / "gorl" / "tools" / f"{name}.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = "\n  - ".join(str(path) for path in candidates)
    raise ReleaseToolError(
        f"GoRL release tool {name!r} is not installed; checked:\n  - {rendered}"
    )


def _load_tool(name: str) -> ModuleType:
    path = _tool_path(name)
    module_name = f"gorl._release_tool_{name}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ReleaseToolError(f"cannot load GoRL release tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _run(name: str) -> int:
    module = _load_tool(name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise ReleaseToolError(f"GoRL release tool has no main function: {name}")
    result = main()
    return 0 if result is None else int(result)


def fetch_dependencies_main() -> int:
    return _run("fetch_dependencies")

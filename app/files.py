"""Validate relative paths and publish outputs only after a complete write."""
import json
import os
import tempfile
from pathlib import Path, PurePosixPath


def safe_path(root, relative):
    # Reject both POSIX and Windows traversal, including on a Linux server.
    raw = str(relative)
    if not raw or "\\" in raw or ":" in raw or "\x00" in raw:
        raise ValueError("Use a relative path without drive letters or backslashes")
    rel = PurePosixPath(raw)
    if rel.is_absolute() or any(p in {"..", "."} or p.startswith(".") for p in raw.split("/")):
        raise ValueError("Hidden files and unsafe relative paths are not allowed")
    root = Path(root).resolve()
    target = root.joinpath(*rel.parts).resolve()
    if target == root or root not in target.parents:
        raise ValueError("Path is outside the allowed folder")
    return target


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".writing-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def list_files(settings, group):
    from .config import FORMATS
    items = []
    root = settings.root / group
    for category, extensions in FORMATS.items():
        for path in sorted((root / category).rglob("*")):
            relative = path.relative_to(root).as_posix()
            if not path.is_file() or path.is_symlink() or any(p.startswith(".") for p in path.relative_to(root).parts):
                continue
            try:
                safe_path(root, relative)
            except ValueError:
                continue
            if group == "input" and path.suffix.lower() not in extensions:
                continue
            if group == "output" and not path.name.endswith(".txt"):
                continue
            items.append({"path": relative, "category": category, "bytes": path.stat().st_size})
    return items

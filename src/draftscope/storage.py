"""Exclusive run folders and atomic result writes."""

from pathlib import Path
import hashlib
import json
import os
import tempfile


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(raw + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def create_run(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(root, manifest):
    root = Path(root).resolve()
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError(f"Unsafe manifest path: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(manifest)

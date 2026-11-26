"""Shared artifact serialization and file identity plumbing."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def write_json(path: Path, value) -> None:
    """Atomically replace a JSON artifact after writing the complete contents."""
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.",
                                       suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

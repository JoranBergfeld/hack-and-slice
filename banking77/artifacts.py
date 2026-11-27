"""Shared artifact serialization and file identity plumbing."""
import errno
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import tempfile
import time


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


@contextmanager
def launch_lock(output: Path):
    """Serialize readers and writers that update one Azure launch manifest."""
    lock_path = output / ".launch.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(0.1)

            def unlock():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            unlock()

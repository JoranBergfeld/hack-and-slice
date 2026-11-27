"""Temporarily load the verified implementation saved with a training launch."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys


@contextmanager
def captured_modules(source: Path, names: tuple[str, ...]):
    package = importlib.import_module("banking77")
    previous_path = list(package.__path__)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_modules = {}
    package.__path__ = [str((source / "banking77").resolve()), *previous_path]
    sys.dont_write_bytecode = True
    try:
        for name in names:
            module_name = f"banking77.{name}"
            previous_modules[module_name] = sys.modules.pop(module_name, None)
        modules = {
            name: importlib.import_module(f"banking77.{name}")
            for name in names
        }
        yield modules
    finally:
        for name in reversed(names):
            module_name = f"banking77.{name}"
            sys.modules.pop(module_name, None)
            previous = previous_modules.get(module_name)
            if previous is not None:
                sys.modules[module_name] = previous
        package.__path__ = previous_path
        sys.dont_write_bytecode = previous_dont_write_bytecode

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
    missing = object()
    previous_attributes = {
        name: getattr(package, name, missing) for name in names
    }
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
            attribute = previous_attributes[name]
            if attribute is missing:
                package.__dict__.pop(name, None)
            else:
                setattr(package, name, attribute)
        package.__path__ = previous_path
        sys.dont_write_bytecode = previous_dont_write_bytecode

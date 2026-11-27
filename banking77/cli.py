"""Consistent actionable failures for provided participant commands."""
from functools import wraps
import pickle
import sys


CLI_ERRORS = (
    NotImplementedError, AssertionError, ValueError, OSError, RuntimeError,
    KeyError, AttributeError, TypeError, IndexError, ZeroDivisionError,
    ImportError, EOFError, pickle.UnpicklingError,
)


def cli_errors(action: str, challenge: str):
    """Report expected command failures without hiding unexpected exceptions."""
    def decorate(command):
        @wraps(command)
        def guarded(*args, **kwargs):
            try:
                return command(*args, **kwargs)
            except CLI_ERRORS as error:
                print(f"{action} failed: {error}. See docs/challenges/{challenge}.",
                      file=sys.stderr, flush=True)
                return 1
        return guarded
    return decorate

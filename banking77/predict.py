"""Provided CLI for separate saved-classifier demonstration prediction."""
import argparse
from pathlib import Path
import pickle
import sys

from banking77.checkpoint import load_classifier
from banking77.prediction import predict_intent


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a demonstration intent on local CPU.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--message", required=True)
    arguments = parser.parse_args()
    try:
        saved = load_classifier(arguments.model, device="cpu")
        intent = predict_intent(saved, arguments.message)
        if not isinstance(intent, str) or intent not in saved.label_mapping:
            raise ValueError("predict_intent must return an original saved intent name")
    except (NotImplementedError, AssertionError, ValueError, OSError, RuntimeError,
            KeyError, AttributeError, TypeError, IndexError, ZeroDivisionError,
            ImportError, EOFError, pickle.UnpicklingError) as error:
        print(f"Prediction failed: {error}. "
              "See docs/challenges/05-use-classifier.md.", file=sys.stderr)
        return 1
    print(f"Demonstration prediction on local CPU\nPredicted intent: {intent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

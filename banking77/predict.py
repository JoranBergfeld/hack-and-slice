"""Provided CLI for separate saved-classifier demonstration prediction."""
import argparse
from pathlib import Path

from banking77.checkpoint import load_classifier
from banking77.prediction import predict_intent
from banking77.cli import cli_errors


@cli_errors("Prediction", "05-use-classifier.md")
def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a demonstration intent on local CPU.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--message", required=True)
    arguments = parser.parse_args()
    saved = load_classifier(arguments.model, device="cpu")
    intent = predict_intent(saved, arguments.message)
    if not isinstance(intent, str) or intent not in saved.label_mapping:
        raise ValueError("predict_intent must return an original saved intent name")
    print(f"Demonstration prediction on local CPU\nPredicted intent: {intent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

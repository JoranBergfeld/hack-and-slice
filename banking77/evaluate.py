"""Provided selection-gated local CPU evaluation plumbing."""
import argparse
import json
import math
from pathlib import Path
import pickle
import sys

import torch

from banking77.checkpoint import SavedClassifier, load_classifier
from banking77.compare import validate_results
from banking77.data import load_test, read_messages
from banking77.evaluation import accuracy, macro_f1
from banking77.training import build_loader


def selected_classifier(launch: Path) -> tuple[dict, dict, SavedClassifier]:
    """Validate the whole launch and its selected winner before reading test data."""
    selection_path = launch / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"{selection_path} is missing; run compare select first")
    records = validate_results(launch)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    index = selection.get("configuration_index")
    if type(index) is not int or not 0 <= index < len(records):
        raise ValueError("selection.json: invalid configuration index")
    winner = max(range(len(records)),
                 key=lambda position: records[position]["best_validation_accuracy"])
    expected = {
        "launch_id": records[index]["provenance"]["launch_id"],
        "configuration_index": index, "run": f"runs/{index}",
        "best_validation_accuracy": records[index]["best_validation_accuracy"],
    }
    if selection != expected or index != winner:
        raise ValueError("selection.json does not match this launch's validation winner")
    saved = load_classifier(launch / "runs" / str(index), device="cpu")
    return selection, records[index], saved


def evaluate(launch: Path, data_dir: Path, *, download: bool = False,
             test_csv: Path | None = None) -> dict:
    selection, record, saved = selected_classifier(launch)
    # Check learning functions on synthetic IDs before touching held-out data.
    from banking77.check_evaluation import check
    check()
    messages = (read_messages(test_csv, list(saved.label_mapping))
                if test_csv is not None else load_test(data_dir, download=download))
    if any(message.label not in saved.label_mapping for message in messages):
        raise ValueError("test labels do not match the saved label mapping")
    loader = build_loader(messages, saved.vocabulary, saved.label_mapping,
                          record["configuration"]["batch_size"], shuffle=False)
    targets, predictions = [], []
    saved.model.eval()
    with torch.no_grad():
        for tokens, offsets, expected in loader:
            logits = saved.model(tokens.to("cpu"), offsets.to("cpu"))
            if (logits.shape != (len(expected), len(saved.label_mapping))
                    or not torch.isfinite(logits).all()):
                raise ValueError("classifier must return finite logits for every intent")
            predictions.extend(logits.argmax(dim=1).tolist())
            targets.extend(expected.tolist())
    if len(targets) != len(messages):
        raise ValueError("evaluation loader must include every test message exactly once")
    scores = {"accuracy": accuracy(targets, predictions),
              "macro_f1": macro_f1(targets, predictions, len(saved.label_mapping))}
    if any(type(score) is not float or not math.isfinite(score) or not 0 <= score <= 1
           for score in scores.values()):
        raise ValueError("metrics must return finite floats in [0, 1]")
    report = {**selection, "device": "cpu",
              "test_source": "synthetic" if test_csv is not None else "official held-out",
              "message_count": len(messages), **scores}
    temporary = launch / "evaluation.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(launch / "evaluation.json")
    print(f"Final evaluation on local CPU: configuration "
          f"{selection['configuration_index'] + 1}, "
          f"{report['message_count']} {report['test_source']} messages")
    print(f"Accuracy: {scores['accuracy']:.4f}\nMacro-F1: {scores['macro_f1']:.4f}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the selected winner on local CPU.")
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--download", action="store_true")
    sources.add_argument("--test-csv", type=Path, help="explicit synthetic offline test CSV")
    arguments = parser.parse_args()
    try:
        evaluate(arguments.launch, arguments.data_dir, download=arguments.download,
                 test_csv=arguments.test_csv)
    except (NotImplementedError, AssertionError, ValueError, OSError, RuntimeError,
            KeyError, AttributeError, TypeError, IndexError, ZeroDivisionError,
            ImportError, EOFError, pickle.UnpicklingError) as error:
        print(f"Evaluation failed: {error}. "
              "See docs/challenges/04-evaluate-classifier.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

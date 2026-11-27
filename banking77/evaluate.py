"""Provided selection-gated local CPU evaluation plumbing."""
import argparse
import json
import math
from pathlib import Path

import torch

from banking77.captured import captured_modules
from banking77.compare import selection_record, validate_results
from banking77.evaluation import accuracy, macro_f1
from banking77.artifacts import write_json
from banking77.cli import cli_errors


def selected_classifier(launch: Path) -> tuple[dict, dict]:
    """Validate the whole launch and its selected winner before reading test data."""
    selection_path = launch / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"{selection_path} is missing; run compare select first")
    records = validate_results(launch)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    index = selection.get("configuration_index")
    if type(index) is not int or not 0 <= index < len(records):
        raise ValueError("selection.json: invalid configuration index")
    if selection != selection_record(records):
        raise ValueError("selection.json does not match this launch's validation winner; "
                         "check select_configuration, then re-run "
                         f"python -m banking77.compare select --launch \"{launch}\" "
                         "before evaluation")
    return selection, records[index]


def evaluate(launch: Path, data_dir: Path, *, download: bool = False,
             test_csv: Path | None = None) -> dict:
    selection, record = selected_classifier(launch)
    # Check learning functions on synthetic IDs before touching held-out data.
    from banking77.check_evaluation import check
    check()
    with captured_modules(
        launch / "source", ("data", "model", "training", "checkpoint")
    ) as captured:
        saved = captured["checkpoint"].load_classifier(
            launch / "runs" / str(selection["configuration_index"]), device="cpu")
        messages = (
            captured["data"].read_messages(test_csv, list(saved.label_mapping))
            if test_csv is not None
            else captured["data"].load_test(data_dir, download=download)
        )
        if any(message.label not in saved.label_mapping for message in messages):
            raise ValueError("test labels do not match the saved label mapping")
        loader = captured["training"].build_loader(
            messages, saved.vocabulary, saved.label_mapping,
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
    write_json(launch / "evaluation.json", report)
    print(f"Final evaluation on local CPU: configuration "
          f"{selection['configuration_index'] + 1}, "
          f"{report['message_count']} {report['test_source']} messages")
    print(f"Accuracy: {scores['accuracy']:.4f}\nMacro-F1: {scores['macro_f1']:.4f}")
    return report


@cli_errors("Evaluation", "04-evaluate-classifier.md")
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the selected winner on local CPU.")
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--download", action="store_true")
    # Non-participant seam for offline synthetic fixtures.
    sources.add_argument("--test-csv", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    evaluate(arguments.launch, arguments.data_dir, download=arguments.download,
             test_csv=arguments.test_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

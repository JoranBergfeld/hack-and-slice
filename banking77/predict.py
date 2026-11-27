"""Provided CLI for separate saved-classifier demonstration prediction."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

from banking77.captured import captured_modules
from banking77.cli import cli_errors


def _comparison_run(model_path: Path) -> tuple[Path, int] | None:
    resolved = model_path.resolve()
    if resolved.parent.name != "runs":
        return None
    launch = resolved.parent.parent
    if not (launch / "launch.json").is_file():
        return None
    try:
        index = int(resolved.name)
    except ValueError as error:
        raise ValueError(f"{model_path} is not a configuration run in its comparison") from error
    return launch, index


def _comparison_prediction(
    model_path: Path, message: str, launch: Path, index: int
):
    from banking77.compare import validate_results

    manifest = json.loads((launch / "launch.json").read_text(encoding="utf-8"))
    records = validate_results(launch)
    if not 0 <= index < len(records):
        raise ValueError(f"{model_path} is not a completed run in this comparison")
    selection_path = launch / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection != {
        "launch_id": manifest["launch_id"],
        "configuration_index": index,
        "run": f"runs/{index}",
        "best_validation_accuracy": records[index]["best_validation_accuracy"],
    }:
        raise ValueError(f"{model_path} is not the selected run; use the run in "
                         f"{selection_path}")

    canonical_run = launch / "runs" / str(index)
    with captured_modules(launch / "source", ("data", "model", "checkpoint")) as captured:
        saved = captured["checkpoint"].load_classifier(canonical_run, device="cpu")
        module_name = "banking77.prediction"
        previous = sys.modules.pop(module_name, None)
        try:
            specification = importlib.util.spec_from_file_location(
                module_name, Path(__file__).with_name("prediction.py"))
            if specification is None or specification.loader is None:
                raise ImportError("cannot load the current participant prediction implementation")
            participant_prediction = importlib.util.module_from_spec(specification)
            sys.modules[module_name] = participant_prediction
            specification.loader.exec_module(participant_prediction)
            intent = participant_prediction.predict_intent(saved, message)
        finally:
            sys.modules.pop(module_name, None)
            if previous is not None:
                sys.modules[module_name] = previous
    return saved, intent


@cli_errors("Prediction", "05-use-classifier.md")
def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a demonstration intent on local CPU.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--message", required=True)
    arguments = parser.parse_args()
    comparison = _comparison_run(arguments.model)
    if comparison is None:
        from banking77.checkpoint import load_classifier

        prediction = importlib.import_module("banking77.prediction")
        saved = load_classifier(arguments.model, device="cpu")
        intent = prediction.predict_intent(saved, arguments.message)
    else:
        saved, intent = _comparison_prediction(
            arguments.model, arguments.message, *comparison)
    if not isinstance(intent, str) or intent not in saved.label_mapping:
        raise ValueError("predict_intent must return an original saved intent name")
    print(f"Demonstration prediction on local CPU\nPredicted intent: {intent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

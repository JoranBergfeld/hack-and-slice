"""Provided local launch capture, comparison chart and result-validation plumbing."""
import argparse
from dataclasses import asdict
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import pickle
import os
import shutil
import subprocess
import sys
import uuid

import sklearn
import torch

from banking77.runner import PREPARED_FILES, SEED, TrainingConfiguration


RESULT_FILES = ("run.json", "best-model.pt", "vocabulary.json", "labels.json")


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(hashes: dict) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _software() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "scikit-learn": sklearn.__version__}


def _configurations(path: Path) -> list[dict]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("provide a list of at least two configurations")
    configurations = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("each configuration must be an object")
        configuration = TrainingConfiguration(**value)
        configuration.validate()
        if not math.isfinite(configuration.learning_rate):
            raise ValueError("learning rate must be finite")
        configurations.append(asdict(configuration))
    return configurations


def _chart(output: Path, records: list[dict]) -> None:
    height = 90 + 70 * len(records)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" '
        f'viewBox="0 0 1000 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#24292f">',
        '<text x="20" y="30">Best validation accuracy (0 to 1)</text>',
    ]
    for index, record in enumerate(records):
        y = 70 + index * 70
        score = record["best_validation_accuracy"]
        settings = ", ".join(f"{key}={value}" for key, value in
                             record["configuration"].items())
        parts.extend([
            f'<text x="20" y="{y}">Configuration {index + 1}: '
            f'{score:.4f} at epoch {record["best_epoch"]}</text>',
            f'<rect x="370" y="{y - 17}" width="{score * 580:.2f}" '
            'height="20" fill="#0969da"/>',
            f'<text x="20" y="{y + 24}" font-size="12">{html.escape(settings)}</text>',
        ])
    parts.append("</g></svg>")
    (output / "comparison.svg").write_text("\n".join(parts), encoding="utf-8")


def capture(configurations: Path, prepared: Path, output: Path, *, examples: int) -> dict:
    """Capture current participant code and prepared data without running training."""
    settings = _configurations(configurations)
    if examples < 0:
        raise ValueError("examples must not be negative")
    prepared, output = prepared.resolve(), output.resolve()
    for name in PREPARED_FILES:
        if not (prepared / name).is_file():
            raise FileNotFoundError(f"{prepared / name} is missing; prepare data first")
    # Never reuse a directory, including an interrupted launch's partial outputs.
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    source = output / "source"
    (source / "banking77").mkdir(parents=True)
    for path in sorted((project / "banking77").rglob("*.py")):
        destination = source / path.relative_to(project)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    if (project / "requirements.txt").exists():
        shutil.copyfile(project / "requirements.txt", source / "requirements.txt")
    hashes = {p.relative_to(source).as_posix(): _hash(p)
              for p in sorted(source.rglob("*")) if p.is_file()}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True,
    )
    if revision.returncode and (project / ".git").exists():
        raise RuntimeError(f"cannot record code revision: {revision.stderr.strip()}")
    data = output / "prepared"
    data.mkdir()
    for name in PREPARED_FILES:
        shutil.copyfile(prepared / name, data / name)
    manifest = {
        "schema_version": 1, "launch_id": uuid.uuid4().hex, "status": "running",
        "target": "local-cpu",
        "configurations": settings, "seed": SEED, "device": "cpu",
        "examples": examples, "software": _software(),
        "environment": {
            "platform": platform.platform(), "architecture": platform.machine(),
            "torch_threads": torch.get_num_threads(),
            "thread_settings": {name: os.environ[name] for name in
                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
                                if name in os.environ},
        },
        "source": {"revision": revision.stdout.strip() if revision.returncode == 0 else None,
                   "identity": _identity(hashes), "sha256": hashes},
        "prepared": {"sha256": {name: _hash(data / name) for name in PREPARED_FILES},
                     "metadata": json.loads((data / "metadata.json").read_text(encoding="utf-8"))},
        "results": [],
    }
    _write(output / "launch.json", manifest)
    return manifest


def launch(configurations: Path, prepared: Path, output: Path, *, examples: int) -> None:
    """Capture immutable inputs, then run every configuration in order on CPU."""
    manifest = capture(configurations, prepared, output, examples=examples)
    output = output.resolve()
    settings = manifest["configurations"]
    source, data = output / "source", output / "prepared"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    for index, configuration in enumerate(settings):
        print(f"Configuration {index + 1}/{len(settings)}: {configuration}", flush=True)
        run = output / "runs" / str(index)
        command = [sys.executable, "-m", "banking77.train", "--prepared", str(data),
                   "--output", str(run), "--examples", str(examples)]
        for key, value in configuration.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        result = subprocess.run(command, cwd=source, env=environment)
        if result.returncode:
            manifest["status"] = "failed"
            manifest["error"] = f"configuration {index + 1} failed (exit {result.returncode})"
            _write(output / "launch.json", manifest)
            raise RuntimeError(manifest["error"] + "; no winner; use a new output directory")
        record = json.loads((run / "run.json").read_text(encoding="utf-8"))
        record["provenance"] = {"launch_id": manifest["launch_id"],
                                "configuration_index": index, "source": manifest["source"],
                                "environment": manifest["environment"]}
        _write(run / "run.json", record)
        manifest["results"].append({
            "sha256": {name: _hash(run / name) for name in RESULT_FILES},
        })
        _write(output / "launch.json", manifest)
    manifest["status"] = "completed"
    _write(output / "launch.json", manifest)
    records = validate_results(output)
    _chart(output, records)
    print(f"Comparison complete: {output / 'comparison.svg'}. Select from this launch.")


def validate_results(output: Path, *, allow_partial: bool = False) -> list[dict]:
    """Reject partial or mixed comparisons before exposing records to selection."""
    manifest = json.loads((output / "launch.json").read_text(encoding="utf-8"))
    return _validate_results(output, manifest, allow_partial=allow_partial)


def _validate_results(output: Path, manifest: dict, *, allow_partial: bool,
                      pending_run: tuple[int, Path] | None = None) -> list[dict]:
    """Validate a candidate download before exposing it as a collected result."""
    configurations = manifest["configurations"]
    if (manifest["schema_version"] != 1 or len(configurations) < 2
            or (not allow_partial and (manifest["status"] != "completed"
                or len(manifest["results"]) != len(configurations)
                or not all(manifest["results"])))):
        raise ValueError("incomplete or failed comparison; every configuration needs valid results")
    target = manifest.get("target", "local-cpu")
    if target not in ("local-cpu", "azure"):
        raise ValueError("invalid launch target")
    if target == "local-cpu" and manifest["device"] != "cpu":
        raise ValueError("local comparisons require CPU; no compute fallback")
    if target == "azure":
        azure = manifest["azure"]
        if (azure["device"] not in ("cpu", "gpu") or azure["tier"] != "Standard"
                or manifest["device"] != ("cuda" if azure["device"] == "gpu" else "cpu")
                or len(manifest["results"]) != len(configurations)
                or len(manifest["jobs"]) > len(configurations)):
            raise ValueError("mismatched Azure compute or configuration list")
    source = manifest["source"]
    if not source["sha256"] or source["identity"] != _identity(source["sha256"]):
        raise ValueError("mismatched captured-code identity")
    for name, digest in source["sha256"].items():
        path = (output / "source" / name).resolve()
        if not path.is_relative_to((output / "source").resolve()) or _hash(path) != digest:
            raise ValueError(f"mismatched captured source: {name}")
    if set(manifest["prepared"]["sha256"]) != set(PREPARED_FILES):
        raise ValueError("incomplete prepared-data identity")
    for name, digest in manifest["prepared"]["sha256"].items():
        if _hash(output / "prepared" / name) != digest:
            raise ValueError(f"mismatched prepared data: {name}")
    records = []
    for index, configuration in enumerate(configurations):
        if allow_partial and (index >= len(manifest["results"])
                              or manifest["results"][index] is None):
            continue
        run = (pending_run[1] if pending_run is not None and pending_run[0] == index
               else output / "runs" / str(index))
        if set(manifest["results"][index]["sha256"]) != set(RESULT_FILES):
            raise ValueError(f"incomplete artifact identity for configuration {index + 1}")
        for name, digest in manifest["results"][index]["sha256"].items():
            if _hash(run / name) != digest:
                raise ValueError(f"mismatched result for configuration {index + 1}: {name}")
        record = json.loads((run / "run.json").read_text(encoding="utf-8"))
        if (record["schema_version"] != 1 or record["status"] != "completed"
                or record["configuration"] != configuration
                or record["provenance"] != {"launch_id": manifest["launch_id"],
                                            "configuration_index": index, "source": source,
                                            "environment": manifest["environment"]}
                or record["prepared"] != manifest["prepared"]
                or record["seed"] != manifest["seed"] or record["seed"] != SEED
                or record["device"] != manifest["device"]
                or record["device"] not in ("cpu", "cuda")
                or record["software"] != (
                    manifest["results"][index]["software"]
                    if manifest.get("target") == "azure" else manifest["software"])):
            raise ValueError(f"mismatched launch provenance for configuration {index + 1}")
        if manifest.get("target") == "azure":
            if (record.get("training_environment") != manifest["training_environment"]
                    or record.get("compute") != manifest["azure"]
                    or record.get("job_name") != manifest["jobs"][index]["name"]
                    or record.get("job_name") != f"banking77-{manifest['launch_id']}-{index}"):
                raise ValueError(f"mismatched Azure provenance for configuration {index + 1}")
            expected = manifest["software"]
            if (record["software"]["python"] != expected["python"]
                    or record["software"]["scikit-learn"] != expected["scikit-learn"]
                    or record["software"]["torch"] != expected["torch"].split("+")[0]
                    + ("+cu124" if manifest["device"] == "cuda" else "+cpu")):
                raise ValueError(f"mismatched Azure training software for configuration {index + 1}")
        settings = TrainingConfiguration(**configuration)
        settings.validate()
        history = record["history"]
        if (not history or type(record["epochs_completed"]) is not int
                or len(history) != record["epochs_completed"]
                or len(history) > settings.max_epochs
                or record["checkpoint"] != "best-model.pt"):
            raise ValueError(f"invalid history or checkpoint for configuration {index + 1}")
        for epoch, entry in enumerate(history, 1):
            if (type(entry["epoch"]) is not int or entry["epoch"] != epoch
                    or type(entry["training_loss"]) not in (int, float)
                    or not math.isfinite(entry["training_loss"]) or entry["training_loss"] < 0
                    or type(entry["validation_accuracy"]) not in (int, float)
                    or not math.isfinite(entry["validation_accuracy"])
                    or not 0 <= entry["validation_accuracy"] <= 1):
                raise ValueError(f"invalid epoch metrics for configuration {index + 1}")
        accuracies = [entry["validation_accuracy"] for entry in history]
        best_epoch = accuracies.index(max(accuracies)) + 1
        stopped = len(history) < settings.max_epochs
        if (type(record["best_epoch"]) is not int or record["best_epoch"] != best_epoch
                or type(record["best_validation_accuracy"]) not in (int, float)
                or record["best_validation_accuracy"] != max(accuracies)
                or type(record["stopped_early"]) is not bool
                or record["stopped_early"] != stopped
                or (stopped and len(history) - best_epoch < settings.patience)):
            raise ValueError(f"invalid best epoch or early stopping for configuration {index + 1}")
        for name in ("vocabulary.json", "labels.json"):
            if _hash(run / name) != manifest["prepared"]["sha256"][name]:
                # The run writes JSON with its own formatting; compare mapping content.
                if json.loads((run / name).read_text(encoding="utf-8")) != json.loads(
                    (output / "prepared" / name).read_text(encoding="utf-8")
                ):
                    raise ValueError(f"mismatched checkpoint mapping: {name}")
        checkpoint = torch.load(run / "best-model.pt", map_location="cpu", weights_only=True)
        model_settings = {
            "vocabulary_size": len(json.loads(
                (run / "vocabulary.json").read_text(encoding="utf-8"))),
            "num_labels": len(json.loads((run / "labels.json").read_text(encoding="utf-8"))),
            "embedding_size": settings.embedding_size, "hidden_size": settings.hidden_size,
            "dropout": settings.dropout,
        }
        if (checkpoint["schema_version"] != 1 or checkpoint["model_settings"] != model_settings
                or type(checkpoint["epoch"]) is not int
                or checkpoint["epoch"] != record["best_epoch"]
                or checkpoint["validation_accuracy"] != record["best_validation_accuracy"]
                or not checkpoint["state_dict"]):
            raise ValueError(f"mismatched checkpoint for configuration {index + 1}")
        records.append(record)
    return records


def select(output: Path) -> None:
    from banking77.training import select_configuration

    records = validate_results(output)
    index = select_configuration(records)
    if type(index) is not int or not 0 <= index < len(records):
        raise ValueError("select_configuration: return a zero-based index from the result list")
    _write(output / "selection.json", {
        "launch_id": records[index]["provenance"]["launch_id"],
        "configuration_index": index, "run": f"runs/{index}",
        "best_validation_accuracy": records[index]["best_validation_accuracy"],
    })
    print(f"Selected configuration {index + 1}: {output / 'runs' / str(index)}")


def main(argv=None, *, azure_service=None) -> int:
    parser = argparse.ArgumentParser(description="Compare configurations on local CPU or Azure ML.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("launch")
    run.add_argument("--configurations", type=Path, required=True)
    run.add_argument("--prepared", type=Path, default=Path("prepared"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--examples", type=int, default=5)
    run.add_argument("--target", choices=("local-cpu", "azure"), default="local-cpu",
                     help="Training target (default local-cpu for existing commands)")
    run.add_argument("--azure-device", choices=("cpu", "gpu"))
    run.add_argument("--instance-type", help="Explicit Azure VM size; no automatic fallback")
    run.add_argument("--subscription")
    run.add_argument("--resource-group")
    run.add_argument("--workspace")
    choose = commands.add_parser("select")
    choose.add_argument("--launch", type=Path, required=True)
    for name in ("status", "collect"):
        inspect = commands.add_parser(name)
        inspect.add_argument("--launch", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "launch":
            if arguments.target == "azure":
                from banking77.azure import launch_azure
                launch_azure(arguments, service=azure_service)
            else:
                if any((arguments.azure_device, arguments.instance_type,
                        arguments.subscription, arguments.resource_group, arguments.workspace)):
                    raise ValueError("Azure options require --target azure; no compute fallback")
                launch(arguments.configurations, arguments.prepared, arguments.output,
                       examples=arguments.examples)
        elif arguments.command == "select":
            select(arguments.launch)
        else:
            from banking77.azure import inspect_azure
            inspect_azure(arguments.launch, collect=arguments.command == "collect",
                          service=azure_service)
    except (NotImplementedError, ValueError, OSError, RuntimeError, KeyError,
            AttributeError, TypeError, IndexError, ZeroDivisionError,
            ImportError, EOFError, pickle.UnpicklingError) as error:
        print(f"Comparison failed: {error}. See docs/challenges/03-train-locally.md.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Provided local launch capture, comparison chart and result-validation plumbing."""
import argparse
from dataclasses import asdict
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import os
import shutil
import subprocess
import sys
import uuid

import sklearn
import torch

from banking77.runner import PREPARED_FILES, SEED, TrainingConfiguration
from banking77.artifacts import file_sha256, write_json
from banking77.azure_settings import job_name, torch_build, torch_device
from banking77.captured import captured_modules
from banking77.cli import cli_errors


RESULT_FILES = ("run.json", "best-model.pt", "vocabulary.json", "labels.json")


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


def write_chart(output: Path, records: list[dict]) -> None:
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


def validate_capture_inputs(
    configurations: Path, prepared: Path, examples: int
) -> list[dict]:
    settings = _configurations(configurations)
    if examples < 0:
        raise ValueError("examples must not be negative")
    prepared = prepared.resolve()
    for name in PREPARED_FILES:
        if not (prepared / name).is_file():
            raise FileNotFoundError(f"{prepared / name} is missing; prepare data first")
    return settings


def capture(
    configurations: Path, prepared: Path, output: Path, *, examples: int,
    create_output: bool = True,
) -> dict:
    """Capture current participant code and prepared data without running training."""
    settings = validate_capture_inputs(configurations, prepared, examples)
    prepared, output = prepared.resolve(), output.resolve()
    # Never reuse a directory, including an interrupted launch's partial outputs.
    if create_output:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError(f"{output} must exist before capturing into it")
    project = Path(__file__).resolve().parents[1]
    source = output / "source"
    (source / "banking77").mkdir(parents=True)
    for path in sorted((project / "banking77").rglob("*.py")):
        destination = source / path.relative_to(project)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    if (project / "requirements.txt").exists():
        shutil.copyfile(project / "requirements.txt", source / "requirements.txt")
    hashes = {p.relative_to(source).as_posix(): file_sha256(p)
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
        "prepared": {"sha256": {name: file_sha256(data / name) for name in PREPARED_FILES},
                     "metadata": json.loads((data / "metadata.json").read_text(encoding="utf-8"))},
        "results": [],
    }
    write_json(output / "launch.json", manifest)
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
            write_json(output / "launch.json", manifest)
            raise RuntimeError(manifest["error"] + "; no winner; use a new output directory")
        record = json.loads((run / "run.json").read_text(encoding="utf-8"))
        record["provenance"] = {"launch_id": manifest["launch_id"],
                                "configuration_index": index, "source": manifest["source"],
                                "environment": manifest["environment"]}
        write_json(run / "run.json", record)
        manifest["results"].append({
            "sha256": {name: file_sha256(run / name) for name in RESULT_FILES},
        })
        write_json(output / "launch.json", manifest)
    manifest["status"] = "completed"
    write_json(output / "launch.json", manifest)
    records = validate_results(output)
    write_chart(output, records)
    print(f"Comparison complete: {output / 'comparison.svg'}. Select from this launch.")


def validate_results(output: Path, *, allow_partial: bool = False) -> list[dict]:
    """Reject partial or mixed comparisons before exposing records to selection."""
    manifest = json.loads((output / "launch.json").read_text(encoding="utf-8"))
    return validate_manifest_results(output, manifest, allow_partial=allow_partial)


def expected_attempt(manifest: dict, index: int) -> dict:
    """Job, compute and device expected for a configuration's latest attempt."""
    recoveries = [entry for entry in manifest.get("recoveries", [])
                  if entry["configuration_index"] == index]
    if recoveries:
        latest = recoveries[-1]
        return {"job_name": latest["job_name"], "device": latest["device"],
                "compute": latest["compute"],
                "training_environment": latest["training_environment"],
                "recovery": latest}
    return {"job_name": job_name(manifest["launch_id"], index),
            "device": manifest["device"], "compute": manifest.get("azure"),
            "training_environment": manifest.get("training_environment"),
            "recovery": None}


def validate_manifest_results(output: Path, manifest: dict, *, allow_partial: bool,
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
                or manifest["device"] != torch_device(azure["device"])
                or len(manifest["results"]) != len(configurations)
                or len(manifest["jobs"]) > len(configurations)):
            raise ValueError("mismatched Azure compute or configuration list")
    source = manifest["source"]
    if not source["sha256"] or source["identity"] != _identity(source["sha256"]):
        raise ValueError("mismatched captured-code identity")
    for name, digest in source["sha256"].items():
        path = (output / "source" / name).resolve()
        if not path.is_relative_to((output / "source").resolve()) or file_sha256(path) != digest:
            raise ValueError(f"mismatched captured source: {name}")
    if set(manifest["prepared"]["sha256"]) != set(PREPARED_FILES):
        raise ValueError("incomplete prepared-data identity")
    for name, digest in manifest["prepared"]["sha256"].items():
        if file_sha256(output / "prepared" / name) != digest:
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
            if file_sha256(run / name) != digest:
                raise ValueError(f"mismatched result for configuration {index + 1}: {name}")
        record = json.loads((run / "run.json").read_text(encoding="utf-8"))
        attempt = expected_attempt(manifest, index)
        if (record["schema_version"] != 1 or record["status"] != "completed"
                or record["configuration"] != configuration
                or record["provenance"] != {"launch_id": manifest["launch_id"],
                                            "configuration_index": index, "source": source,
                                            "environment": manifest["environment"]}
                or record["prepared"] != manifest["prepared"]
                or record["seed"] != manifest["seed"] or record["seed"] != SEED
                or record["device"] != attempt["device"]
                or record["device"] not in ("cpu", "cuda")
                or record["software"] != (
                    manifest["results"][index]["software"]
                    if manifest.get("target") == "azure" else manifest["software"])):
            raise ValueError(f"mismatched launch provenance for configuration {index + 1}")
        if manifest.get("target") == "azure":
            if (record.get("training_environment") != attempt["training_environment"]
                    or record.get("compute") != attempt["compute"]
                    or record.get("job_name") != manifest["jobs"][index]["name"]
                    or record.get("job_name") != attempt["job_name"]
                    or record.get("recovery") != attempt["recovery"]):
                raise ValueError(f"mismatched Azure provenance for configuration {index + 1}")
            expected = manifest["software"]
            if (record["software"]["python"] != expected["python"]
                    or record["software"]["scikit-learn"] != expected["scikit-learn"]
                    or record["software"]["torch"] != expected["torch"].split("+")[0]
                    + "+" + torch_build(attempt["device"])):
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
            if file_sha256(run / name) != manifest["prepared"]["sha256"][name]:
                # The run writes JSON with its own formatting; compare mapping content.
                if json.loads((run / name).read_text(encoding="utf-8")) != json.loads(
                    (output / "prepared" / name).read_text(encoding="utf-8")
                ):
                    raise ValueError(f"mismatched checkpoint mapping: {name}")
        checkpoint = torch.load(run / "best-model.pt", map_location="cpu", weights_only=True)
        model_settings = settings.model_settings(
            len(json.loads(
                (run / "vocabulary.json").read_text(encoding="utf-8"))),
            len(json.loads((run / "labels.json").read_text(encoding="utf-8"))),
        )
        if (checkpoint["schema_version"] != 1 or checkpoint["model_settings"] != model_settings
                or type(checkpoint["epoch"]) is not int
                or checkpoint["epoch"] != record["best_epoch"]
                or checkpoint["validation_accuracy"] != record["best_validation_accuracy"]
                or not checkpoint["state_dict"]):
            raise ValueError(f"mismatched checkpoint for configuration {index + 1}")
        with captured_modules(output / "source", ("data", "model")) as captured:
            model = captured["model"].IntentClassifier(**model_settings)
            try:
                model.load_state_dict(checkpoint["state_dict"], strict=True)
            except (RuntimeError, TypeError) as error:
                raise ValueError(
                    f"invalid model parameters for configuration {index + 1}: {error}"
                ) from error
            if any(not torch.isfinite(parameter).all().item()
                   for parameter in model.state_dict().values()):
                raise ValueError(f"non-finite model parameters for configuration {index + 1}")
        records.append(record)
    return records


def selection_record(records: list[dict]) -> dict:
    """Build the selection using the participant's current selection function."""
    from banking77.training import select_configuration

    index = select_configuration(records)
    if type(index) is not int or not 0 <= index < len(records):
        raise ValueError("select_configuration: return a zero-based index from the result list")
    return {
        "launch_id": records[index]["provenance"]["launch_id"],
        "configuration_index": index, "run": f"runs/{index}",
        "best_validation_accuracy": records[index]["best_validation_accuracy"],
    }


def select(output: Path) -> None:
    selection = selection_record(validate_results(output))
    index = selection["configuration_index"]
    write_json(output / "selection.json", selection)
    print(f"Selected configuration {index + 1}: {output / 'runs' / str(index)}")


@cli_errors("Comparison", "03-train-locally.md")
def main(argv=None, *, azure_service=None) -> int:
    parser = argparse.ArgumentParser(description="Compare configurations on local CPU or Azure ML.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("launch")
    run.add_argument("--configurations", type=Path, required=True)
    run.add_argument("--prepared", type=Path, default=Path("prepared"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--examples", type=int, default=5)
    run.add_argument("--target", choices=("local-cpu", "azure"), required=True,
                     help="Explicit training target; no automatic compute fallback")
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
    recover = commands.add_parser(
        "recover", help="Explicitly recover a failed job or rejected completed output")
    recover.add_argument("--launch", type=Path, required=True)
    recover.add_argument("--job", required=True, help="Current Azure job name from status")
    mode = recover.add_mutually_exclusive_group(required=True)
    mode.add_argument("--resume", action="store_true",
                      help="Continue from the job's latest resumable snapshot")
    mode.add_argument("--restart", action="store_true",
                      help="Train this configuration again from epoch 1")
    recover.add_argument("--azure-device", choices=("cpu", "gpu"),
                         help="Explicit device change (e.g. CPU after GPU); default keeps it")
    recover.add_argument("--instance-type", help="Required with --azure-device")
    arguments = parser.parse_args(argv)
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
    elif arguments.command == "recover":
        from banking77.azure import recover_azure
        recover_azure(arguments, service=azure_service)
    else:
        from banking77.azure import inspect_azure
        inspect_azure(arguments.launch, collect=arguments.command == "collect",
                      service=azure_service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

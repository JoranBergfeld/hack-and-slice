"""Provided plumbing that runs one training configuration with participant code."""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
from typing import Callable

import sklearn
import torch

from banking77.artifacts import file_sha256, write_json

SEED = 42
RUN_RECORD = "run.json"
LATEST_SNAPSHOT = "latest-snapshot.pt"
SNAPSHOT_VERSION = 1
PREPARED_FILES = (
    "training.json", "validation.json", "vocabulary.json", "labels.json", "metadata.json",
)


@dataclass(frozen=True)
class TrainingConfiguration:
    """One participant-editable training configuration and its starting values."""

    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 3
    embedding_size: int = 64
    hidden_size: int = 128
    dropout: float = 0.3

    def model_settings(self, vocabulary_size: int, num_labels: int) -> dict:
        return {
            "vocabulary_size": vocabulary_size, "num_labels": num_labels,
            "embedding_size": self.embedding_size, "hidden_size": self.hidden_size,
            "dropout": self.dropout,
        }

    def validate(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        for name in ("batch_size", "max_epochs", "patience", "embedding_size", "hidden_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name.replace('_', ' ')} must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def _read_prepared(prepared: Path) -> dict:
    contents = {}
    for name in PREPARED_FILES:
        path = prepared / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run python -m banking77.prepare first"
            )
        contents[name] = json.loads(path.read_text(encoding="utf-8"))
    for name in ("training.json", "validation.json"):
        if not contents[name]:
            raise ValueError(f"{prepared / name} contains no messages")
    return contents


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _save_durably(contents: dict, path: Path) -> None:
    """Replace path only after the complete file has reached the disk."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(contents, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _finite_tree(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _restore_snapshot_model(model_factory, settings: dict, state: dict, label: str):
    model = model_factory(settings)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"{label} does not match the captured classifier: {error}") from error
    if not _finite_tree(model.state_dict()):
        raise ValueError(f"{label} contains non-finite parameters")
    return model


def load_snapshot(
    path: Path | str,
    *,
    configuration: TrainingConfiguration,
    model_settings: dict,
    identity: dict,
    prepared_sha256: dict,
    model_factory,
    optimizer_factory,
) -> dict:
    """Validate a resumable snapshot before restoring or submitting it."""
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(snapshot, dict) or snapshot.get("kind") != "latest-resumable-snapshot"
            or type(snapshot.get("schema_version")) is not int
            or snapshot["schema_version"] != SNAPSHOT_VERSION):
        raise ValueError(f"{path}: not a supported latest resumable snapshot")
    required = {
        "identity", "configuration", "model_settings", "seed", "prepared_sha256",
        "device", "completed_epoch", "history", "best", "early_stopping",
        "model_state", "optimizer_state", "random_state",
    }
    if not required.issubset(snapshot):
        raise ValueError(f"{path}: incomplete resumable snapshot payload")
    if (snapshot["identity"] != identity
            or snapshot["configuration"] != asdict(configuration)
            or snapshot["model_settings"] != model_settings
            or snapshot["seed"] != SEED
            or snapshot["prepared_sha256"] != prepared_sha256):
        raise ValueError(f"{path}: snapshot belongs to a different run, configuration "
                         "or prepared data")
    epoch = snapshot["completed_epoch"]
    history = snapshot["history"]
    if (type(epoch) is not int or not 1 <= epoch <= configuration.max_epochs
            or not isinstance(history, list) or len(history) != epoch):
        raise ValueError(f"{path}: snapshot epoch and history are inconsistent")
    accuracies = []
    for number, entry in enumerate(history, start=1):
        if (not isinstance(entry, dict) or type(entry.get("epoch")) is not int
                or entry["epoch"] != number
                or type(entry.get("training_loss")) not in (int, float)
                or (type(entry["training_loss"]) is float
                    and not math.isfinite(entry["training_loss"]))
                or entry["training_loss"] < 0
                or type(entry.get("validation_accuracy")) not in (int, float)
                or (type(entry["validation_accuracy"]) is float
                    and not math.isfinite(entry["validation_accuracy"]))
                or not 0 <= entry["validation_accuracy"] <= 1
                or entry.get("device") not in ("cpu", "cuda")):
            raise ValueError(f"{path}: invalid history at epoch {number}")
        accuracies.append(entry["validation_accuracy"])
    best = snapshot["best"]
    if (not isinstance(best, dict) or type(best.get("epoch")) is not int
            or best["epoch"] != accuracies.index(max(accuracies)) + 1
            or type(best.get("validation_accuracy")) not in (int, float)
            or (type(best["validation_accuracy"]) is float
                and not math.isfinite(best["validation_accuracy"]))
            or best.get("validation_accuracy") != max(accuracies)):
        raise ValueError(f"{path}: best checkpoint does not match validation history")
    early = snapshot["early_stopping"]
    expected_stop = epoch - best["epoch"] >= configuration.patience
    if (not isinstance(early, dict)
            or type(early.get("patience")) is not int
            or early["patience"] != configuration.patience
            or type(early.get("epochs_without_improvement")) is not int
            or early.get("epochs_without_improvement") != epoch - best["epoch"]
            or type(early.get("stop")) is not bool or early["stop"] != expected_stop):
        raise ValueError(f"{path}: early-stopping state does not match history")
    if (snapshot["device"] not in ("cpu", "cuda")
            or history[-1]["device"] != snapshot["device"]):
        raise ValueError(f"{path}: snapshot device does not match latest epoch")

    model = _restore_snapshot_model(
        model_factory, model_settings, snapshot["model_state"], "latest model state")
    best_model = _restore_snapshot_model(
        model_factory, model_settings, best.get("state_dict"), "best model state")
    optimizer = optimizer_factory(model, configuration.learning_rate)
    try:
        optimizer.load_state_dict(snapshot["optimizer_state"])
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: optimizer state is incompatible: {error}") from error
    expected_parameters = sum(parameter.requires_grad for parameter in model.parameters())
    try:
        parameter_group = optimizer.param_groups[0]
        group_parameters = parameter_group["params"]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(f"{path}: optimizer parameter groups are invalid") from error
    if (not optimizer.state or len(optimizer.state) != expected_parameters
            or len(optimizer.param_groups) != 1 or len(group_parameters) != expected_parameters):
        raise ValueError(f"{path}: optimizer state is incomplete")
    if not _finite_tree(optimizer.state_dict()):
        raise ValueError(f"{path}: optimizer state contains non-finite values")
    for parameter, state in optimizer.state.items():
        if (not isinstance(state, dict)
                or not isinstance(state.get("exp_avg"), torch.Tensor)
                or not isinstance(state.get("exp_avg_sq"), torch.Tensor)
                or state["exp_avg"].shape != parameter.shape
                or state["exp_avg_sq"].shape != parameter.shape
                or not isinstance(state.get("step"), torch.Tensor)
                or state["step"].numel() != 1 or not _finite_tree(state["step"])
                or state["step"].item() < 1):
            raise ValueError(f"{path}: optimizer tensors do not match model parameters")
    randomness = snapshot["random_state"]
    if not isinstance(randomness, dict):
        raise ValueError(f"{path}: incomplete random state")
    try:
        for name in ("torch_cpu", "data_order"):
            state = randomness[name]
            if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8:
                raise ValueError(f"invalid {name} generator state")
            torch.Generator(device="cpu").set_state(state)
        random.Random().setstate(randomness["python"])
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid random state: {error}") from error
    cuda_state = randomness.get("torch_cuda")
    if (not isinstance(cuda_state, list)
            or (snapshot["device"] == "cuda" and
                (not cuda_state or any(not isinstance(state, torch.Tensor)
                                       or state.dtype != torch.uint8 or state.ndim != 1
                                       for state in cuda_state)))
            or (snapshot["device"] == "cpu" and cuda_state)):
        raise ValueError(f"{path}: invalid CUDA random state")
    return snapshot


def run_training(
    configuration: TrainingConfiguration,
    prepared: Path,
    output: Path,
    *,
    device: torch.device = torch.device("cpu"),
    examples: int = 5,
    log: Callable[[str], None] = print,
    target: str = "local CPU",
    resume: Path | None = None,
    identity: dict | None = None,
) -> dict:
    """Train one configuration, save its best-validation classifier and run record.

    After every completed epoch a separate latest resumable snapshot is saved.
    resume continues the same run from such a snapshot in a new output folder.
    """
    from banking77.checkpoint import load_classifier, save_classifier
    from banking77.data import Message
    from banking77.model import IntentClassifier, collate_batch
    from banking77.training import (
        build_loader, create_optimizer, early_stopping, train_epoch, validation_accuracy,
    )

    configuration.validate()
    if examples < 0:
        raise ValueError("examples must not be negative")
    if (output / RUN_RECORD).exists():
        raise FileExistsError(
            f"{output / RUN_RECORD} already exists; choose a new --output for each run"
        )
    contents = _read_prepared(prepared)
    vocabulary = contents["vocabulary.json"]
    label_mapping = contents["labels.json"]
    training = [Message(row["text"], row["label"]) for row in contents["training.json"]]
    validation = [Message(row["text"], row["label"]) for row in contents["validation.json"]]

    torch.manual_seed(SEED)
    model_settings = configuration.model_settings(len(vocabulary), len(label_mapping))
    prepared_sha256 = {name: file_sha256(prepared / name) for name in PREPARED_FILES}
    model = IntentClassifier(**model_settings).to(device)
    data_order = torch.Generator().manual_seed(SEED)
    training_loader = build_loader(
        training, vocabulary, label_mapping, configuration.batch_size,
        shuffle=True, generator=data_order,
    )
    validation_loader = build_loader(
        validation, vocabulary, label_mapping, configuration.batch_size, shuffle=False,
    )
    optimizer = create_optimizer(model, configuration.learning_rate)

    log(
        f"Training on {target}: {len(training):,} training / "
        f"{len(validation):,} validation messages, seed {SEED}. "
        f"Settings: {asdict(configuration)}"
    )
    history, accuracies = [], []
    best_epoch, best_state, stop, resumed = 0, None, False, None
    if resume is not None:
        snapshot = load_snapshot(
            resume, configuration=configuration, model_settings=model_settings,
            identity=identity, prepared_sha256=prepared_sha256,
            model_factory=lambda settings: IntentClassifier(**settings),
            optimizer_factory=lambda current_model, learning_rate: create_optimizer(
                current_model, learning_rate),
        )
        model.load_state_dict(snapshot["model_state"])
        optimizer.load_state_dict(snapshot["optimizer_state"])
        history = [dict(entry) for entry in snapshot["history"]]
        accuracies = [entry["validation_accuracy"] for entry in history]
        best_epoch = snapshot["best"]["epoch"]
        best_state = snapshot["best"]["state_dict"]
        stop = snapshot["early_stopping"]["stop"]
        best = IntentClassifier(**model_settings)
        best.load_state_dict(best_state)
        save_classifier(
            output, best, model_settings, vocabulary, label_mapping,
            epoch=best_epoch, validation_accuracy=accuracies[best_epoch - 1],
        )
        # Restore randomness last: building the classifier above consumes it.
        randomness = snapshot["random_state"]
        torch.set_rng_state(randomness["torch_cpu"])
        data_order.set_state(randomness["data_order"])
        random.setstate(randomness["python"])
        if device.type == "cuda" and snapshot["device"] == "cuda" and randomness["torch_cuda"]:
            torch.cuda.set_rng_state_all(randomness["torch_cuda"])
        resumed = {"after_epoch": len(history), "snapshot_device": snapshot["device"]}
        log(
            f"Resuming after epoch {len(history)} from the latest snapshot "
            f"(trained on {snapshot['device']}); best validation "
            f"{accuracies[best_epoch - 1]:.4f} at epoch {best_epoch}."
        )
        if snapshot["device"] != device.type:
            log(f"Continuing on {device.type} after {snapshot['device']}: results are "
                "valid but not guaranteed bit-for-bit identical across devices.")
    for epoch in range(len(history) + 1, configuration.max_epochs + 1):
        if stop:
            break
        loss = float(train_epoch(model, training_loader, optimizer, device))
        accuracy = float(validation_accuracy(model, validation_loader, device))
        history.append(
            {"epoch": epoch, "training_loss": loss, "validation_accuracy": accuracy,
             "device": device.type}
        )
        accuracies.append(accuracy)
        best_epoch, stop = early_stopping(accuracies, configuration.patience)
        improved = best_epoch == epoch
        if improved:
            best_state = _to_cpu(model.state_dict())
            save_classifier(
                output, model, model_settings, vocabulary, label_mapping,
                epoch=epoch, validation_accuracy=accuracy,
            )
        output.mkdir(parents=True, exist_ok=True)
        _save_durably({
            "schema_version": SNAPSHOT_VERSION,
            "kind": "latest-resumable-snapshot",
            "identity": identity,
            "configuration": asdict(configuration),
            "model_settings": model_settings,
            "seed": SEED,
            "prepared_sha256": prepared_sha256,
            "device": device.type,
            "completed_epoch": epoch,
            "history": history,
            "best": {"epoch": best_epoch, "validation_accuracy": accuracies[best_epoch - 1],
                     "state_dict": best_state},
            "early_stopping": {"patience": configuration.patience,
                               "epochs_without_improvement": epoch - best_epoch,
                               "stop": stop},
            "model_state": _to_cpu(model.state_dict()),
            "optimizer_state": _to_cpu(optimizer.state_dict()),
            "random_state": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (_to_cpu(torch.cuda.get_rng_state_all())
                               if device.type == "cuda" else []),
                "data_order": data_order.get_state(),
                "python": random.getstate(),
            },
        }, output / LATEST_SNAPSHOT)
        log(
            f"Epoch {epoch}/{configuration.max_epochs}: training loss {loss:.4f}, "
            f"validation accuracy {accuracy:.4f}" + (" (best)" if improved else "")
        )
    stopped_early = stop and len(history) < configuration.max_epochs
    if stopped_early:
        log(
            f"Early stopping after epoch {len(history)}: no validation improvement "
            f"for {len(history) - best_epoch} epochs."
        )

    saved = load_classifier(output, device)
    if saved.epoch != best_epoch:
        raise RuntimeError(
            f"saved checkpoint is from epoch {saved.epoch}, expected best epoch {best_epoch}"
        )
    indices = {index: label for label, index in label_mapping.items()}
    shown = validation[:examples]
    predictions = []
    if shown:
        tokens, offsets, _ = collate_batch(shown, vocabulary, label_mapping)
        with torch.no_grad():
            predicted = saved.model(tokens.to(device), offsets.to(device)).argmax(dim=1)
        log("Validation examples (never the held-out test set):")
        for message, index in zip(shown, predicted.tolist()):
            predictions.append(
                {"text": message.text, "label": message.label, "predicted": indices[index]}
            )
            log(f"  {message.text!r} -> {indices[index]} (correct: {message.label})")

    record = {
        "schema_version": 1,
        "status": "completed",
        "configuration": asdict(configuration),
        "seed": SEED,
        "device": device.type,
        "history": history,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": accuracies[best_epoch - 1],
        "stopped_early": stopped_early,
        "checkpoint": "best-model.pt",
        "latest_snapshot": LATEST_SNAPSHOT,
        "resumed": resumed,
        "prepared": {
            "metadata": contents["metadata.json"],
            "sha256": prepared_sha256,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "examples": predictions,
    }
    write_json(output / RUN_RECORD, record)
    log(
        f"Best validation accuracy {record['best_validation_accuracy']:.4f} at epoch "
        f"{best_epoch} of {len(history)}. Saved: {output}"
    )
    return record

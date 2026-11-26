"""Provided plumbing that runs one training configuration with participant code."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Callable

import sklearn
import torch

SEED = 42
RUN_RECORD = "run.json"
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

    def validate(self) -> None:
        if not self.learning_rate > 0:
            raise ValueError("learning rate must be positive")
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_training(
    configuration: TrainingConfiguration,
    prepared: Path,
    output: Path,
    *,
    device: torch.device = torch.device("cpu"),
    examples: int = 5,
    log: Callable[[str], None] = print,
    target: str = "local CPU",
) -> dict:
    """Train one configuration, save its best-validation classifier and run record."""
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
    model_settings = {
        "vocabulary_size": len(vocabulary),
        "embedding_size": configuration.embedding_size,
        "hidden_size": configuration.hidden_size,
        "dropout": configuration.dropout,
        "num_labels": len(label_mapping),
    }
    model = IntentClassifier(**model_settings).to(device)
    training_loader = build_loader(
        training, vocabulary, label_mapping, configuration.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(SEED),
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
    best_epoch, stopped_early = 0, False
    for epoch in range(1, configuration.max_epochs + 1):
        loss = float(train_epoch(model, training_loader, optimizer, device))
        accuracy = float(validation_accuracy(model, validation_loader, device))
        history.append(
            {"epoch": epoch, "training_loss": loss, "validation_accuracy": accuracy}
        )
        accuracies.append(accuracy)
        best_epoch, stop = early_stopping(accuracies, configuration.patience)
        improved = best_epoch == epoch
        if improved:
            save_classifier(
                output, model, model_settings, vocabulary, label_mapping,
                epoch=epoch, validation_accuracy=accuracy,
            )
        log(
            f"Epoch {epoch}/{configuration.max_epochs}: training loss {loss:.4f}, "
            f"validation accuracy {accuracy:.4f}" + (" (best)" if improved else "")
        )
        if stop:
            stopped_early = epoch < configuration.max_epochs
            if stopped_early:
                log(
                    f"Early stopping after epoch {epoch}: no validation improvement "
                    f"for {epoch - best_epoch} epochs."
                )
            break

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
        "prepared": {
            "metadata": contents["metadata.json"],
            "sha256": {name: _sha256(prepared / name) for name in PREPARED_FILES},
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "examples": predictions,
    }
    (output / RUN_RECORD).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(
        f"Best validation accuracy {record['best_validation_accuracy']:.4f} at epoch "
        f"{best_epoch} of {len(history)}. Saved: {output}"
    )
    return record

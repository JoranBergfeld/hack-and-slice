"""Provided plumbing for saving and loading a trained classifier."""
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from banking77.model import IntentClassifier
from banking77.artifacts import write_json


BEST_MODEL = "best-model.pt"
VOCABULARY = "vocabulary.json"
LABELS = "labels.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SavedClassifier:
    model: IntentClassifier
    vocabulary: dict[str, int]
    label_mapping: dict[str, int]
    epoch: int
    validation_accuracy: float


def save_classifier(
    directory: Path,
    model: IntentClassifier,
    model_settings: dict,
    vocabulary: dict[str, int],
    label_mapping: dict[str, int],
    *,
    epoch: int,
    validation_accuracy: float,
) -> None:
    """Save the best-validation classifier with the mappings needed to reuse it."""
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / VOCABULARY, vocabulary)
    write_json(directory / LABELS, label_mapping)
    temporary = directory / (BEST_MODEL + ".tmp")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
            "model_settings": dict(model_settings),
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
        },
        temporary,
    )
    temporary.replace(directory / BEST_MODEL)


def load_classifier(
    directory: Path | str, device: torch.device | str = "cpu"
) -> SavedClassifier:
    """Rebuild a saved classifier in evaluation mode with its vocabulary and labels."""
    directory = Path(directory)
    for name in (BEST_MODEL, VOCABULARY, LABELS):
        if not (directory / name).exists():
            raise FileNotFoundError(f"{directory / name} is missing; train a classifier first")
    checkpoint = torch.load(directory / BEST_MODEL, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{directory / BEST_MODEL}: unsupported checkpoint version")
    vocabulary = json.loads((directory / VOCABULARY).read_text(encoding="utf-8"))
    label_mapping = json.loads((directory / LABELS).read_text(encoding="utf-8"))
    for name, mapping in ((VOCABULARY, vocabulary), (LABELS, label_mapping)):
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(key, str) or not key for key in mapping)
                or any(type(identifier) is not int for identifier in mapping.values())
                or set(mapping.values()) != set(range(len(mapping)))):
            raise ValueError(f"{directory / name}: expected unique contiguous integer IDs")
    if vocabulary.get("<unk>") != 0:
        raise ValueError(f"{directory / VOCABULARY}: <unk> must have ID 0")
    settings = checkpoint["model_settings"]
    if (
        settings["vocabulary_size"] != len(vocabulary)
        or settings["num_labels"] != len(label_mapping)
    ):
        raise ValueError(f"{directory}: checkpoint does not match its vocabulary or labels")
    model = IntentClassifier(**settings).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if any(not torch.isfinite(parameter).all().item()
           for parameter in model.state_dict().values()):
        raise ValueError(f"{directory / BEST_MODEL}: non-finite checkpoint parameters")
    model.eval()
    return SavedClassifier(
        model, vocabulary, label_mapping,
        checkpoint["epoch"], checkpoint["validation_accuracy"],
    )

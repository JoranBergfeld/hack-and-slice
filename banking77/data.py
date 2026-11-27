from dataclasses import dataclass
import csv
import json
from pathlib import Path
from urllib.request import urlopen


DATA_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/"
    "task-specific-datasets/master/banking_data"
)


@dataclass(frozen=True)
class Message:
    text: str
    label: str


def split_training(messages: list[Message]) -> tuple[list[Message], list[Message]]:
    """Split training messages into stratified 90/10 train/validation lists.

    Use a library function with seed 42. Never pass held-out test messages here.
    The official 10,003 rows yield 9,002 training and 1,001 validation rows.
    """
    raise NotImplementedError("Implement split_training")


def tokenize(text: str) -> list[str]:
    """Return lowercase tokens matching [a-z0-9']+ from a message string."""
    raise NotImplementedError("Implement tokenize")


def build_vocabulary(training: list[Message]) -> dict[str, int]:
    """Return token IDs from training messages only.

    Keep tokens occurring at least twice; reserve <unk> at ID 0.
    Assign the remaining IDs deterministically in sorted token order.
    """
    raise NotImplementedError("Implement build_vocabulary")


def encode_message(text: str, vocabulary: dict[str, int]) -> list[int]:
    """Return token IDs, using 0 for unseen tokens and [0] for empty tokens."""
    raise NotImplementedError("Implement encode_message")


def build_label_mapping(labels: list[str]) -> dict[str, int]:
    """Return deterministic contiguous IDs for sorted official intent labels.

    Preserve the labels' original spelling, including capitalization.
    """
    raise NotImplementedError("Implement build_label_mapping")


def read_labels(path: Path) -> list[str]:
    """Read the official categories.json list without changing label spelling."""
    labels = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(f"{path}: expected a nonempty list of unique intent names")
    return labels


def read_messages(path: Path, labels: list[str]) -> list[Message]:
    """Read text/category CSV rows and reject missing or unknown intent labels."""
    messages = []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"{path}: expected CSV headers text,category")
        for number, row in enumerate(reader, start=2):
            if row.get("text") is None or row.get("category") not in labels or None in row:
                raise ValueError(f"{path}:{number}: missing text or invalid intent label")
            messages.append(Message(row["text"], row["category"]))
    if not messages:
        raise ValueError(f"{path}: no messages found")
    return messages


def _ensure_file(directory: Path, name: str, download: bool) -> Path:
    path = directory / name
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} is missing; supply official files or use --download"
            )
        directory.mkdir(parents=True, exist_ok=True)
        with urlopen(f"{DATA_URL}/{name}", timeout=60) as response:
            contents = response.read()
        path.write_bytes(contents)
    return path


def load_training(
    directory: Path, *, download: bool = False
) -> tuple[list[Message], list[str]]:
    """Load 10,003 official training rows and 77 labels; never access test.csv.

    Download missing train.csv/categories.json only when explicitly requested.
    """
    labels = read_labels(_ensure_file(directory, "categories.json", download))
    messages = read_messages(_ensure_file(directory, "train.csv", download), labels)
    if len(labels) != 77 or len(messages) != 10003:
        raise ValueError(
            "Official Banking77 training requires 10,003 messages and 77 labels; "
            "supply official train.csv and categories.json with --data-dir or --download"
        )
    return messages, labels


def load_test(directory: Path, *, download: bool = False) -> list[Message]:
    """Load 3,080 held-out rows only for final evaluation after model selection.

    This explicit loader is never called by preparation or quick checks.
    """
    labels = read_labels(_ensure_file(directory, "categories.json", download))
    messages = read_messages(_ensure_file(directory, "test.csv", download), labels)
    if len(labels) != 77 or len(messages) != 3080:
        raise ValueError("Official Banking77 test requires 3,080 messages and 77 labels")
    return messages

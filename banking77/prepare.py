import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from banking77 import data


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the Banking77 training data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("prepared"))
    parser.add_argument(
        "--download", action="store_true",
        help="Download missing official training files",
    )
    parser.add_argument(
        "--training-csv", type=Path,
        help="Use a custom training CSV, without official row-count checks",
    )
    parser.add_argument(
        "--categories", type=Path, help="Label JSON required with --training-csv",
    )
    arguments = parser.parse_args()
    if bool(arguments.training_csv) != bool(arguments.categories):
        parser.error("--training-csv and --categories must be supplied together")
    if arguments.training_csv and arguments.download:
        parser.error("--download cannot be combined with custom training files")
    try:
        if arguments.training_csv:
            source = arguments.training_csv
            categories = arguments.categories
            labels = data.read_labels(categories)
            messages = data.read_messages(source, labels)
        else:
            source = arguments.data_dir / "train.csv"
            categories = arguments.data_dir / "categories.json"
            messages, labels = data.load_training(
                arguments.data_dir, download=arguments.download
            )
        training, validation = data.split_training(messages)
        vocabulary = data.build_vocabulary(training)
        label_mapping = data.build_label_mapping(labels)
        artifacts = {
            "training.json": [asdict(message) for message in training],
            "validation.json": [asdict(message) for message in validation],
            "vocabulary.json": vocabulary,
            "labels.json": label_mapping,
            "metadata.json": {
                "schema_version": 1,
                "seed": 42,
                "training_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "categories_sha256": hashlib.sha256(categories.read_bytes()).hexdigest(),
                "source_rows": len(messages),
                "training_rows": len(training),
                "validation_rows": len(validation),
            },
        }
        arguments.output.mkdir(parents=True, exist_ok=True)
        for name, contents in artifacts.items():
            (arguments.output / name).write_text(
                json.dumps(contents, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except (NotImplementedError, ValueError, OSError) as error:
        print(
            f"Preparation failed: {error}. "
            "See docs/challenges/01-prepare-data.md.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Prepared {len(training):,} training / {len(validation):,} validation / "
        f"{len(label_mapping)} intents; vocabulary size {len(vocabulary):,}. "
        f"Output: {arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

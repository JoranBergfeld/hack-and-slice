import argparse
from pathlib import Path
import sys

from banking77.runner import TrainingConfiguration, run_training


def main() -> int:
    defaults = TrainingConfiguration()
    parser = argparse.ArgumentParser(
        description="Train one Banking77 classifier configuration on local CPU."
    )
    parser.add_argument("--prepared", type=Path, default=Path("prepared"))
    parser.add_argument("--output", type=Path, default=Path("runs/local"))
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--embedding-size", type=int, default=defaults.embedding_size)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument(
        "--examples", type=int, default=5,
        help="Validation messages to show with predictions after training",
    )
    arguments = parser.parse_args()
    configuration = TrainingConfiguration(
        learning_rate=arguments.learning_rate,
        batch_size=arguments.batch_size,
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
        embedding_size=arguments.embedding_size,
        hidden_size=arguments.hidden_size,
        dropout=arguments.dropout,
    )
    try:
        run_training(
            configuration, arguments.prepared, arguments.output,
            examples=arguments.examples,
        )
    except (
        NotImplementedError, ValueError, OSError, RuntimeError,
        KeyError, AttributeError, TypeError, IndexError, ZeroDivisionError,
    ) as error:
        print(
            f"Training failed: {error}. "
            "See docs/challenges/03-train-locally.md.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

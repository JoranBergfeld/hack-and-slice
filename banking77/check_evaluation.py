"""Tiny offline checks; never load held-out data or a trained classifier."""
import sys

from banking77.evaluation import accuracy, macro_f1


def check() -> None:
    targets, predictions = [0, 0, 1, 1, 2], [0, 1, 1, 1, 1]
    assert abs(accuracy(targets, predictions) - 0.6) < 1e-9, (
        "accuracy: divide three correct predictions by five messages"
    )
    assert abs(macro_f1(targets, predictions, 3) - 4 / 9) < 1e-9, (
        "macro_f1: average per-label F1 (2/3, 2/3, 0), not accuracy or weighted F1"
    )
    assert abs(macro_f1(targets, predictions, 4) - 1 / 3) < 1e-9, (
        "macro_f1: include absent labels with zero F1"
    )
    for metric in (accuracy, lambda actual, predicted: macro_f1(actual, predicted, 3)):
        assert metric([0, 1, 2], [0, 1, 2]) == 1.0, (
            "metrics: perfect predictions must score 1"
        )
        assert metric([0, 1, 2], [1, 2, 0]) == 0.0, (
            "metrics: completely wrong predictions must score 0"
        )
        for actual, predicted in (([], []), ([0], []), ([0], [0, 1])):
            try:
                metric(actual, predicted)
            except ValueError:
                pass
            else:
                raise AssertionError("metrics: reject empty or unequal-length inputs")
    for actual, predicted, count in (([0], [0], 0), ([0], [3], 3), ([-1], [0], 3)):
        try:
            macro_f1(actual, predicted, count)
        except ValueError:
            pass
        else:
            raise AssertionError("macro_f1: reject invalid label count or IDs")


def main() -> int:
    try:
        check()
    except (NotImplementedError, AssertionError, ValueError, TypeError,
            ZeroDivisionError) as error:
        print(f"Evaluation check failed: {error}. "
              "See docs/challenges/04-evaluate-classifier.md.", file=sys.stderr)
        return 1
    print("Evaluation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

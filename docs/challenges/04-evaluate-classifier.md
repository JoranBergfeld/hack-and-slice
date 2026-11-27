# Checkpoint 4: Evaluate the selected classifier

## Goal

Implement `accuracy` and `macro_f1` in `banking77/evaluation.py` without
metric-library functions, then evaluate the validation-selected winner on
local CPU. Complete [Checkpoint 3](03-train-locally.md), including selection,
first. There is no required score or written results report.

## Why this step?

Validation helped you choose settings and an epoch. Because those choices
adapted to validation results, validation is no longer an independent estimate
of final performance. The official held-out test set has 3,080 messages that
must stay out of training, vocabulary building, early stopping and selection.
Use it for **one final evaluation**, after committing to the selected winner.
Do not change settings or choose another winner based on test scores. This is
a learning rule, not a lock or one-use token enforced by the command.

**Accuracy** is the number of correct predictions divided by all messages.
It treats each message equally. **Macro-F1** treats each official intent
equally: for a label, count true positives (TP: correctly predicted that label),
false positives (FP: predicted that label but the answer was another), and false
negatives (FN: the answer was that label but the prediction was another).
Precision is TP/(TP+FP), recall is TP/(TP+FN), and their harmonic mean simplifies
to `2*TP/(2*TP+FP+FN)`. Use zero when this denominator is zero. Average those
F1 values over **all saved labels**, including absent labels, without weighting
by their message counts.

For targets `[0, 0, 1, 1, 2]` and predictions `[0, 1, 1, 1, 1]`,
accuracy is `3/5 = 0.6`. Label 0 has TP=1, FP=0, FN=1, so F1=`2/3`;
label 1 has TP=2, FP=2, FN=0, so F1=`4/6 = 2/3`; label 2 has TP=0, FP=0,
FN=1, so F1=0. Macro-F1 is `(2/3 + 2/3 + 0)/3 = 4/9`.
With a fourth label absent from both lists it becomes `1/3`, not `4/9`.

## Implement

| Function | Contract |
|---|---|
| `accuracy(targets, predictions)` | Nonempty equal-length lists of integer intent IDs -> fraction correct as a float. Reject empty or unequal lengths with `ValueError`. |
| `macro_f1(targets, predictions, num_labels)` | Same lists and positive label count -> unweighted mean F1 across IDs `0` through `num_labels-1`. Reject invalid lengths, count and out-of-range IDs with `ValueError`; zero F1 for zero denominators. |

Do not call scikit-learn or other metric libraries. Provided evaluation plumbing
validates the completed comparison and `selection.json`, loads its saved
best-validation model and mappings on CPU, then reads test data. It uses ordered
batches, evaluation mode and no gradients, with the same encoding as training.
It does not retrain, rebuild mappings or change the selection.

## Result check

```powershell
python -m banking77.check_evaluation
```

This quick check uses only the worked synthetic example above, perfect and
completely wrong predictions, absent labels, and invalid inputs. It needs no
prepared data, model, network, official test set or training. Initially it
reports `Implement accuracy`; completed work prints `Evaluation checks passed.`

After passing it, evaluate your already selected launch:

```powershell
python -m banking77.evaluate --launch runs/comparison-1 --data-dir data
```

Supply official `categories.json` and `test.csv` in `data`, or add `--download`
to explicitly download missing official files. Selection and saved artifacts
are checked **before** the test loader can read or download anything. Missing,
invalid, mixed or incomplete artifacts fail with a pointer to this checkpoint.
The consistency check uses your current `select_configuration`. If you change
that function after selection, check it and re-run
`python -m banking77.compare select --launch runs/comparison-1` before evaluating;
do not edit `selection.json` by hand.
The command prints the selected configuration, message count, accuracy and
macro-F1. It saves `runs/comparison-1/evaluation.json` with the launch ID,
selected index, run directory and final metrics. The test set never enters the
launch's captured training/validation files. Repeating evaluation is not
technically blocked; follow the one-final-evaluation rule yourself.

Use the official data path for the guided exercise. Do not substitute training,
validation or custom data for final evaluation.

Continue with [Checkpoint 5: Use the classifier](05-use-classifier.md) to label
a separate demonstration message, not a row chosen from the test set.

<details>
<summary>Optional hint</summary>

Use `zip` after checking lengths for accuracy. For each ID, count pairs where
both entries match the ID (TP), only the prediction matches (FP), and only the
target matches (FN). Divide `2*TP` by `2*TP+FP+FN`, or use zero for a zero
denominator, then divide the sum by the full number of labels.

</details>

<details>
<summary>Complete solution</summary>

Replace the two incomplete functions with these implementations. This spoiler
is not executable starter code.

```python
def accuracy(targets: list[int], predictions: list[int]) -> float:
    if not targets or len(targets) != len(predictions):
        raise ValueError("accuracy requires nonempty equal-length inputs")
    return sum(actual == predicted for actual, predicted in
               zip(targets, predictions)) / len(targets)


def macro_f1(targets: list[int], predictions: list[int], num_labels: int) -> float:
    if not targets or len(targets) != len(predictions):
        raise ValueError("macro_f1 requires nonempty equal-length inputs")
    if num_labels <= 0 or any(
        not 0 <= label < num_labels for label in targets + predictions
    ):
        raise ValueError("macro_f1 requires a positive label count and valid IDs")
    total = 0.0
    for label in range(num_labels):
        tp = sum(actual == label and predicted == label
                 for actual, predicted in zip(targets, predictions))
        fp = sum(actual != label and predicted == label
                 for actual, predicted in zip(targets, predictions))
        fn = sum(actual == label and predicted != label
                 for actual, predicted in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        total += 2 * tp / denominator if denominator else 0.0
    return total / num_labels
```

</details>

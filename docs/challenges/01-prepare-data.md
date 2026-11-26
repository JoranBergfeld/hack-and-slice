# Checkpoint 1: Prepare Banking77 data

## Goal

Load Banking77 and implement the five starter functions in `banking77/data.py`:
split the official training messages, tokenize text, build a training-only
vocabulary, encode messages, and map official intent labels to indices.
This checkpoint runs locally on CPU; no Azure account or model training is needed.

## Why this step?

An intent is the category of a banking message, such as `card_arrival`.
Training messages teach the classifier. Validation messages let you compare
training configurations without grading them on the final exam. The official
test set is that final exam: leave it untouched until final evaluation after
model selection.

Stratification keeps each intent's share similar in training and validation.
Seed 42 makes the split repeatable so later comparisons use the same data.
A vocabulary gives each word an integer ID for the classifier's learned
embeddings. Building it from training messages only avoids leaking information
from validation or test messages. Rare and unseen words use `<unk>` (ID 0);
even an empty message needs one token so mean pooling remains defined.

## Setup and data loading

You need Git, Python 3.11 or 3.12, and command-line familiarity, but no prior
machine-learning experience. These versions are supported by the pinned
PyTorch release. From the repository root, create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate` instead.
If PowerShell activation is disabled, run `.\.venv\Scripts\python.exe` instead
of `python` in the commands below; no activation is required.

The loader uses the official
[PolyAI Banking77 files](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).
It downloads only `train.csv` and `categories.json` when you explicitly request
`--download`. It reuses local files on subsequent runs. The official training
set has 10,003 messages across 77 intents. The 3,080 test messages are kept in a
separate `test.csv`: this checkpoint never reads or downloads that file.
Later final evaluation can call `load_test` explicitly.

Read the function docstrings, implement the functions, and run:

```powershell
python -m banking77.check_data
python -m banking77.prepare --download --data-dir data --output prepared
```

Downloading needs internet access once. For offline use, copy the official
`train.csv` and `categories.json` into `data` and omit `--download`:

```powershell
python -m banking77.prepare --data-dir data --output prepared
```

Leave `test.csv` separate. Never call `split_training` on it, build a vocabulary
from it, or use it to choose training settings.

## Implement

- Use `sklearn.model_selection.train_test_split` with `test_size=0.1`,
  `random_state=42`, and `stratify` set to the message labels.
- Lowercase text, then find tokens with the regular expression `[a-z0-9']+`.
  Count occurrences, not just the number of messages containing a word.
- Keep tokens with at least two occurrences **in the training split only**.
  Reserve `<unk>` at ID 0, then assign sorted retained tokens IDs starting at 1.
- Encode unseen tokens as 0. If tokenization produces no tokens, return `[0]`.
- Sort the official labels to assign IDs starting at 0; do not lowercase labels.
  For example, Banking77 includes `Refund_not_showing_up`.

## Result check

`python -m banking77.check_data` uses tiny synthetic messages, no network,
no test set, and no training. Initially it fails with the name of an unfinished
function. After you implement all five functions, it prints `Data checks passed.`
Checks include stratification, reproducibility, training-only word counts,
unknown tokens, empty input, and label ordering.

Official preparation should report **9,002 training / 1,001 validation / 77 intents**.
Vocabulary size depends on the data, not on a required accuracy target.
The output folder contains UTF-8 JSON files for subsequent checkpoints:

| File | Shape |
|---|---|
| `training.json` | List of `{"text": "...", "label": "..."}` messages |
| `validation.json` | Same shape, disjoint validation rows |
| `vocabulary.json` | Object mapping tokens to integer IDs; `<unk>` is 0 |
| `labels.json` | Object mapping official intent labels to integer IDs |
| `metadata.json` | Schema version, seed, input SHA-256, and row counts |

These artifacts contain text and labels, not model parameters. Later training
uses `encode_message` and these same mappings. Save the vocabulary and label
mapping with the selected model for consistent inference. Data and output
folders are ignored by Git. Rerunning preparation replaces these outputs.

For a small synthetic CSV of your own, the plumbing also supports:

```powershell
python -m banking77.prepare --training-csv fixture.csv --categories fixture-labels.json --output prepared
```

The CSV must have `text,category` headers and the labels file is a JSON list of
intent names. This explicit custom-input mode does not enforce official row
counts; include enough examples per intent for a stratified 10% validation split.
It still never reads a test set.

<details>
<summary>Optional hint</summary>

Pass the `Message` objects themselves to `train_test_split`; labels are only the
`stratify` argument. `re.findall` handles tokenization. A `Counter` over all
training tokens handles frequency counting. Use `dict.get(token, 0)` during
encoding, then supply `[0]` if the result is empty.

</details>

<details>
<summary>Complete solution</summary>

Replace the five incomplete functions with these implementations and add the
imports. Keep `Message` and the provided loader code. This is a spoiler, not
executable starter code.

```python
from collections import Counter
import re

from sklearn.model_selection import train_test_split


def split_training(messages: list[Message]) -> tuple[list[Message], list[Message]]:
    training, validation = train_test_split(
        messages,
        test_size=0.1,
        random_state=42,
        stratify=[message.label for message in messages],
    )
    return training, validation


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocabulary(training: list[Message]) -> dict[str, int]:
    counts = Counter(token for message in training for token in tokenize(message.text))
    vocabulary = {"<unk>": 0}
    vocabulary.update(
        (token, index)
        for index, token in enumerate(
            sorted(token for token, count in counts.items() if count >= 2),
            start=1,
        )
    )
    return vocabulary


def encode_message(text: str, vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary.get(token, 0) for token in tokenize(text)] or [0]


def build_label_mapping(labels: list[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(sorted(set(labels)))}
```

</details>

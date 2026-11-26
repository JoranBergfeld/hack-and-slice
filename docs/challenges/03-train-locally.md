# Checkpoint 3: Train a classifier locally

## Goal

Implement the six starter functions in `banking77/training.py`, then train a
from-scratch classifier on your local CPU. Watch its training loss and
validation accuracy each epoch, let early stopping end the run, and keep the
classifier from its best validation epoch. No Azure account, GPU, or official
test set is needed. Compare at least two configurations of your choice, inspect
their chart, and select the best using validation accuracy, never the test set.

## Why this step?

Training repeatedly shows the classifier small **batches** of training
messages. For each batch, the model produces logits, **cross-entropy** measures
how far they are from the correct intents (lower is better), and
backpropagation computes a **gradient** for every weight, including the
embeddings. The **Adam** optimizer then nudges each weight in the direction that
lowers the loss; the **learning rate** sets the size of these nudges. One
**epoch** is one pass over all training messages, shuffled into a new order each
time so the model does not learn the order itself.

Training loss shows whether the model fits the training messages. Validation
accuracy, measured on messages the model never trains on, shows whether it
generalizes. When validation accuracy stops improving for a few epochs (the
**patience**), further training mostly memorizes the training set, so the run
stops early and keeps the checkpoint from its best validation epoch. On equal
validation accuracy the earlier epoch wins: it reached the same result with
less training. The official test set stays untouched until final evaluation.

All runs use seed 42 for weight initialization and shuffling, together with the
same split and preprocessing from [Checkpoint 1](01-prepare-data.md), so
results are repeatable and later comparisons are fair.

## Implement

Complete [Checkpoint 1](01-prepare-data.md) and
[Checkpoint 2](02-build-classifier.md) first. Then implement these contracts
in `banking77/training.py`; the docstrings describe inputs and outputs:

| Function | Contract |
|---|---|
| `build_loader(messages, vocabulary, label_mapping, batch_size, *, shuffle, generator=None)` | `torch.utils.data.DataLoader` yielding `collate_batch` tensors. Shuffle every epoch with `generator` when `shuffle=True`; keep order otherwise. |
| `create_optimizer(model, learning_rate)` | `torch.optim.Adam` over all parameters with default betas and eps and no weight decay. |
| `train_epoch(model, loader, optimizer, device)` | One training pass with cross-entropy on raw logits and one optimizer step per batch. Returns the mean loss per message as a `float`. |
| `validation_accuracy(model, loader, device)` | Fraction of messages whose highest logit is the correct intent, in evaluation mode without gradients, as a `float`. |
| `early_stopping(validation_accuracies, patience)` | `(best_epoch, stop)`: earliest 1-based epoch with the highest accuracy, and whether `patience` epochs have passed without improvement. |
| `select_configuration(results)` | Zero-based index of the highest `best_validation_accuracy`, breaking ties by configuration-list order; reject an empty list with `ValueError`. Provided plumbing validates the entire comparison before calling this function. |

Use PyTorch's built-in cross-entropy and Adam; do not implement
backpropagation or Adam by hand. The provided `python -m banking77.train`
command handles seeding, the epoch loop, progress output, and checkpoint
files, and calls your functions.

## Result check

From the repository root:

```powershell
python -m banking77.check_training
```

The check uses tiny synthetic batches on CPU, with no prepared data, network,
or training run. Initially it names the first unfinished function. When
complete, it prints `Training checks passed.` It covers batching and
reproducible per-epoch shuffling, Adam settings, training mode, per-message loss
averaging, clearing gradients before each step, evaluation mode without
gradients, per-message accuracy, earliest-epoch ties, patience, highest-score
selection, configuration-order ties, and empty-result rejection.

Then train one configuration on your prepared data:

```powershell
python -m banking77.train --prepared prepared --output runs/local
```

Each completed epoch prints a line such as
`Epoch 4/30: training loss 1.2345, validation accuracy 0.7712 (best)`.
After the run stops, a few validation messages are shown with the predicted
and correct intents. The exact numbers depend on your settings; there is no
required accuracy.

### Starting settings

These suggested starting values are defaults, not restrictions. Change any of
them with the matching option, for example `--learning-rate 0.003`:

| Option | Default | Meaning |
|---|---|---|
| `--learning-rate` | `0.001` | Adam step size; `0.0003` and `0.003` are interesting alternatives |
| `--batch-size` | `64` | Training messages per optimizer step |
| `--max-epochs` | `30` | Upper limit on training passes |
| `--patience` | `3` | Epochs without validation improvement before stopping |
| `--embedding-size` | `64` | Learned vector size per token |
| `--hidden-size` | `128` | Hidden-layer features |
| `--dropout` | `0.3` | Fraction of hidden features zeroed during training |
| `--examples` | `5` | Validation messages shown with predictions after training |

Seed 42, the data split, preprocessing, the layer sequence, and the optimizer
stay fixed. Larger models and more epochs take longer on CPU.

### Run outputs

The output folder (`runs/` is ignored by Git) contains everything needed to use
the classifier later:

| File | Contents |
|---|---|
| `best-model.pt` | Best-validation epoch, validation accuracy, model settings, and `state_dict` |
| `vocabulary.json` | The prepared vocabulary used for training |
| `labels.json` | The prepared intent-to-index mapping |
| `run.json` | Configuration, seed, device, per-epoch history, best epoch and validation accuracy, early-stopping outcome, prepared-data identity, software versions, and the example predictions |

`banking77.checkpoint.load_classifier("runs/local")` rebuilds the saved
classifier with its vocabulary and labels on CPU. Training refuses to replace
a completed run: choose a new `--output` for each run. Missing prepared files,
invalid settings, and unfinished functions are reported with a pointer back to
this checkpoint.

## Compare your configurations

Create `configurations.json` in the repository root. This is an editable JSON
list, not a mandatory sweep. For example, start with:

```json
[
  {"learning_rate": 0.0003},
  {"learning_rate": 0.001},
  {"learning_rate": 0.003}
]
```

Use **at least two** entries, with no fixed maximum. Each entry may override
`learning_rate`, `batch_size`, `max_epochs`, `patience`, `embedding_size`,
`hidden_size`, and `dropout`; omitted values use the starting settings above.
You can compare model sizes or patience instead of learning rates. More runs
take more time on CPU and, when Azure training is added, can incur additional
Azure compute charges. This command supports local CPU only; it does not submit
Azure jobs or use a local GPU.

```powershell
python -m banking77.check_training
python -m banking77.compare launch --configurations configurations.json --prepared prepared --output runs/comparison-1
python -m banking77.compare select --launch runs/comparison-1
```

The launch runs each configuration **sequentially**, with live per-epoch loss
and validation output and validation-message predictions. Every run uses the
same architecture sequence, Adam, seed 42 and captured prepared split and
preprocessing. Open `runs/comparison-1/comparison.svg` in a browser to compare
best validation accuracies; each bar identifies its configuration, settings,
score and best epoch. This is not an accuracy target. A tie **within a run**
keeps its earliest best epoch; a tie **between runs** keeps the first
configuration in your list.

The launch captures the current `banking77/**/*.py` project code (including
uncommitted edits), `requirements.txt` when present, normalized configuration
list, example count, software versions, platform and CPU-thread settings, and
prepared artifacts. It executes the
captured code, not the changing working copy. `source/` holds those files and
`prepared/` holds the training/validation data, vocabulary, labels and split
metadata; neither reads the official test set. Keep the launch folder intact.
Edits to your working copy or configurations affect only later launches.

`launch.json` records the launch ID, Git revision (or explicitly `null` outside
a Git checkout), captured-file SHA-256 hashes and aggregate code identity,
data/split hashes and metadata, seed, CPU target, versions, configurations,
completion state, and result-file hashes. Each `runs/0`, `runs/1`, ... directory
contains the same four run outputs as a single training run. Its `run.json`
also records the original launch ID, configuration-list index and captured
source identity alongside settings, history, data identity, software, checkpoint
epoch and score. Git revision alone is insufficient: uncommitted edits change
the captured-code hash without changing the revision.

Selection reads **only the specified launch**, validating every configuration's
completed record, provenance, files, checkpoint and metrics before calling
your `select_configuration`. It refuses partial, failed, missing, corrupt or
mismatched results rather than comparing whichever runs happen to be available.
It uses your current selection function, so you can finish that learning step
after training without retraining. `selection.json` identifies the winner by
launch ID and zero-based configuration index, with its relative run directory
and validation score. Do not tune using the official held-out test set.

Never reuse an output directory, even after a failure: choose a new launch
folder. Completed configurations from a failed launch remain available for
inspection but cannot produce a winner. Interrupted launches remain incomplete.
Recovery/resume snapshots are a later checkpoint, not part of this command.

After selecting, continue with [Checkpoint 4: Final evaluation](04-evaluate-classifier.md).
Only then use the held-out test set; early examples and comparison must remain
validation-only.

<details>
<summary>Optional hint</summary>

Bind the mappings with `functools.partial(collate_batch, vocabulary=...,
label_mapping=...)` and pass it as the `DataLoader`'s `collate_fn`, together
with `shuffle` and `generator`. In `train_epoch`, call `model.train()`, then for
each batch: `optimizer.zero_grad()`, compute
`torch.nn.functional.cross_entropy(logits, targets)`, `loss.backward()`, and
`optimizer.step()`. Multiply each batch's mean loss by its number of messages
before averaging. In `validation_accuracy`, call `model.eval()` and wrap the loop
in `torch.no_grad()`; `logits.argmax(dim=1)` gives each message's prediction.
For `early_stopping`, `list.index(max(...))` finds the earliest best epoch.
For selection, scan the results in order and replace your winner only on a
strictly higher score, or use `max` over their indices (it keeps the first tie).

</details>

<details>
<summary>Complete solution</summary>

Replace the six incomplete functions with these implementations and add the
imports. This spoiler is not executable starter code.

```python
from functools import partial

from banking77.model import collate_batch


def build_loader(
    messages: list[Message],
    vocabulary: dict[str, int],
    label_mapping: dict[str, int],
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        messages,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=partial(
            collate_batch, vocabulary=vocabulary, label_mapping=label_mapping
        ),
    )


def create_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss, total_messages = 0.0, 0
    for tokens, offsets, targets in loader:
        tokens, offsets, targets = tokens.to(device), offsets.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(tokens, offsets), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        total_messages += len(targets)
    return total_loss / total_messages


def validation_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for tokens, offsets, targets in loader:
            logits = model(tokens.to(device), offsets.to(device))
            correct += (logits.argmax(dim=1) == targets.to(device)).sum().item()
            total += len(targets)
    return correct / total


def early_stopping(
    validation_accuracies: list[float], patience: int
) -> tuple[int, bool]:
    if not validation_accuracies:
        raise ValueError("early_stopping requires at least one validation accuracy")
    best_epoch = validation_accuracies.index(max(validation_accuracies)) + 1
    return best_epoch, len(validation_accuracies) - best_epoch >= patience


def select_configuration(results: list[dict]) -> int:
    if not results:
        raise ValueError("select_configuration requires completed results")
    return max(
        range(len(results)),
        key=lambda index: results[index]["best_validation_accuracy"],
    )
```

</details>

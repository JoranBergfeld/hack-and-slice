# Checkpoint 3: Train a classifier locally

## Goal

Implement the five starter functions in `banking77/training.py`, then train one
from-scratch classifier on your local CPU. Watch its training loss and
validation accuracy each epoch, let early stopping end the run, and keep the
classifier from its best validation epoch. No Azure account, GPU, or official
test set is needed. A later checkpoint compares several configurations; this
one trains a single configuration.

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
gradients, per-message accuracy, earliest-epoch ties, and patience.

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

</details>

<details>
<summary>Complete solution</summary>

Replace the five incomplete functions with these implementations and add the
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
```

</details>

# Checkpoint 3: Train a classifier locally or on Azure ML

## Goal

Implement the six starter functions in `banking77/training.py`, then train a
from-scratch classifier on your local CPU or explicitly chosen Azure ML CPU/GPU.
Watch its training loss and
validation accuracy each epoch, let early stopping end the run, and keep the
classifier from its best validation epoch. Local training needs no Azure account,
GPU, or official test set. Compare at least two configurations of your choice, inspect
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
| `latest-snapshot.pt` | Latest resumable snapshot, replaced after every completed epoch (see below) |
| `vocabulary.json` | The prepared vocabulary used for training |
| `labels.json` | The prepared intent-to-index mapping |
| `run.json` | Configuration, seed, device, per-epoch history, best epoch and validation accuracy, early-stopping outcome, prepared-data identity, software versions, and the example predictions |

`banking77.checkpoint.load_classifier("runs/local")` rebuilds the saved
classifier with its vocabulary and labels on CPU. Training refuses to replace
a completed run: choose a new `--output` for each run. Missing prepared files,
invalid settings, and unfinished functions are reported with a pointer back to
this checkpoint.

Two different files are saved, for different purposes:

- `best-model.pt` is the **best-validation checkpoint**. It changes only when
  validation accuracy improves and is what selection, evaluation and prediction
  use.
- `latest-snapshot.pt` is the **latest resumable snapshot**. The provided
  training loop replaces it durably after **every** completed epoch, even when
  validation got worse. It holds everything needed to continue training as if
  nothing happened: model and optimizer state, the completed epoch and history,
  the best epoch and its weights, early-stopping progress, the random-number
  state (data order, dropout), the configuration and the prepared-data identity.
  It is never used to choose or evaluate a model.

Keep both: after a later interruption, the latest snapshot may be at epoch 3
while the best model is still the one from epoch 2.

## Compare your configurations

Edit the provided [`configurations.json`](../../configurations.json) in the
repository root. Its three learning rates (`0.0003`, `0.001`, `0.003`) with
the other defaults are a suggested starting comparison, not a mandatory sweep.

Use **at least two** entries, with no fixed maximum. Each entry may override
`learning_rate`, `batch_size`, `max_epochs`, `patience`, `embedding_size`,
`hidden_size`, and `dropout`; omitted values use the starting settings above.
You can compare model sizes or patience instead of learning rates. More runs
take more time on CPU and can incur additional Azure compute charges.
Every launch requires an explicit `--target local-cpu`
for sequential local training or `--target azure` as described below; omitting
it is an error, not an automatic local run. Local GPU and Apple Silicon are
not supported.

```powershell
python -m banking77.check_training
python -m banking77.compare launch --target local-cpu --configurations configurations.json --prepared prepared --output runs/comparison-1
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
inspection but cannot produce a winner. Interrupted local launches remain
incomplete; explicit recovery of a single failed configuration is available for
Azure launches (see [Recover an interrupted Azure job](#recover-an-interrupted-azure-job)).

## Run the same comparison on Azure ML

Complete the same participant data-loading, model and training functions first.
Azure runs your **captured implementations**, not provided executable solutions.
Preparation still happens locally; only the captured training/validation split,
vocabulary, label mapping and metadata are uploaded, never the held-out test set.
The launch uploads the current captured `banking77` code, including uncommitted
edits, and submits **one independent asynchronous job per configuration**. It
returns names/IDs without waiting for training (upload, authentication and
submission themselves can take time). There is no sweep or fixed run limit.

### Existing workspace and authentication

You need your own existing Azure subscription and Azure ML workspace, with
workspace access such as **Azure ML Data Scientist** or **Contributor**, permission
to access its storage, and available Azure ML compute quota in its region for
your chosen VM family. Ask your facilitator or workspace administrator for the
subscription ID, resource group, workspace name and permitted CPU/GPU VM sizes.
Network restrictions or storage-role restrictions can also prevent jobs from
starting. This helper uses the Azure CLI's authenticated account
(`AzureCliCredential`) and user credential passthrough for job storage access;
it does not switch accounts or fall back to another credential or compute type.

Install the optional SDK locally and sign in using the
[Azure CLI authentication guide](https://learn.microsoft.com/en-us/cli/azure/authenticate-azure-cli-interactively):

```powershell
python -m pip install -r requirements-azure.txt
az login
az account set --subscription YOUR_SUBSCRIPTION_ID
python -m banking77.check_training
```

The SDK is not needed for local training or offline fake-service tests.
For Azure CPU, explicitly supply **both** the device and VM size:

```powershell
python -m banking77.compare launch --target azure --azure-device cpu --instance-type Standard_D2s_v3 --subscription YOUR_SUBSCRIPTION_ID --resource-group YOUR_RESOURCE_GROUP --workspace YOUR_WORKSPACE --configurations configurations.json --prepared prepared --output runs/azure-comparison-1
```

For GPU, use the same command with `--azure-device gpu` and a GPU VM size allowed
in your region, for example `--instance-type Standard_NC4as_T4_v3`. These are
examples, not availability or price promises. The helper checks the workspace's
VM-size catalog against your CPU/GPU choice. It submits **Standard-tier
serverless**, one instance per job; no shared cluster is selected or created.
Missing choices, incompatible VM sizes, authentication, quota and submission
errors are reported. A GPU job without usable CUDA fails rather than training
on CPU. There is no automatic compute fallback.

Each launch captures a conda environment definition: the current Python and
scikit-learn versions, plus the PyTorch version pinned in `requirements.txt`.
Azure uses the CPU PyTorch build or the CUDA 12.6 build for the explicit GPU
choice, on an Azure ML Ubuntu base-image reference. Additional package
specifiers in the captured `requirements.txt` are included. Local-file
requirements, include files and pip options are rejected rather than silently
ignored. Install the pinned requirements locally before launching so local
training and Azure use the same patched PyTorch release. Unsupported versions
or unavailable packages fail environment setup. The base-image reference is
recorded, not a promise of a bit-for-bit immutable OS image. Each collected run
also records actual software, runtime platform/thread information and CUDA
version; CPU/GPU results need not be bit-for-bit identical.

### Inspect, collect, then finish locally

```powershell
python -m banking77.compare status --launch runs/azure-comparison-1
python -m banking77.compare collect --launch runs/azure-comparison-1
```

Both commands show the chosen target, each job's status, its output URI when
assigned, and its Azure ML studio link. Open that link's **Outputs + logs** to
follow live epoch loss/validation progress and service/training errors. The
worker flushes progress each epoch to stdout and `progress.log`; collection
prints that log for newly collected completed runs. Queued jobs may be waiting
for quota, and environment builds can take time before epoch one.

`collect` downloads and validates **completed** jobs even while others are queued,
running, failed or canceled. It reports `Collected 1/3`, for example. Invalid or
mixed outputs are reported and never promoted to a local result; other valid
jobs are still collected. A command with job/status/download/validation errors
returns a nonzero exit code even if some results were successfully collected.
Repeat collection later to pick up newly completed jobs; already validated
results are preserved. The four result files have the same local shape as
local runs, with additional original job, compute, environment and launch
provenance. Keep the entire launch folder intact, including `source/`,
`prepared/`, `submission/`, `recoveries/` and `launch.json`. Changing participant code or your
configuration file after submission does not change these jobs or their
comparison.

Only when **every configuration** has a valid matching result does collection
create `comparison.svg` and allow the existing local selection workflow:

```powershell
python -m banking77.compare select --launch runs/azure-comparison-1
python -m banking77.evaluate --launch runs/azure-comparison-1 --download
```

Finish the participant selection/evaluation functions as described here and in
Checkpoint 4. Selection uses your current local selection function without
retraining. Evaluation loads the selected saved model on local CPU, including
when it was trained on GPU. For Checkpoint 5, pass the winner's run directory
printed by selection as `python -m banking77.predict --model ... --message ...`;
do not assume configuration zero won. Do not evaluate the held-out test set
while choosing configurations.

A failed setup or submission retains its original launch metadata and any
submitted job names; other independent submissions are still attempted. If a
network submission was ambiguous, inspect the recorded job name in your
workspace before doing anything else. Nothing is resubmitted automatically. A
new comparison must use a new output directory and gets a new launch identity.

### Recover an interrupted Azure job

Long jobs can be preempted, hit a node failure or be canceled. Because the
worker writes its latest resumable snapshot to the job's output after every
completed epoch, an interrupted configuration does not have to start over.
Recovery is always **explicit** and affects **only the configuration you name**:

```powershell
python -m banking77.compare status --launch runs/azure-comparison-1
python -m banking77.compare recover --launch runs/azure-comparison-1 --job FAILED_JOB_NAME --resume
```

`--resume` downloads the failed job's outputs, checks that its
`latest-snapshot.pt` belongs to this launch, configuration and prepared data,
prints the last completed epoch and the best validation result so far, and
submits a new job named `...-<index>-r1` (then `-r2`, ...). That job runs the
**same captured code and data** as the original launch, not your current
working copy, and continues from the next epoch with the restored model,
optimizer, best checkpoint, early-stopping progress and randomness. On the same
CPU setup a resumed run matches an uninterrupted one. Use `--restart` instead
to train that configuration again from epoch 1, for example when the job failed
before finishing its first epoch (resume then refuses with "no usable latest
resumable snapshot") or never started.

If a job says `Completed` but `collect` rejects its outputs (for example missing
files or mismatched software), retry collection if the download problem was
temporary. Otherwise explicitly use `recover --restart`, or `--resume` if the
job has a valid latest snapshot. Run `collect` first to record the rejection;
`status` preserves and shows that collection error. A resumed snapshot already
at the final epoch can rebuild the result without more training.

Recovery refuses jobs that are still queued/running, completed without a
recorded collection rejection, already collected, or not the current job of a
configuration in this launch. Valid collected configurations and the original
launch ID are kept;
`launch.json` records each attempt under `recoveries`. Afterwards use `status`
and `collect` as usual; the collected `run.json` records the recovery (mode,
failed job, epoch resumed after, previous device), and each history entry
records the device that trained that epoch.

The recovered job keeps the original CPU/GPU choice and VM size by default.
If GPU quota or capacity is gone you may **explicitly** continue a GPU
configuration on CPU; both options are required and nothing switches silently:

```powershell
python -m banking77.compare recover --launch runs/azure-comparison-1 --job FAILED_JOB_NAME --resume --azure-device cpu --instance-type Standard_D2s_v3
```

That job uses the CPU PyTorch build and CPU compute and is recorded as such.
The result is valid, but there is **no guarantee** that it is bit-for-bit
identical to an uninterrupted GPU run (GPU and CPU arithmetic differ), and the
remaining epochs on CPU can be slower. A recovery job is still paid compute;
cancel it like any other job when no longer needed.

### Quota, charges and cleanup

Read the [serverless compute guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-serverless-compute?view=azureml-api-2),
[Azure ML quotas](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2)
and [Azure ML pricing](https://azure.microsoft.com/en-us/pricing/details/machine-learning/)
before launching. Standard-tier CPU/GPU jobs consume paid compute and regional
workspace/subscription quota. More configurations, larger VMs and longer
training increase charges; storage, environment builds and downloads can also
incur costs. Confirm your region's prices and budget with your administrator.

Cancel **your own named jobs** in Azure ML studio when they are no longer needed,
or use `az ml job cancel --name JOB_NAME --subscription YOUR_SUBSCRIPTION_ID
--resource-group YOUR_RESOURCE_GROUP --workspace-name YOUR_WORKSPACE`.
Follow your workspace's retention policy for your uploaded captures, logs and
outputs. Do not delete shared workspaces, storage accounts, environments or
compute resources. The helper only submits/inspects/downloads jobs; it never
provisions or deletes shared Azure resources.

Automated tests run tiny synthetic data through a fake external Azure service,
including independent submissions, partial collection, failures, hard-killed
(interrupted) jobs with explicit resume/restart, CPU-after-GPU recovery and
local completion. They do **not** verify a live service, GPU training,
authentication, quota or charges. A live smoke check is optional and must be
reported separately.

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

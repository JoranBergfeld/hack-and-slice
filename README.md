# Banking77 guided exercise

Work through the five checkpoints to build, train, evaluate, and use a
classifier for the Banking77 dataset. Starter learning functions are
intentionally incomplete.

## Checkpoints

### 1. Prepare data

[Open Checkpoint 1: Prepare data](docs/challenges/01-prepare-data.md) for local
Python setup, starter-function contracts, offline quick checks, and a collapsed
solution.

### 2. Build the classifier

[Open Checkpoint 2: Build the classifier](docs/challenges/02-build-classifier.md)
to implement learned mean embeddings, configurable model dimensions, and
batching. Check the model without a training run using
`python -m banking77.check_model`.

### 3. Train locally or on Azure ML

[Open Checkpoint 3: Train locally or on Azure ML](docs/challenges/03-train-locally.md)
to implement cross-entropy, Adam, per-epoch validation, and early stopping.
Check the training logic offline with `python -m banking77.check_training`,
then train one configuration on CPU with `python -m banking77.train`. This
saves the classifier with the best validation performance.

The checkpoint also covers participant-written model selection with
`python -m banking77.compare launch` and `select`:

- Edit the provided [`configurations.json`](configurations.json) starting list.
- For local runs, choose `--target local-cpu` to run configurations sequentially
  on CPU with captured code and data. Inspect the SVG comparison chart, then
  select only from a complete, matching launch.
- Alternatively, choose `--target azure` and explicitly select Azure ML CPU or
  GPU. Submit independent Standard-tier serverless jobs with the same captured
  participant code, inspect status and log links, and collect completed outputs
  while other jobs are pending or failed. Then use the same local
  selection/evaluation/prediction workflow.

After every epoch, training saves a latest resumable snapshot, separate from
the best-validation classifier. If an Azure configuration is interrupted, use
`python -m banking77.compare recover` to explicitly resume or restart it,
optionally continuing a GPU configuration on CPU, without touching completed
results. Jobs whose outputs are rejected during collection can also be
explicitly restarted or resumed from a valid snapshot.

The guide covers existing workspace access, authentication, quota, charges,
and cleanup. The helper does not provision or delete shared Azure resources.

### 4. Evaluate the selected classifier

[Open Checkpoint 4: Evaluate the selected classifier](docs/challenges/04-evaluate-classifier.md)
to implement accuracy and macro-F1 by hand and check them with synthetic IDs.
Then use `python -m banking77.evaluate` to evaluate only the selected winner on
the official held-out test set, on CPU. Follow the explained one-final-
evaluation rule; there is no enforced one-use lock.

### 5. Use the classifier

[Open Checkpoint 5: Use the classifier](docs/challenges/05-use-classifier.md)
to implement saved-model prediction, run
`python -m banking77.check_prediction`, and use `python -m banking77.predict`
on a separate demonstration message. Optional investigations include LLM
comparison, inference cost, deployment/registration, Foundry compatibility,
and pretrained embeddings. Documentation links are provided, but no solutions
are included for these investigations.

## Run the tests

Install `requirements.txt`, then run the repository's automated behavior tests:

```powershell
python -m unittest discover -s tests -v
```

The tests use synthetic CSVs and the documented spoiler solution in temporary
workspaces. They require neither network access, the official held-out test set,
nor a full training run; training tests use a few epochs on tiny synthetic data.
Evaluation tests also generate official-shaped synthetic files to exercise the
3,080-row loader boundary; these are not downloaded Banking77 test messages.
Azure tests substitute only the external service boundary and exercise the
captured worker offline. They require no Azure SDK, credentials or paid
services, and do not count as a live Azure smoke check. Real Azure submission
requires the optional `requirements-azure.txt`.

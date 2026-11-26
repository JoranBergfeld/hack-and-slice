# Banking77 guided exercise

Start with [Checkpoint 1: Prepare data](docs/challenges/01-prepare-data.md).
It includes local Python setup, starter-function contracts, offline quick
checks, and a collapsed solution. Starter learning functions are intentionally
incomplete.

Continue with [Checkpoint 2: Build the classifier](docs/challenges/02-build-classifier.md).
It introduces learned mean embeddings, configurable model dimensions, batching,
and `python -m banking77.check_model` without a training run.

Continue with [Checkpoint 3: Train locally](docs/challenges/03-train-locally.md).
It covers cross-entropy, Adam, per-epoch validation and early stopping, checked
offline with `python -m banking77.check_training`, then trains one configuration
on CPU with `python -m banking77.train` and saves its best-validation classifier.

Run the repository's automated behavior tests after installing
`requirements.txt`:

```powershell
python -m unittest discover -s tests -v
```

The tests use synthetic CSVs and the documented spoiler solution in temporary
workspaces. They require neither network access, the official held-out test set,
nor a full training run; training tests use a few epochs on tiny synthetic data.

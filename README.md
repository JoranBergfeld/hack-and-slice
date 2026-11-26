# Banking77 guided exercise

Start with [Checkpoint 1: Prepare data](docs/challenges/01-prepare-data.md).
It includes local Python setup, starter-function contracts, offline quick
checks, and a collapsed solution. Starter learning functions are intentionally
incomplete.

Continue with [Checkpoint 2: Build the classifier](docs/challenges/02-build-classifier.md).
It introduces learned mean embeddings, configurable model dimensions, batching,
and `python -m banking77.check_model` without a training run.

Run the repository's automated behavior tests after installing
`requirements.txt`:

```powershell
python -m unittest discover -s tests -v
```

The tests use synthetic CSVs and the documented spoiler solution in temporary
workspaces. They require neither network access, the official held-out test set,
nor model training.

# Banking77 guided exercise

Start with [Checkpoint 1: Prepare data](docs/challenges/01-prepare-data.md).
It includes local Python setup, starter-function contracts, offline quick
checks, and a collapsed solution. Starter learning functions are intentionally
incomplete.

Continue with [Checkpoint 2: Build the classifier](docs/challenges/02-build-classifier.md).
It introduces learned mean embeddings, configurable model dimensions, batching,
and `python -m banking77.check_model` without a training run.

Continue with [Checkpoint 3: Train locally or on Azure ML](docs/challenges/03-train-locally.md).
It covers cross-entropy, Adam, per-epoch validation and early stopping, checked
offline with `python -m banking77.check_training`, then trains one configuration
on CPU with `python -m banking77.train` and saves its best-validation classifier.
The same checkpoint now includes participant-written selection and
`python -m banking77.compare launch` / `select`: edit a list of configurations,
run them sequentially on CPU with captured code and data, inspect the SVG
comparison chart, then select only from a complete, matching launch.
Alternatively choose Azure ML CPU or GPU explicitly: submit independent
Standard-tier serverless jobs with the same captured participant code, inspect
status/log links, collect completed outputs while other jobs are pending or
failed, then use the same local selection/evaluation/prediction workflow.
The guide covers existing workspace access, authentication, quota, charges and
cleanup; the helper does not provision or delete shared Azure resources.

Continue with [Checkpoint 4: Evaluate the selected classifier](docs/challenges/04-evaluate-classifier.md).
Implement accuracy and macro-F1 by hand, check them with synthetic IDs, then
use `python -m banking77.evaluate` to evaluate only the selected winner on
the official held-out test set on CPU. Follow the explained one-final-evaluation
rule; there is no enforced one-use lock.

Finish with [Checkpoint 5: Use the classifier](docs/challenges/05-use-classifier.md).
Implement saved-model prediction, run `python -m banking77.check_prediction`,
then use `python -m banking77.predict` on a separate demonstration message.
The checkpoint also suggests optional LLM comparison, inference-cost,
deployment/registration, Foundry compatibility and pretrained-embedding
investigations with documentation links and no provided solutions.

Run the repository's automated behavior tests after installing
`requirements.txt`:

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

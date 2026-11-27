# Checkpoint 5: Use your trained classifier

## Goal

Implement `predict_intent` in `banking77/prediction.py` and use the saved
classifier to label a **demonstration message** on local CPU. This is the guided
exercise's finish line: a participant-trained from-scratch classifier producing
an intent for a new message. Finish [final evaluation](04-evaluate-classifier.md)
first; choose a demonstration message separate from the held-out test set.
You can complete this at home; there is no required accuracy.

## Why this step?

Training changed the model's weights; prediction reuses them without learning
anything new. The saved vocabulary translates words into the same IDs as
training, and the saved label mapping translates the highest logit back into
the original official intent name. Rebuilding either mapping can change the
meaning of a weight or an output column, even if their sizes stay the same.

Use the same lowercase tokenization and unknown-token behavior as
[Checkpoint 1](01-prepare-data.md). Empty or punctuation-only text becomes
one unknown token, not an empty tensor. A single message needs a flattened
CPU long tensor of its tokens and one starting offset `[0]`. Logits need no
softmax to find the largest one. `model.eval()` disables training-time dropout;
`torch.no_grad()` avoids tracking gradients. A predicted label is not a
guarantee of correctness, nor does its highest logit provide calibrated confidence.

## Implement

`predict_intent(saved, text) -> str` receives a `SavedClassifier` from
`banking77.checkpoint` and a message string. `saved.model` is on CPU;
`saved.vocabulary` is token-to-ID, and `saved.label_mapping` is intent-to-ID.
Return the original saved intent name whose logit is highest, preserving
capitalization. Use `encode_message`, evaluation mode and no gradients.
Do not retrain, load test data, change mappings or use a pretrained model.

## Result check

```powershell
python -m banking77.check_prediction
```

The offline check uses a tiny synthetic lookup classifier and two synthetic
labels. It checks lowercase encoding, mixed known/unknown tokens, entirely
unknown text, empty text, punctuation-only text, CPU long tensors, the `[0]`
offset, evaluation mode, disabled gradients and original label spelling. It
requires your completed encoding function from Checkpoint 1, but no training,
prepared data, saved model, network, Azure or held-out messages. Unfinished work
names the function and this checkpoint; success prints `Prediction checks passed.`

Find the chosen run in `runs/comparison-1/selection.json` (`run` is relative to
the launch folder). For example, if it says `"run": "runs/1"`:

```powershell
python -m banking77.predict --model runs/comparison-1/runs/1 --message "Why hasn't my refund arrived?"
```

For a comparison run, the command verifies the selected run and loads its model
and encoder from the source captured at launch, while running your current
`predict_intent` implementation. For a standalone run, it loads the artifacts
with the current `banking77.checkpoint.load_classifier`. It prints
`Predicted intent: ...`, uses local CPU, and does not load the test set or write
into training artifacts. Missing files, malformed mappings, incompatible or
non-finite weights, unfinished functions and invalid predicted labels produce
clear errors pointing here.
The command also accepts a saved single-run directory, but for the guided
exercise use the winner you already selected, not a run chosen by test scores.
Demonstration prediction is a separate operation; it does not rerun evaluation.

<details>
<summary>Optional hint</summary>

Call `encode_message(text, saved.vocabulary)`, make a `torch.long` tensor,
and give the model that tensor with `torch.tensor([0], dtype=torch.long)`.
Call `saved.model.eval()` and use `torch.no_grad()`. The largest column in
the output corresponds to a saved label ID; invert `saved.label_mapping`.

</details>

<details>
<summary>Complete solution</summary>

Replace the incomplete function and add these imports. This spoiler is not
executable starter code.

```python
import torch

from banking77.data import encode_message


def predict_intent(saved: SavedClassifier, text: str) -> str:
    tokens = torch.tensor(encode_message(text, saved.vocabulary), dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    saved.model.eval()
    with torch.no_grad():
        logits = saved.model(tokens, offsets)
        if logits.shape != (1, len(saved.label_mapping)) or not torch.isfinite(logits).all():
            raise ValueError("classifier must return finite logits for every intent")
        index = logits.argmax(dim=1).item()
    labels = {identifier: label for label, identifier in saved.label_mapping.items()}
    return labels[index]
```

</details>

## Optional next steps

These are independent recommendations, not required checkpoints. There are
**no provided solutions**, scoring formula or required written report. Hosted
services may incur charges: check access, pricing and cleanup before starting.

| Next step | Goal and reason | Documentation |
|---|---|---|
| LLM comparison | Ask an LLM to choose among the official intents and compare its behavior with your classifier. This explores a different approach and prompt sensitivity. Develop with training/validation messages, not repeated tuning against the held-out set. | [Foundry model catalog](https://learn.microsoft.com/en-us/azure/foundry/concepts/model-catalog), [model evaluation](https://learn.microsoft.com/en-us/azure/foundry/how-to/evaluate-generative-ai-app) |
| Inference-cost investigation | Compare local compute, hosted endpoint uptime and per-token LLM charges for a realistic message volume. Cost and accuracy answer different questions; there is no official combined score for this workshop. | [Azure ML pricing](https://azure.microsoft.com/en-us/pricing/details/machine-learning/), [Foundry pricing](https://azure.microsoft.com/en-us/pricing/details/ai-foundry/), [Azure pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator/) |
| Endpoint deployment and registration | Register the model and its required mappings, then investigate serving predictions through an endpoint. This explores reproducible packaging and operational access beyond a local script. Manage your own resources and remove chargeable deployments when finished. | [Register and manage models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?view=azureml-api-2), [managed online endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2) |
| Foundry custom-model investigation | Check which custom models and import paths Foundry currently supports before assuming compatibility. The documented Fireworks custom-weight import requires a supported base architecture and associated files. It is **not a documented drop-in for this EmbeddingBag/MLP intent classifier**; a PyTorch checkpoint alone does not establish support. | [Fireworks custom-model import requirements](https://learn.microsoft.com/en-us/azure/foundry/how-to/fireworks/import-custom-models) |
| Pretrained embeddings | Investigate replacing exercise-learned embeddings with pretrained word vectors or an embedding model. This explores transfer learning and how representation, preprocessing and classifier inputs may need to change; it is outside the core from-scratch exercise. | [PyTorch pretrained embedding interface](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained), [Sentence Transformers pretrained models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) |

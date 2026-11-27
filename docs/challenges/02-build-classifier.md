# Checkpoint 2: Build the from-scratch classifier

## Goal

Implement `IntentClassifier.__init__`, `IntentClassifier.forward`, and
`collate_batch` in `banking77/model.py`. Start from the vocabulary and label
mapping from [Checkpoint 1](01-prepare-data.md). No training, pretrained encoder,
Azure account, or official dataset download is needed for this checkpoint.

## Why this step?

An embedding is a learned vector for a token ID. `EmbeddingBag(mode="mean")`
averages a message's token vectors, so variable-length messages become vectors
of the same size. The hidden linear layer mixes these features, ReLU keeps
positive values and replaces negative ones with zero, and dropout randomly
zeros some features during training. The final linear layer produces one raw
score, or **logit**, per intent. Later cross-entropy training will learn all these
weights from scratch; do not apply softmax inside the model.

Dropout is active in `model.train()` and disabled in `model.eval()`. Switch to
evaluation mode before inspecting predictions. An untrained model's top-scoring
intent is arbitrary, not evidence of useful accuracy.

## Implement

Install the updated `requirements.txt` in your local Python environment, then
complete the starter contracts. Use these public interfaces for later training:

| Interface | Contract |
|---|---|
| `IntentClassifier(vocabulary_size, embedding_size=64, hidden_size=128, dropout=0.3, num_labels=77)` | Sizes are positive integers; dropout is in `[0, 1]`. Official mappings have 77 labels. |
| `model(tokens, offsets)` | CPU initially, or the model's device later; 1-D long tensors in, raw `[batch_size, num_labels]` logits out. |
| `collate_batch(messages, vocabulary, label_mapping)` | Nonempty list of `Message` objects in; CPU long `(tokens, offsets, targets)` tensors out in message order. Unknown labels are errors. |

Create layers named `embeddings`, `hidden`, `activation`, `dropout`, and `output`,
in that order: `nn.EmbeddingBag`, `nn.Linear`, `nn.ReLU`, `nn.Dropout`,
`nn.Linear`. Use learned embeddings, `mode="mean"` and the default
`include_last_offset=False`. Do not set a padding index: `<unk>` at ID 0 is a
real learned token, including for an empty message. Embedding and hidden sizes
and dropout are configurable starting values, not restrictions.

For batching, reuse `banking77.data.encode_message`. Flatten all encoded token
lists and record where each message starts. For example, `[1, 2]`, `[0]`, and
`[3, 1, 2]` become tokens `[1, 2, 0, 3, 1, 2]` and offsets `[0, 2, 3]`.
There is no final sentinel offset. Preserve the deterministic IDs from
`labels.json`, including official label capitalization. Do not build new
mappings from a batch or read held-out messages.

## Result check

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m banking77.check_model
```

The check uses tiny synthetic inputs on CPU, with no network or training run.
Initially it names the incomplete model step. When complete, it prints
`Model checks passed.` Use the error and the function docstrings to find
unfinished or incorrect work. Finish the encoding functions in Checkpoint 1
first if it names `encode_message` or `tokenize`.

The checks cover mixed message lengths, mean pooling, raw logits, ReLU, unknown
and empty input, learned embeddings receiving gradients, dropout in training
versus evaluation, configurable dimensions, and preservation of label IDs.
A single backward pass checks gradient flow; it does not train the model.

After completing both checkpoints, inspect a small batch of your prepared
messages (the untrained scores are only a shape demonstration):

```python
import json
from pathlib import Path
import torch
from banking77.data import Message
from banking77.model import IntentClassifier, collate_batch

prepared = Path("prepared")
vocabulary = json.loads((prepared / "vocabulary.json").read_text(encoding="utf-8"))
labels = json.loads((prepared / "labels.json").read_text(encoding="utf-8"))
rows = json.loads((prepared / "training.json").read_text(encoding="utf-8"))[:4]
messages = [Message(row["text"], row["label"]) for row in rows]
tokens, offsets, targets = collate_batch(messages, vocabulary, labels)
model = IntentClassifier(len(vocabulary), num_labels=len(labels))
model.eval()
with torch.no_grad():
    logits = model(tokens, offsets)
print(logits.shape, targets.tolist())  # [4, 77] for an official prepared batch
```

Later training can use
`partial(collate_batch, vocabulary=vocabulary, label_mapping=labels)` from
`functools` as a `DataLoader`'s `collate_fn`, then move **all three** batch
tensors to the selected training device. Save model settings with its
`state_dict`, vocabulary, and label mapping so you can reconstruct the same
classifier for local inference. Keep the named layers stable for checkpoint
compatibility.

<details>
<summary>Optional hint</summary>

Call `super().__init__()` before registering layers. Pass the flattened tokens
and starting offsets to the embedding bag, then pass its result through the
remaining layers. Append the current flattened-token length to offsets
**before** extending tokens for each message. Use `torch.long` for tokens,
offsets, and targets.

</details>

<details>
<summary>Complete solution</summary>

Replace the incomplete class and function with the code below. Keep the imports,
and add `encode_message` to the import from `banking77.data`.
This spoiler is not executable starter code.

```python
from banking77.data import encode_message


class IntentClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.3,
        num_labels: int = 77,
    ) -> None:
        super().__init__()
        sizes = (vocabulary_size, embedding_size, hidden_size, num_labels)
        if any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("IntentClassifier sizes must be positive integers")
        if not 0 <= dropout <= 1:
            raise ValueError("IntentClassifier dropout must be in [0, 1]")
        self.embeddings = nn.EmbeddingBag(vocabulary_size, embedding_size, mode="mean")
        self.hidden = nn.Linear(embedding_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(
        self, tokens: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.embeddings(tokens, offsets)
        features = self.dropout(self.activation(self.hidden(pooled)))
        return self.output(features)


def collate_batch(
    messages: list[Message],
    vocabulary: dict[str, int],
    label_mapping: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not messages:
        raise ValueError("collate_batch requires a nonempty message batch")
    tokens, offsets, targets = [], [], []
    for message in messages:
        if message.label not in label_mapping:
            raise ValueError(f"collate_batch: unknown intent label {message.label!r}")
        offsets.append(len(tokens))
        tokens.extend(encode_message(message.text, vocabulary))
        targets.append(label_mapping[message.label])
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )
```

</details>

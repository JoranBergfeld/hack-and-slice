import sys

import torch
from torch import nn

from banking77.data import Message
from banking77.model import IntentClassifier, collate_batch


def check_model() -> None:
    torch.manual_seed(42)
    model = IntentClassifier(
        vocabulary_size=4, embedding_size=2, hidden_size=2, dropout=0.5
    )
    tokens, offsets, targets = collate_batch(
        [
            Message("CARD cash", "cash"),
            Message("unseen", "Refund_not_showing_up"),
            Message("?!", "card"),
            Message("negative", "card"),
        ],
        {"<unk>": 0, "card": 1, "cash": 2, "negative": 3},
        {"Refund_not_showing_up": 0, "card": 1, "cash": 2},
    )
    assert tokens.dtype == offsets.dtype == targets.dtype == torch.long, (
        "collate_batch: tokens, offsets and targets must be long tensors"
    )
    assert tokens.tolist() == [1, 2, 0, 0, 3], (
        "collate_batch: reuse encode_message for unknown and empty input"
    )
    assert offsets.tolist() == [0, 2, 3, 4], (
        "collate_batch: use starting offsets without a final sentinel"
    )
    assert targets.tolist() == [2, 0, 1, 1], (
        "collate_batch: preserve message order and prepared label IDs"
    )
    with torch.no_grad():
        model.embeddings.weight.copy_(torch.tensor([
            [2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [-4.0, -6.0],
        ]))
        model.hidden.weight.copy_(torch.eye(2))
        model.hidden.bias.zero_()
        model.output.weight.zero_()
        model.output.weight[:2].copy_(torch.eye(2))
        model.output.bias.fill_(-2.0)
    model.eval()
    logits = model(tokens, offsets)
    assert logits.shape == (4, 77), (
        "IntentClassifier.forward: return one row per message and 77 intent logits"
    )
    assert torch.isfinite(logits).all(), (
        "IntentClassifier.forward: unknown and empty messages need finite logits"
    )
    assert torch.allclose(logits[:, :2], torch.tensor([
        [3.0, 5.0], [0.0, 2.0], [0.0, 2.0], [-2.0, -2.0],
    ])) and torch.all(logits[:, 2:] == -2), (
        "IntentClassifier.forward: mean pooling -> hidden -> ReLU -> dropout "
        "(disabled in eval) -> raw logits, not softmax"
    )
    assert all(parameter.requires_grad for parameter in model.parameters()), (
        "IntentClassifier: all parameters, including embeddings, must be learned"
    )
    logits.sum().backward()
    assert model.embeddings.weight.grad is not None and torch.any(
        model.embeddings.weight.grad[0] != 0
    ), "IntentClassifier: learned <unk> embeddings must receive gradients"
    assert [type(layer) for layer in model.children()] == [
        nn.EmbeddingBag, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear,
    ], "IntentClassifier: keep EmbeddingBag -> Linear -> ReLU -> dropout -> Linear"
    assert model.embeddings.mode == "mean", (
        "IntentClassifier: EmbeddingBag must use mean mode"
    )
    assert model.embeddings.padding_idx is None, (
        "IntentClassifier: <unk> is learned, not a padding index"
    )
    repeated_tokens = torch.ones(32, dtype=torch.long)
    repeated_offsets = torch.arange(32)
    model.train()
    with torch.no_grad():
        first = model(repeated_tokens, repeated_offsets)
        second = model(repeated_tokens, repeated_offsets)
    assert not torch.equal(first, second), (
        "IntentClassifier.forward: apply dropout during training"
    )
    model.eval()
    with torch.no_grad():
        first = model(repeated_tokens, repeated_offsets)
        second = model(repeated_tokens, repeated_offsets)
    assert torch.equal(first, second), (
        "IntentClassifier.forward: disable dropout during evaluation"
    )
    configurable = IntentClassifier(
        vocabulary_size=4, embedding_size=3, hidden_size=5,
        dropout=0.0, num_labels=3,
    )
    assert configurable(tokens, offsets).shape == (4, 3), (
        "IntentClassifier: honor embedding_size, hidden_size and num_labels"
    )
    assert configurable.embeddings.embedding_dim == 3, (
        "IntentClassifier: honor embedding_size"
    )
    assert configurable.hidden.out_features == 5 and configurable.dropout.p == 0, (
        "IntentClassifier: honor hidden_size and dropout"
    )


def main() -> int:
    try:
        check_model()
    except (
        NotImplementedError, AssertionError, ValueError, RuntimeError,
        KeyError, AttributeError, TypeError, IndexError,
    ) as error:
        print(
            f"Model check failed: {error}. "
            "See docs/challenges/02-build-classifier.md.",
            file=sys.stderr,
        )
        return 1
    print("Model checks passed. 4 messages / 77 intent logits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

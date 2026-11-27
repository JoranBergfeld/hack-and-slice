import sys

import torch
from torch import nn

from banking77.data import Message
from banking77.training import (
    build_loader, create_optimizer, early_stopping, train_epoch,
    validation_accuracy, select_configuration,
)


class _Lookup(nn.Module):
    """Tiny classifier whose logits are each token's row in a fixed table."""

    def __init__(self) -> None:
        super().__init__()
        self.table = nn.EmbeddingBag(3, 3, mode="mean")
        with torch.no_grad():
            self.table.weight.copy_(torch.tensor([
                [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 4.0],
            ]))
        self.modes = []

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        self.modes.append((self.training, torch.is_grad_enabled()))
        return self.table(tokens, offsets)


CPU = torch.device("cpu")


def _check_loader() -> None:
    messages = [Message(f"word{index}", "a" if index % 2 else "b") for index in range(5)]
    vocabulary = {"<unk>": 0, "word0": 1, "word1": 2, "word2": 3, "word3": 4, "word4": 5}
    labels = {"a": 0, "b": 1}
    ordered = build_loader(messages, vocabulary, labels, 2, shuffle=False)
    batches = list(ordered)
    assert [batch[2].tolist() for batch in batches] == [[1, 0], [1, 0], [1]], (
        "build_loader: use batch_size messages per batch and keep order "
        "when shuffle=False"
    )
    assert [batch[0].tolist() for batch in batches] == [[1, 2], [3, 4], [5]] and [
        batch[1].tolist() for batch in batches
    ] == [[0, 1], [0, 1], [0]], (
        "build_loader: batch messages with collate_batch and the prepared mappings"
    )

    def order(loader):
        return [token for batch in loader for token in batch[0].tolist()]

    first = build_loader(
        messages, vocabulary, labels, 2, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    second = build_loader(
        messages, vocabulary, labels, 2, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    epochs = [order(first) for _ in range(3)]
    assert sorted(epochs[0]) == [1, 2, 3, 4, 5], (
        "build_loader: every message must appear once per epoch"
    )
    assert any(epoch != [1, 2, 3, 4, 5] for epoch in epochs) and len(
        {tuple(epoch) for epoch in epochs}
    ) > 1, "build_loader: shuffle=True must reshuffle training messages every epoch"
    assert epochs == [order(second) for _ in range(3)], (
        "build_loader: pass the generator so a seed 42 shuffle repeats exactly"
    )


def _check_optimizer() -> None:
    model = _Lookup()
    optimizer = create_optimizer(model, 0.003)
    assert isinstance(optimizer, torch.optim.Adam), (
        "create_optimizer: use torch.optim.Adam"
    )
    group = optimizer.param_groups[0]
    assert len(optimizer.param_groups) == 1 and [id(p) for p in group["params"]] == [
        id(p) for p in model.parameters()
    ], "create_optimizer: optimize all model parameters"
    assert group["lr"] == 0.003, "create_optimizer: use the configured learning rate"
    assert group["weight_decay"] == 0, "create_optimizer: do not add weight decay"
    assert group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8 and not group[
        "amsgrad"
    ], "create_optimizer: keep Adam's default betas, eps and amsgrad"


def _batches() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        (torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([0, 1])),
        (torch.tensor([1]), torch.tensor([0]), torch.tensor([0])),
    ]


def _check_train_epoch() -> None:
    model = _Lookup()
    model.eval()
    loss = train_epoch(model, _batches(), torch.optim.Adam(model.parameters(), lr=0), CPU)
    assert isinstance(loss, float), "train_epoch: return the loss as a Python float"
    assert model.modes and all(mode == (True, True) for mode in model.modes), (
        "train_epoch: call model.train() so dropout is active, and track gradients"
    )
    assert abs(loss - 0.5560) > 1e-3, (
        "train_epoch: average the loss over messages, not over batches"
    )
    assert abs(loss - 0.736870) < 1e-4, (
        "train_epoch: use cross-entropy on the raw logits and return the mean "
        "loss per training message"
    )
    model = _Lookup()
    train_epoch(model, _batches(), torch.optim.Adam(model.parameters(), lr=0.1), CPU)
    weights = model.table.weight.detach()
    assert not torch.allclose(weights[1], torch.tensor([5.0, 0.0, 0.0])), (
        "train_epoch: backpropagate and call optimizer.step() for every batch"
    )
    assert not torch.allclose(weights[0], torch.tensor([0.2, 0.2, -0.2]), atol=1e-4), (
        "train_epoch: call optimizer.zero_grad() before each backward pass"
    )
    assert torch.allclose(weights, torch.tensor([
        [0.167006, 0.167006, -0.167006],
        [5.074414, -0.074414, -0.074414],
        [0.0, 0.0, 4.0],
    ]), atol=1e-4), "train_epoch: take exactly one optimizer step per batch"


def _check_validation_accuracy() -> None:
    model = _Lookup()
    model.train()
    batches = [
        (torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor([0, 2])),
        (torch.tensor([1]), torch.tensor([0]), torch.tensor([1])),
    ]
    accuracy = validation_accuracy(model, batches, CPU)
    assert isinstance(accuracy, float), (
        "validation_accuracy: return the accuracy as a Python float"
    )
    assert model.modes and all(mode == (False, False) for mode in model.modes), (
        "validation_accuracy: call model.eval() and use torch.no_grad()"
    )
    assert abs(accuracy - 0.5) > 1e-6, (
        "validation_accuracy: count correct messages, not average batch accuracies"
    )
    assert abs(accuracy - 2 / 3) < 1e-6, (
        "validation_accuracy: compare each message's highest logit with its target"
    )


def _check_early_stopping() -> None:
    assert early_stopping([0.1, 0.3, 0.2], 3) == (2, False), (
        "early_stopping: return the 1-based best epoch and keep training "
        "within patience"
    )
    assert early_stopping([0.1, 0.3, 0.3, 0.2, 0.25], 3) == (2, True), (
        "early_stopping: on equal accuracy keep the earliest epoch, and stop after "
        "patience epochs without improvement"
    )
    assert early_stopping([0.5, 0.5], 1) == (1, True), (
        "early_stopping: an equal accuracy is not an improvement; keep the earliest epoch"
    )
    assert early_stopping([0.2, 0.4], 1) == (2, False), (
        "early_stopping: a higher accuracy resets patience"
    )
    try:
        early_stopping([], 3)
    except ValueError:
        pass
    else:
        raise AssertionError("early_stopping: reject an empty accuracy list with ValueError")


def check_training() -> None:
    _check_loader()
    _check_optimizer()
    _check_train_epoch()
    _check_validation_accuracy()
    _check_early_stopping()
    assert select_configuration([
        {"best_validation_accuracy": 0.4}, {"best_validation_accuracy": 0.8},
        {"best_validation_accuracy": 0.6},
    ]) == 1, "select_configuration: choose the highest validation accuracy"
    assert select_configuration([
        {"best_validation_accuracy": 0.8}, {"best_validation_accuracy": 0.8},
        {"best_validation_accuracy": 0.6},
    ]) == 0, "select_configuration: break ties by configuration-list order"
    try:
        select_configuration([])
    except ValueError:
        pass
    else:
        raise AssertionError("select_configuration: reject empty results with ValueError")


def main() -> int:
    try:
        check_training()
    except (
        NotImplementedError, AssertionError, ValueError, RuntimeError,
        KeyError, AttributeError, TypeError, IndexError, ZeroDivisionError,
    ) as error:
        print(
            f"Training check failed: {error}. "
            "See docs/challenges/03-train-locally.md.",
            file=sys.stderr,
        )
        return 1
    print("Training checks passed. Loader, Adam, one epoch, validation, early stopping and selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

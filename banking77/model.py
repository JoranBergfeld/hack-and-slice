import torch
from torch import nn

from banking77.data import Message


class IntentClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.3,
        num_labels: int = 77,
    ) -> None:
        """Create learned mean embeddings -> hidden linear -> ReLU -> dropout
        -> intent linear, with positive sizes and dropout in [0, 1].

        vocabulary_size includes <unk> at ID 0. num_labels is the size of
        the prepared labels mapping (77 for official Banking77).
        """
        super().__init__()
        raise NotImplementedError("Implement IntentClassifier.__init__")

    def forward(
        self, tokens: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Return raw float logits of shape [batch_size, num_labels].

        tokens is a flattened 1-D long tensor of encoded token IDs.
        offsets is a 1-D long tensor of each message's starting position,
        without a final sentinel offset. Both tensors share the model device.
        Do not apply softmax: cross-entropy consumes logits.
        """
        raise NotImplementedError("Implement IntentClassifier.forward")


def collate_batch(
    messages: list[Message],
    vocabulary: dict[str, int],
    label_mapping: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a nonempty message batch using the prepared mappings.

    Return CPU long tensors (flattened tokens, starting offsets, targets).
    Preserve message order; encode_message handles unseen and empty tokens.
    Targets are deterministic label IDs; reject unknown intent labels.
    Can be bound with functools.partial as a DataLoader collate_fn.
    """
    raise NotImplementedError("Implement collate_batch")

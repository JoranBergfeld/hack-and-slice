"""Synthetic demonstration prediction checks, without data or training."""
import sys

import torch
from torch import nn

from banking77.checkpoint import SavedClassifier
from banking77.prediction import predict_intent


class _Lookup(nn.Module):
    def forward(self, tokens, offsets):
        assert not self.training and not torch.is_grad_enabled(), (
            "predict_intent: call model.eval() and use torch.no_grad()"
        )
        assert tokens.device.type == offsets.device.type == "cpu", (
            "predict_intent: run on local CPU"
        )
        assert tokens.dtype == offsets.dtype == torch.long and offsets.tolist() == [0], (
            "predict_intent: use long tokens and a single starting offset [0]"
        )
        assert tokens.ndim == 1 and tokens.numel() > 0, (
            "predict_intent: encode empty text as one unknown token"
        )
        rows = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        return rows[tokens].mean(dim=0, keepdim=True)


def check() -> None:
    saved = SavedClassifier(_Lookup(), {"<unk>": 0, "refund": 1},
                            {"cash": 0, "Refund_not_showing_up": 1}, 1, 0.5)
    for text, expected in (
        ("REFUND!", "Refund_not_showing_up"),
        ("refund neverseen", "Refund_not_showing_up"),
        ("unseen", "cash"),
        ("", "cash"),
        ("?!", "cash"),
    ):
        saved.model.train()
        assert predict_intent(saved, text) == expected, (
            "predict_intent: reuse lowercase preprocessing, unknown/empty encoding "
            "and the original intent label spelling"
        )


def main() -> int:
    try:
        check()
    except (NotImplementedError, AssertionError, ValueError, RuntimeError,
            KeyError, AttributeError, TypeError, IndexError, ZeroDivisionError) as error:
        print(f"Prediction check failed: {error}. "
              "See docs/challenges/05-use-classifier.md.", file=sys.stderr)
        return 1
    print("Prediction checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

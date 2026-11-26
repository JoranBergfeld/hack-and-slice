"""Participant-written final metrics, independent of metric libraries."""


def accuracy(targets: list[int], predictions: list[int]) -> float:
    """Return the fraction of matching IDs as a float in [0, 1].

    Inputs are nonempty equal-length lists of integer intent IDs.
    Reject empty or unequal-length lists with ValueError.
    Do not use metric-library functions.
    """
    raise NotImplementedError("Implement accuracy")


def macro_f1(targets: list[int], predictions: list[int], num_labels: int) -> float:
    """Return the unweighted mean F1 across all IDs 0 .. num_labels-1.

    Inputs are nonempty equal-length integer lists and a positive label count.
    Reject invalid lengths, label counts or out-of-range IDs with ValueError.
    For each label compute 2*TP / (2*TP + FP + FN); use 0 if the denominator
    is zero, including labels absent from both lists. Do not use metric libraries.
    """
    raise NotImplementedError("Implement macro_f1")

"""Participant-written prediction for a separate demonstration message."""
from banking77.checkpoint import SavedClassifier


def predict_intent(saved: SavedClassifier, text: str) -> str:
    """Return the original saved intent name with the highest logit for text.

    saved contains a CPU classifier, training vocabulary and intent-to-ID mapping.
    Use encode_message, including its lowercase, unknown and empty-text behavior.
    Build CPU long tokens and a single starting offset [0]. Use evaluation mode
    and no gradients. Invert the saved mapping without changing label spelling.
    """
    raise NotImplementedError("Implement predict_intent")

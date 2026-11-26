import sys
from collections import Counter

from banking77 import data


def check_data() -> None:
    messages = [
        data.Message(f"message {number}", label)
        for label in ("card", "cash")
        for number in range(30)
    ]
    training, validation = data.split_training(messages)
    assert len(training) == 54 and len(validation) == 6, (
        "split_training: 60 messages must yield 54 training and 6 validation rows"
    )
    assert Counter(training + validation) == Counter(messages), (
        "split_training: every input row must appear exactly once"
    )
    assert not set(training).intersection(validation), (
        "split_training: training and validation must not overlap"
    )
    assert Counter(row.label for row in validation) == {"card": 3, "cash": 3}, (
        "split_training: validation must preserve intent proportions"
    )
    assert (training, validation) == data.split_training(messages), (
        "split_training: repeated calls must return the same split"
    )
    assert [(row.label, row.text) for row in validation] == [
        ("cash", "message 27"), ("cash", "message 28"),
        ("card", "message 19"), ("cash", "message 9"),
        ("card", "message 6"), ("card", "message 14"),
    ], "split_training: use the library's stratified 90/10 split with seed 42"
    assert data.tokenize("CARD, can't! 42 -- Caf\u00e9") == [
        "card", "can't", "42", "caf"
    ], "tokenize: lowercase and use exactly [a-z0-9']+"
    assert data.tokenize("?! \t") == [], "tokenize: punctuation has no tokens"
    vocabulary = data.build_vocabulary([
        data.Message("Shared shared card card once", "card"),
        data.Message("CASH cash shared", "cash"),
    ])
    assert vocabulary == {"<unk>": 0, "card": 1, "cash": 2, "shared": 3}, (
        "build_vocabulary: count token occurrences, keep frequency >= 2, "
        "reserve ID 0 and sort remaining tokens"
    )
    validation_words = [
        data.Message("validationonly validationonly", "card")
    ]
    assert "validationonly" not in vocabulary, (
        "build_vocabulary: validation words must not enter the vocabulary"
    )
    assert data.encode_message(validation_words[0].text, vocabulary) == [0, 0], (
        "encode_message: unseen validation words must map to <unk>"
    )
    assert data.encode_message("CARD mystery", vocabulary) == [1, 0], (
        "encode_message: use the same tokenizer and map unknown tokens to 0"
    )
    assert data.encode_message("", vocabulary) == [0], (
        "encode_message: empty text must produce one <unk> token"
    )
    assert data.encode_message("?!", vocabulary) == [0], (
        "encode_message: an empty token sequence must produce one <unk> token"
    )
    assert data.build_label_mapping(["cash", "Refund_not_showing_up", "card"]) == {
        "Refund_not_showing_up": 0, "card": 1, "cash": 2
    }, "build_label_mapping: sort official labels and preserve their spelling"


def main() -> int:
    try:
        check_data()
    except (NotImplementedError, AssertionError, ValueError) as error:
        print(
            f"Data check failed: {error}. "
            "See docs/challenges/01-prepare-data.md.",
            file=sys.stderr,
        )
        return 1
    print("Data checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

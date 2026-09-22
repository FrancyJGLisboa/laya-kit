"""Turn answers into decisions: a threshold from your own labelled history, and an abstention.

The model is a classifier, not an authority. Two rules carry over from larger systems and are
worth keeping even in a small script:

1. **Abstain by threshold, not by hoping the model refuses.** Measured on 110 labelled decisions,
   Laya's `typed-decisions` checkpoint chose the "cannot tell" option zero times, even when the
   criterion invited it. Every abstention has to come from the confidence gate.
2. **Never let a threshold fall to zero.** On a small sample every confidence band can look
   perfect, and the rule then returns the lowest floor, leaving the decision ungated. That is
   overfitting, not a licence: `MIN_THRESHOLD` floors it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .client import Answer

EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MIN_ACCURACY = 0.97
MIN_SAMPLES = 30
MIN_THRESHOLD = 0.5


@dataclass(frozen=True)
class Gate:
    """What the threshold did to one answer."""
    answer: Answer
    label: str | None          # the decision taken: the model's choice, or None when abstained
    abstained: bool
    threshold: float | None
    reason: str | None = None


def decide(answer: Answer, threshold: float | None) -> Gate:
    """Apply a calibrated threshold. No threshold means abstain: uncalibrated is not permission."""
    if threshold is None:
        return Gate(answer, None, True, None, "uncalibrated")
    confidence = answer.confidence if answer.confidence is not None else 0.0
    if confidence < threshold:
        return Gate(answer, None, True, threshold, f"below_threshold:{threshold}")
    return Gate(answer, answer.choice, False, threshold)


def bins(pairs: Iterable[tuple[float, bool]], edges: Sequence[float] = EDGES) -> dict[str, dict[str, Any]]:
    """{band: {n, acc}} from (confidence, was_correct) pairs."""
    table = {f"{lo:g}-{hi:g}": {"n": 0, "hits": 0} for lo, hi in zip(edges, edges[1:])}
    for confidence, correct in pairs:
        value = 0.0 if confidence is None else float(confidence)
        for lo, hi in zip(edges, edges[1:]):
            if lo <= value < hi or (value >= hi and hi == edges[-1]):
                cell = table[f"{lo:g}-{hi:g}"]
                cell["n"] += 1
                cell["hits"] += int(bool(correct))
                break
    return {key: {"n": cell["n"], "acc": (cell["hits"] / cell["n"]) if cell["n"] else None}
            for key, cell in table.items()}


def calibrate(pairs: Sequence[tuple[float, bool]], *, min_accuracy: float = MIN_ACCURACY,
              min_samples: int = MIN_SAMPLES, min_threshold: float = MIN_THRESHOLD) -> float | None:
    """The lowest confidence band whose cumulative accuracy from the top stays >= min_accuracy.

    Returns None when there are too few labelled examples or no band qualifies: the caller must
    treat that as "abstain", never as "act".
    """
    pairs = list(pairs)
    if len(pairs) < min_samples:
        return None
    table = bins(pairs)
    ordered = sorted(((float(key.split("-")[0]), cell["n"], (cell["acc"] or 0.0) * cell["n"])
                      for key, cell in table.items() if cell["n"]), reverse=True)
    total = hits = 0.0
    best: float | None = None
    for floor, count, hit in ordered:
        total += count
        hits += hit
        if total and hits / total >= min_accuracy:
            best = floor
        else:
            break
    if best is None:
        return None
    return max(best, min_threshold)

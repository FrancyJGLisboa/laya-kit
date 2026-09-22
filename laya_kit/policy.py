"""Turn answers into decisions: a threshold from your own labelled history, and an abstention.

The model is a classifier, not an authority. Two rules carry over from larger systems and are
worth keeping even in a small script:

1. **Abstain by threshold, not by hoping the model refuses.** Measured on 110 labelled decisions,
   Laya's `typed-decisions` checkpoint chose the "cannot tell" option zero times, even when the
   criterion invited it. Every abstention has to come from the confidence gate.
2. **Confidence cannot see out-of-distribution input.** An English checkpoint handed another
   script answers confidently and wrongly -- 0.952 confidence at 0.000 accuracy on Khmer. That
   is checked before the call, and an answer carrying ``out_of_script`` is refused here whatever
   its confidence says.
3. **One threshold per bucket.** Laya fits a temperature per (question type, option count), so a
   confidence means what it says only inside its own bucket -- 1.76 for `choice:3-5` against 1.98
   for `noul:2`. ``calibrate()`` refuses a history that mixes a fitted bucket with an unfitted one
   (`choice:11+`, which the library clamps, or an unfitted checkpoint such as `multilingual`),
   because the two numbers are not the same measurement.
4. **Never let a threshold fall to zero.** On a small sample every confidence band can look
   perfect, and the rule then returns the lowest floor, leaving the decision ungated. That is
   overfitting, not a licence: `MIN_THRESHOLD` floors it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .client import Answer

EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MIN_ACCURACY = 0.97
MIN_SAMPLES = 30
MIN_THRESHOLD = 0.5


class MixedCalibration(ValueError):
    """The labelled history spans buckets whose confidences do not mean the same thing."""


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
    if getattr(answer, "out_of_script", False):
        return Gate(answer, None, True, threshold,
                    f"out_of_script:{(answer.script or 'non-latin').lower()}")
    if threshold is None:
        return Gate(answer, None, True, None, "uncalibrated")
    confidence = answer.confidence if answer.confidence is not None else 0.0
    if confidence < threshold:
        return Gate(answer, None, True, threshold, f"below_threshold:{threshold}")
    return Gate(answer, answer.choice, False, threshold)


def _normalise(pairs: Sequence[tuple[Any, bool]]) -> list[tuple[float, bool]]:
    """Accept (confidence, correct) or (Answer, correct), and refuse a history of mixed buckets."""
    seen: dict[str, bool] = {}
    out: list[tuple[float, bool]] = []
    for item, correct in pairs:
        if isinstance(item, Answer):
            out.append((item.confidence if item.confidence is not None else 0.0, bool(correct)))
            if item.bucket is not None:
                seen[item.bucket] = item.fitted
        else:
            out.append((0.0 if item is None else float(item), bool(correct)))
    fitted = sorted(name for name, ok in seen.items() if ok)
    unfitted = sorted(name for name, ok in seen.items() if not ok)
    if fitted and unfitted:
        raise MixedCalibration(
            f"this history mixes buckets the checkpoint fitted ({', '.join(fitted)}) with buckets "
            f"it did not ({', '.join(unfitted)}); their confidences are not the same measurement. "
            f"Calibrate one threshold per bucket."
        )
    if len(seen) > 1:
        warnings.warn(
            f"calibrating across {len(seen)} buckets ({', '.join(sorted(seen))}): each carries its "
            f"own temperature, so one threshold over all of them is coarser than one per bucket.",
            UserWarning,
            stacklevel=3,
        )
    return out


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


def calibrate(pairs: Sequence[tuple[Any, bool]], *, min_accuracy: float = MIN_ACCURACY,
              min_samples: int = MIN_SAMPLES, min_threshold: float = MIN_THRESHOLD) -> float | None:
    """The lowest confidence band whose cumulative accuracy from the top stays >= min_accuracy.

    Pairs are ``(confidence, was_correct)`` or ``(Answer, was_correct)``. Passing answers lets
    this refuse a history whose buckets do not share a scale; raw floats cannot be checked.

    Returns None when there are too few labelled examples or no band qualifies: the caller must
    treat that as "abstain", never as "act".
    """
    pairs = _normalise(list(pairs))
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

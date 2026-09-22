"""The model boundary: build a question, send it, get a typed answer back.

Laya (convaiinnovations/laya, Apache-2.0) runs in process. Its context is
512-1024 tokens, so this module sends one question per call and compacts the
state. Answers come back in the same shape TypeSafe System One returns, so code
written against one runs against the other.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

REPO = "convaiinnovations/laya"
CHECKPOINTS = {"typed-decisions", "multilingual"}   # anything else loads the default (english)
DEFAULT_CHECKPOINT = os.environ.get("LAYA_CHECKPOINT", "typed-decisions")
MAX_STATE_CHARS = int(os.environ.get("LAYA_MAX_STATE_CHARS", "2000"))

_LOCK = threading.Lock()
_AGENTS: dict[str, Any] = {}


class LayaUnavailable(RuntimeError):
    """The Laya package is not importable in this interpreter."""


def load(checkpoint: str = DEFAULT_CHECKPOINT) -> Any:
    """Load (once) and return the underlying Laya agent."""
    with _LOCK:
        agent = _AGENTS.get(checkpoint)
        if agent is None:
            try:
                import laya  # type: ignore
            except ImportError as exc:
                raise LayaUnavailable(
                    'laya is not installed in this interpreter: pip install laya, and on Intel macOS '
                    'use a python3.11 venv with torch==2.2.2 transformers==4.48.3 "numpy<2"'
                ) from exc
            agent = laya.load(REPO, subfolder=checkpoint if checkpoint in CHECKPOINTS else None)
            _AGENTS[checkpoint] = agent
    return agent


def warm(checkpoint: str = DEFAULT_CHECKPOINT) -> float:
    """Load the model now so the first real call is not slow. Returns seconds taken."""
    started = time.perf_counter()
    load(checkpoint)
    return time.perf_counter() - started


# --- questions ---------------------------------------------------------------------------

def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
    """One of N options. Give every option a concrete criterion, and include a no-match option.

    Keep each criterion to a line: the model reads ~512-1024 tokens in total.
    """
    if len(criteria) < 2:
        raise ValueError("a choice needs at least two options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def noul(instructions: str) -> dict[str, Any]:
    """A yes/no judgement returned as a probability. Use several rather than one N-way choice
    when the answers are not mutually exclusive."""
    return {"type": "noul", "instructions": instructions}


def score(instructions: str, levels: Iterable[str]) -> dict[str, Any]:
    """An ordered scale. Each level is a concrete situation, not an adjective."""
    levels = list(levels)
    if len(levels) < 2:
        raise ValueError("a score needs at least two levels")
    return {"type": "score", "instructions": instructions, "criteria": levels}


# --- answers -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Answer:
    """One typed judgement.

    ``confidence`` is the winning probability, which means the same thing across providers.
    Laya's own calibrated margin is kept as ``native_confidence``; it tops out well below 1,
    so never threshold on it.
    """
    question_id: str
    type: str
    choice: str | None = None
    value: float | None = None                    # noul probability, or score level index
    confidence: float | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    native_confidence: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def above(self, threshold: float) -> bool:
        return self.confidence is not None and self.confidence >= threshold


def _compact(state: Any, max_chars: int) -> Any:
    """Shrink the state to fit a small context: strings truncate, mappings drop trailing keys."""
    if isinstance(state, str):
        return state[:max_chars]
    text = json.dumps(state, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return state
    if isinstance(state, Mapping):
        out: dict[str, Any] = {}
        budget = max_chars
        for key, value in state.items():
            piece = json.dumps(value, ensure_ascii=False, default=str)
            if len(piece) > budget:
                out[key] = piece[: max(0, budget)]
                break
            out[key] = value
            budget -= len(piece) + len(str(key)) + 4
            if budget <= 0:
                break
        return out
    return text[:max_chars]


def _answer(question_id: str, question: Mapping[str, Any], raw: Mapping[str, Any]) -> Answer:
    kind = question.get("type", "choice")
    probabilities = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
    native = raw.get("confidence")
    confidence = max(probabilities.values()) if probabilities else None
    value = None
    picked = raw.get("choice")
    if kind == "noul":
        value = float(raw.get("noul")) if raw.get("noul") is not None else None
        if value is not None:
            picked = "yes" if value >= 0.5 else "no"
            confidence = value if value >= 0.5 else 1.0 - value
    elif kind == "score":
        value = float(raw.get("score")) if raw.get("score") is not None else None
        levels = question.get("criteria") or []
        if value is not None and levels:
            picked = str(int(max(0, min(len(levels) - 1, round(value)))))
    return Answer(question_id=question_id, type=kind, choice=picked, value=value,
                  confidence=confidence, probabilities=probabilities,
                  native_confidence=float(native) if native is not None else None, raw=dict(raw))


def ask(state: Any, questions: Mapping[str, Mapping[str, Any]], *,
        checkpoint: str = DEFAULT_CHECKPOINT, max_state_chars: int = MAX_STATE_CHARS) -> dict[str, Answer]:
    """Ask every question about one state. One model call per question, as Laya prefers."""
    agent = load(checkpoint)
    compact = _compact(state, max_state_chars)
    answers: dict[str, Answer] = {}
    for question_id, question in questions.items():
        with _LOCK:
            result = agent.predict(compact, {question_id: dict(question)})
        raw = (result.get("answers") or {}).get(question_id)
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"laya returned no answer for {question_id!r}")
        answers[question_id] = _answer(question_id, question, raw)
    return answers


class Agent:
    """A reusable handle when you ask the same questions of many items."""

    def __init__(self, questions: Mapping[str, Mapping[str, Any]], *,
                 checkpoint: str = DEFAULT_CHECKPOINT, max_state_chars: int = MAX_STATE_CHARS) -> None:
        self.questions = dict(questions)
        self.checkpoint = checkpoint
        self.max_state_chars = max_state_chars

    def __call__(self, state: Any) -> dict[str, Answer]:
        return ask(state, self.questions, checkpoint=self.checkpoint, max_state_chars=self.max_state_chars)

    def batch(self, states: Iterable[Any], *, on_item: Callable[[int, dict[str, Answer]], None] | None = None) -> list[dict[str, Answer]]:
        """Run over many items in order. Expect ~1.5 s per question per item on a CPU."""
        out = []
        for index, state in enumerate(states):
            answers = self(state)
            if on_item:
                on_item(index, answers)
            out.append(answers)
        return out

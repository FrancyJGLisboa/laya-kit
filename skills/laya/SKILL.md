---
name: laya
license: MIT
description: >
  Build with Laya: a small typed-decision classifier that runs in process, with no
  API key and no network. It answers Choice, Noul and Score questions about a state
  and returns a probability, the same shape TypeSafe System One returns, so a design
  ports between them. Use when a feature needs a bounded judgment and the data cannot
  leave the machine, the budget is zero, or an offline or high-volume path is wanted.
  Applications include routing, triage, extraction gates, evidence checks and
  screening. Not for text generation or multi-step reasoning.
---

# Build with Laya

Laya (`convaiinnovations/laya`, Apache-2.0) is a System-One-shaped classifier that runs
locally: `laya.load(...).predict(state, questions)` returns typed answers with
probabilities. No key, no request, no per-call cost. The model is ~2.2 GB and lives in
`~/.cache/huggingface` after the first run.

Reach for it when the judgment is bounded **and** one of these is true: evidence cannot
leave the network, cost per call must be zero, the workload runs offline, or volume makes
an API impractical. When none of those hold, a hosted classifier is usually more accurate
(see the measured gap below) and this skill still applies: write the design the same way
and switch the transport.

Not a fit: generating text, multi-step reasoning, or anything a regex, lookup or existing
rule already answers.

## Use laya-kit, don't hand-roll the call

`laya-kit` (https://github.com/FrancyJGLisboa/laya-kit, MIT) wraps the model boundary and
the confidence gate. Prefer it over calling `laya` directly: the raw API has two traps
this package already handles (confidence scale, one-question-per-call).

```python
from laya_kit import Agent, calibrate, choice, decide, noul, score

triage = Agent({
    "kind": choice("Read the customer message. What is it asking for? Pick one.", {
        "bug":     "Says something is broken, erroring, or behaving wrongly.",
        "billing": "About an invoice, a charge, a refund, or a plan change.",
        "howto":   "Asks how to do something that already works.",
        "unclear": "Not enough in the message to tell. Pick this whenever two readings fit.",
    }),
    "urgent": noul("The customer says their work is blocked right now."),
})

answers = triage(message)                     # one model call per question
gate = decide(answers["kind"], threshold)     # threshold from calibrate(history)
gate.label                                     # the decision, or None
gate.abstained                                 # True -> a person decides this one
```

Install: `pip install -e ".[model]"`. On **Intel macOS** the chain is pinned because
PyTorch stopped shipping Intel builds after 2.2.2 — use a Python 3.11 venv and the
`model-intel-macos` extra (`torch==2.2.2`, `transformers==4.48.3`, `numpy<2`). Loading
costs ~20 s once per process (`laya_kit.warm()`); after that ~1.5 s per question per item
on a laptop CPU, milliseconds on a GPU.

## The three primitives

| | Use it for | Read back |
|---|---|---|
| `choice(instructions, criteria)` | one of N mutually exclusive options | `.choice`, `.probabilities` |
| `noul(instructions)` | a yes/no judgment | `.value` (probability), `.choice` |
| `score(instructions, levels)` | an ordered scale | `.value`, `.choice` (level index) |

Every answer carries `.confidence`, **the winning probability**. That is the one number
that means the same thing across providers, so a threshold calibrated here transfers in
meaning to a hosted classifier.

## Five rules, each one measured

These came from running this model over 110 labelled decisions in a real internal-control
test, not from documentation. Ignoring any of them produces a demo that does not survive
contact with data.

1. **Write short, contrastive criteria.** The context is 512–1024 tokens. Cutting criteria
   from ~900 characters to ~330 moved raw agreement from 90.0% to 93.6% and nearly doubled
   how many decisions cleared their threshold. One dimension per question, one question per
   call, a concrete situation per option, and always a no-match option.
2. **The model will not abstain for you.** Given a "cannot tell" option whose criterion
   invited it, the `typed-decisions` checkpoint chose it **zero times in 110 decisions**.
   Design the abstention as a confidence gate, never as an option you hope it picks.
3. **Never threshold on the native confidence.** Laya's own field is a calibrated margin
   that tops out near 0.3; every record lands in the lowest bin and nothing ever
   calibrates. Use the winning probability, which `laya-kit` already reports as
   `.confidence` (the native value is kept as `.native_confidence`).
4. **Never let a calibrated threshold reach zero.** On a small sample every confidence band
   can look perfect, and the naive rule then returns the lowest floor, leaving the decision
   ungated — which is exactly where the only wrong conclusion of that project appeared.
   `calibrate()` floors at 0.5. A gate that calibration can switch off is not a gate.
5. **`typed-decisions` is the checkpoint.** The general `english` one scored 78.2% against
   93.6% on the same questions. Set `LAYA_CHECKPOINT` only to compare deliberately.

## How to design a feature with it

1. **Name the decision.** Who acts differently on a hit, and when? If nothing changes, this
   is a curiosity, not a feature.
2. **Bound the answers.** List the options a person would accept, plus the honest no-match
   option. If the list cannot be written down, this is not a classifier problem.
3. **Split what code already knows.** Dates, counts, lookups, permissions and thresholds
   stay in code. The model gets only the part that requires reading a sentence.
4. **Write the questions.** One dimension each; several Nouls rather than one N-way Choice
   when the labels are not mutually exclusive; name the baseline of any comparison
   explicitly ("versus the author's own previous forecast", not "versus before").
5. **Decide what happens below the bar.** Route to a person, fall back to the old path, or
   ask for more input — but decide it before measuring, not after.
6. **Label a few dozen cases and calibrate.** `calibrate(history)` returns the threshold, or
   `None` when the evidence is too thin. `None` means abstain; it never means act.
7. **Report both numbers.** Raw agreement *and* what happened after the gate. The second is
   the one a reviewer cares about: how many decisions were concluded, and how many were
   wrong.

## What to expect from it

One measured domain, 110 labelled decisions: **93.6%** raw agreement against **97.3%** for
a hosted classifier on the same questions. After calibration both produced **zero wrong
conclusions**; the local model concluded 62 of 110 decisions where the hosted one concluded
105. Privacy and zero cost buy less coverage, not less reliability — a trade worth stating
plainly to whoever is deciding.

That is one domain, on text written for the test. For a new problem the answer is a
labelled run, not an extrapolation. Say so rather than quoting these numbers as a forecast.

## Going further

For a full decision system — an evidence-graded action vocabulary, a generated adapter,
preconditions re-checked before any side effect, a decision log and a release gate a human
answers — see [action-vocabulary-forge](https://github.com/FrancyJGLisboa/action-vocabulary-forge),
which treats a local or hosted classifier as an interchangeable provider. `laya-kit` is its
minimal cousin: the same discipline, small enough for one script.

For the hosted side of the same programming model, use the `typesafe-ai` skill. A design
written against either one ports to the other; only the transport changes.

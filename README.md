# laya-kit

Typed judgements from a classifier that runs on your machine. Three primitives — `choice`,
`noul`, `score` — a confidence you can threshold, and a threshold you calibrate on your own
labelled decisions. No API key, no network after the model is cached, no per-call cost.

```python
from laya_kit import Agent, calibrate, choice, decide, noul

triage = Agent({
    "kind": choice("Read the customer message. What is it asking for? Pick one.", {
        "bug":     "Says something is broken, erroring, or behaving wrongly.",
        "billing": "About an invoice, a charge, a refund, or a plan change.",
        "howto":   "Asks how to do something that already works.",
        "unclear": "Not enough in the message to tell. Pick this whenever two readings fit.",
    }),
    "urgent": noul("The customer says their work is blocked right now."),
})

answers   = triage("The export button returns a 500 and nobody can pull reports.")
threshold = calibrate(my_labelled_history)      # (confidence, was_correct) pairs
gate      = decide(answers["kind"], threshold)

gate.label      # "bug", or None when the gate did not accept the answer
gate.abstained  # True -> a person decides this one
```

## Install

```bash
git clone https://github.com/FrancyJGLisboa/laya-kit && cd laya-kit
python3 -m venv .venv && .venv/bin/pip install -e ".[model]"
.venv/bin/python examples/triage.py
```

On **Intel macOS** the whole chain is pinned, because PyTorch stopped shipping Intel builds after
2.2.2: use a Python 3.11 venv and `pip install -e ".[model-intel-macos]"`. The first run downloads
~2.2 GB to `~/.cache/huggingface`; everything after that is offline. Loading the model takes ~20 s
once per process, so call `laya_kit.warm()` at startup if that matters.

The pure-Python half (questions, gate, calibration) has no dependencies at all, so the tests run
anywhere: `python3 -m unittest discover -s tests -t .`

## The three primitives

| | Use it for | You get back |
|---|---|---|
| `choice(instructions, criteria)` | one of N options, mutually exclusive | `.choice`, `.probabilities` |
| `noul(instructions)` | a yes/no judgement | `.value` (probability), `.choice` (`"yes"`/`"no"`) |
| `score(instructions, levels)` | an ordered scale | `.value`, `.choice` (level index) |

Every answer carries `.confidence`, which is **the winning probability** — the one number that
means the same thing across providers, so code written here runs unchanged against a hosted
classifier. The model's own calibrated margin is kept as `.native_confidence`; it tops out well
below 1, so never threshold on it.

## Four things learned the hard way

These came from running this model over 110 labelled decisions in a real project, not from the
docs. They are the difference between a demo and something you can rely on.

1. **Write short, contrastive criteria.** The context is 512–1024 tokens. Cutting criteria from
   ~900 characters to ~330 moved raw agreement from 90.0% to 93.6% and nearly doubled how many
   decisions cleared their threshold. One question per call, which is what `ask()` does.
2. **The model will not abstain for you.** Given a "cannot tell" option with a criterion inviting
   it, this checkpoint chose it **zero times in 110 decisions**. Every abstention has to come from
   the confidence gate. Design for that.
3. **Never let a threshold reach zero.** On a small sample every confidence band can look perfect,
   and the naive rule then returns the lowest floor, leaving the decision ungated — which is
   exactly where the only wrong conclusion of that project appeared. `calibrate()` floors at 0.5.
4. **`typed-decisions` is the checkpoint to use.** The general `english` one scored 78.2% against
   93.6% on the same questions. Set `LAYA_CHECKPOINT` if you want to compare for yourself.

## Scripts

The English checkpoints do not fail quietly off their script — they fail confidently. On Khmer
the published model reported 0.952 confidence at 0.000 accuracy, which no confidence gate can
catch. So `ask()` measures the script of the state before the call, marks every answer
`out_of_script`, and `decide()` abstains regardless of the threshold:

```python
answers = ask(khmer_text, questions)          # warns
answers["kind"].out_of_script                 # True
decide(answers["kind"], 0.8).reason           # "out_of_script:khmer"
```

Use `checkpoint="multilingual"` for that text and calibrate it separately: a threshold fitted on
English does not transfer. `non_latin(text)` is the check on its own, and `LAYA_NON_LATIN_LIMIT`
(0.2) and `LAYA_SCRIPT_GUARD=off` tune or disable it.

## How good is it

One measured domain, 110 labelled decisions: **93.6%** raw agreement against **97.3%** for a
hosted classifier on the same questions. After calibration both produced **zero wrong
conclusions**; the local model simply concluded 62 of 110 decisions where the hosted one concluded
105. Privacy and zero cost buy you less coverage, not less reliability.

That is one domain, on text written for the test. For your problem the answer is a labelled run,
not an extrapolation — which is what `calibrate()` is for.

Speed: about 1.5 s per question per item on a laptop CPU, milliseconds on a GPU. Model:
[Laya](https://huggingface.co/convaiinnovations/laya), Apache-2.0.

## As an agent skill

`skills/laya/SKILL.md` teaches Claude Code, Codex, Copilot and Gemini when a local classifier
is the right tool, how to write questions for a 512–1024 token context, and the seven rules
below. Install it alongside the package:

```bash
./scripts/install.sh          # symlinks into every CLI found (--copy, --uninstall)
```

Then ask the agent for a feature that needs a bounded judgment and it will reach for `/laya`.
The companion skill for the hosted side of the same programming model is `typesafe-ai`; a
design written against either ports to the other.

## Layout

```
laya_kit/client.py   the model boundary: questions in, typed answers out
laya_kit/policy.py   the gate: calibrate a threshold, decide or abstain
examples/triage.py   a working tool in 70 lines
skills/laya/         the agent skill (/laya), installed by scripts/install.sh
tests/               14 tests, no model needed
```

For a full decision system — evidence-graded action vocabularies, generated adapters, release
gates — see [action-vocabulary-forge](https://github.com/FrancyJGLisboa/action-vocabulary-forge),
which this kit is the minimal cousin of.

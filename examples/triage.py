#!/usr/bin/env python3
"""Smallest useful thing: triage support messages with a local model, gated by a threshold.

    python3 examples/triage.py            # decide six messages, showing what the gate did
    python3 examples/triage.py --calibrate # what a labelled history would set the threshold to

Nothing here reaches the network after the model is cached. Swap the questions and the labels
and this is your own tool.
"""
from __future__ import annotations

import argparse
import sys

from laya_kit import Agent, calibrate, choice, decide, noul

QUESTIONS = {
    "kind": choice(
        "Read the customer message. What is it asking for? Pick one.",
        {
            "bug": "Says something is broken, erroring, or behaving wrongly.",
            "billing": "About an invoice, a charge, a refund, or a plan change.",
            "howto": "Asks how to do something that already works.",
            "unclear": "Not enough in the message to tell. Pick this whenever two readings fit.",
        },
    ),
    "urgent": noul("The customer says their work is blocked right now."),
}

MESSAGES = [
    "The export button returns a 500 since this morning and nobody on my team can pull reports.",
    "I was charged twice for March, can you refund one of them?",
    "Where do I change the timezone on my account?",
    "hi",
    "Invoice 8821 looks wrong and the API is also timing out, we are stuck before a board meeting.",
    "It doesn't work.",
]

# Stand-in for a labelled history. In a real tool these come from decisions a person already made.
HISTORY = [(0.91, True), (0.88, True), (0.95, True), (0.72, True), (0.63, False), (0.58, False),
           (0.94, True), (0.86, True), (0.79, True), (0.55, False), (0.61, True), (0.83, True),
           (0.92, True), (0.47, False), (0.68, True), (0.90, True), (0.75, True), (0.52, False),
           (0.97, True), (0.81, True), (0.66, False), (0.89, True), (0.93, True), (0.59, False),
           (0.84, True), (0.71, True), (0.96, True), (0.64, False), (0.87, True), (0.78, True)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true", help="show the threshold the history implies")
    parser.add_argument("--threshold", type=float, default=None, help="override the threshold")
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else calibrate(HISTORY)
    if args.calibrate:
        print(f"threshold from {len(HISTORY)} labelled decisions: {threshold}")
        print("(None would mean: not enough evidence, so every answer abstains)")
        return 0

    agent = Agent(QUESTIONS)
    print(f"threshold {threshold}   ·   local model, nothing leaves this machine\n")
    routed = blocked = 0
    for message in MESSAGES:
        answers = agent(message)
        gate = decide(answers["kind"], threshold)
        urgent = answers["urgent"]
        where = gate.label if not gate.abstained else "HUMAN"
        if gate.abstained:
            blocked += 1
        else:
            routed += 1
        flag = " [blocked]" if urgent.value and urgent.value >= 0.6 else ""
        print(f"{where:8} {answers['kind'].confidence:.2f}{flag}  {message[:64]}")
        if gate.abstained:
            print(f"         ^ {gate.reason}: the model said {answers['kind'].choice!r}, the gate did not accept it")
    print(f"\nrouted {routed}, sent to a person {blocked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

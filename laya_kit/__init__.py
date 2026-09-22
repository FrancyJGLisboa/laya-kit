"""laya-kit: one local classifier, three typed primitives, no API.

    from laya_kit import ask, choice, noul, score

    answers = ask("The reviewer wrote: I reviewed all listed access for Q1.", {
        "attested": choice("Did the reviewer say the access was already reviewed?", {
            "yes": "Says it was reviewed. Past tense, by this reviewer, all of it.",
            "no": "Says it will be reviewed, or someone else did, or only part.",
            "unclear": "Too short or vague to tell.",
        }),
    })
    answers["attested"].choice        # "yes"
    answers["attested"].confidence    # winning probability, comparable across providers
"""
from .client import Agent, Answer, ask, choice, noul, score, load, warm
from .policy import Gate, calibrate, decide

__all__ = ["Agent", "Answer", "ask", "choice", "noul", "score", "load", "warm",
           "Gate", "calibrate", "decide"]
__version__ = "0.1.0"

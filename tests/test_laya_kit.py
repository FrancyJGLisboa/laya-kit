import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laya_kit import Agent, ask, calibrate, choice, decide, load, non_latin, noul, score  # noqa: E402
from laya_kit import client, policy  # noqa: E402


class FakeAgent:
    def __init__(self, answers=None):
        self.calls = []
        self.answers = answers or {}

    def predict(self, state, questions):
        (qid, q), = questions.items()
        self.calls.append((state, qid))
        if qid in self.answers:
            return {"answers": {qid: self.answers[qid]}}
        first = next(iter(q["criteria"])) if isinstance(q.get("criteria"), dict) else None
        return {"answers": {qid: {"choice": first, "confidence": 0.12,
                                  "probabilities": {first: 0.7, "other": 0.3}}}}


def fake_laya(agent):
    client._AGENTS.clear()
    return mock.patch.dict(sys.modules, {"laya": types.SimpleNamespace(load=lambda repo, subfolder=None: agent)})


class QuestionTests(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(choice("q", {"a": "x", "b": "y"})["type"], "choice")
        self.assertEqual(noul("q")["type"], "noul")
        self.assertEqual(score("q", ["low", "high"])["criteria"], ["low", "high"])
        with self.assertRaises(ValueError):
            choice("q", {"only": "one"})
        with self.assertRaises(ValueError):
            score("q", ["only"])


class AnswerTests(unittest.TestCase):
    def test_confidence_is_the_winning_probability_not_the_native_margin(self):
        agent = FakeAgent()
        with fake_laya(agent):
            answers = ask("text", {"k": choice("q", {"a": "x", "b": "y"})})
        answer = answers["k"]
        self.assertEqual(answer.choice, "a")
        self.assertAlmostEqual(answer.confidence, 0.7)
        self.assertAlmostEqual(answer.native_confidence, 0.12)
        self.assertTrue(answer.above(0.6))
        self.assertFalse(answer.above(0.8))

    def test_noul_answer_is_folded_to_yes_no(self):
        agent = FakeAgent({"u": {"noul": 0.82}})
        with fake_laya(agent):
            answers = ask("text", {"u": noul("blocked?")})
        self.assertEqual(answers["u"].choice, "yes")
        self.assertAlmostEqual(answers["u"].confidence, 0.82)
        agent = FakeAgent({"u": {"noul": 0.1}})
        with fake_laya(agent):
            answers = ask("text", {"u": noul("blocked?")})
        self.assertEqual(answers["u"].choice, "no")
        self.assertAlmostEqual(answers["u"].confidence, 0.9)

    def test_score_answer_maps_to_a_level(self):
        agent = FakeAgent({"s": {"score": 1.4}})
        with fake_laya(agent):
            answers = ask("text", {"s": score("q", ["low", "mid", "high"])})
        self.assertEqual(answers["s"].choice, "1")

    def test_one_call_per_question_and_state_is_compacted(self):
        agent = FakeAgent()
        with fake_laya(agent):
            ask("z" * 5000, {"a": choice("q", {"x": "1", "y": "2"}), "b": choice("q", {"x": "1", "y": "2"})},
                max_state_chars=100)
        self.assertEqual([qid for _, qid in agent.calls], ["a", "b"])
        self.assertEqual(len(agent.calls[0][0]), 100)

    def test_compact_drops_trailing_keys_of_a_mapping(self):
        out = client._compact({"a": "x" * 10, "b": "y" * 200, "c": 1}, 40)
        self.assertIn("a", out)
        self.assertNotIn("c", out)

    def test_missing_model_is_a_clear_error(self):
        client._AGENTS.clear()
        with mock.patch.dict(sys.modules, {"laya": None}):
            with self.assertRaises(client.LayaUnavailable) as ctx:
                load("typed-decisions")
        self.assertIn("pip install laya", str(ctx.exception))


class GateTests(unittest.TestCase):
    def answer(self, confidence):
        agent = FakeAgent({"k": {"choice": "a", "confidence": 0.1,
                                 "probabilities": {"a": confidence, "b": 1 - confidence}}})
        with fake_laya(agent):
            return ask("t", {"k": choice("q", {"a": "x", "b": "y"})})["k"]

    def test_above_threshold_acts_below_abstains(self):
        gate = decide(self.answer(0.91), 0.8)
        self.assertEqual((gate.label, gate.abstained), ("a", False))
        gate = decide(self.answer(0.62), 0.8)
        self.assertEqual((gate.label, gate.abstained), (None, True))
        self.assertIn("below_threshold", gate.reason)

    def test_no_threshold_means_abstain(self):
        gate = decide(self.answer(0.99), None)
        self.assertTrue(gate.abstained)
        self.assertEqual(gate.reason, "uncalibrated")


class CalibrationTests(unittest.TestCase):
    def test_picks_the_lowest_band_that_holds(self):
        history = [(0.95, True)] * 40 + [(0.85, True)] * 10 + [(0.55, False)] * 10
        self.assertEqual(calibrate(history), 0.8)

    def test_too_few_labels_returns_none(self):
        self.assertIsNone(calibrate([(0.95, True)] * 10))

    def test_no_band_qualifies_returns_none(self):
        self.assertIsNone(calibrate([(0.95, False)] * 40))

    def test_a_perfect_small_history_is_floored_not_zeroed(self):
        history = [(0.30, True)] * 20 + [(0.95, True)] * 20
        self.assertEqual(calibrate(history), policy.MIN_THRESHOLD)
        self.assertEqual(calibrate(history, min_threshold=0.0), 0.0)


class AgentTests(unittest.TestCase):
    def test_batch_reports_progress(self):
        agent = FakeAgent()
        seen = []
        with fake_laya(agent):
            out = Agent({"k": choice("q", {"a": "x", "b": "y"})}).batch(["one", "two"], on_item=lambda i, a: seen.append(i))
        self.assertEqual(len(out), 2)
        self.assertEqual(seen, [0, 1])


if __name__ == "__main__":
    unittest.main()


KHMER = "សួស្តី ពិភពលោក"


class ScriptTests(unittest.TestCase):
    """An English checkpoint answers other scripts confidently and wrongly, so the script is
    checked before the call and the gate refuses the answer whatever the confidence says."""

    def confident(self, state, **kwargs):
        agent = FakeAgent({"k": {"choice": "a", "confidence": 0.1,
                                 "probabilities": {"a": 0.99, "b": 0.01}}})
        with fake_laya(agent):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ask(state, {"k": choice("q", {"a": "x", "b": "y"})}, **kwargs)["k"]

    def test_latin_text_is_not_flagged(self):
        self.assertEqual(non_latin("The export button returns a 500."), (0.0, None))

    def test_another_script_is_named_with_its_share(self):
        share, script = non_latin(KHMER)
        self.assertEqual(script, "KHMER")
        self.assertGreater(share, 0.9)

    def test_a_stray_foreign_word_stays_under_the_limit(self):
        share, script = non_latin("The reviewer signed off on the Q1 access review in Munchen" + "\u00e9")
        self.assertLessEqual(share, client.NON_LATIN_LIMIT)

    def test_off_script_state_is_refused_however_confident(self):
        answer = self.confident(KHMER)
        self.assertTrue(answer.out_of_script)
        self.assertEqual(answer.confidence, 0.99)
        gate = decide(answer, 0.5)
        self.assertTrue(gate.abstained)
        self.assertIsNone(gate.label)
        self.assertEqual(gate.reason, "out_of_script:khmer")

    def test_the_multilingual_checkpoint_is_not_guarded(self):
        answer = self.confident(KHMER, checkpoint="multilingual")
        self.assertFalse(answer.out_of_script)
        self.assertFalse(decide(answer, 0.5).abstained)

    def test_warns_before_the_call(self):
        agent = FakeAgent()
        with fake_laya(agent):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ask(KHMER, {"k": choice("q", {"a": "x", "b": "y"})})
        self.assertTrue(any("KHMER" in str(w.message) for w in caught))

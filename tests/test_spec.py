import unittest

from speclab.eval import build_pair, run_evaluation
from speclab.model import CharNGram
from speclab.spec import EvalCounter, speculative_run, target_only


class TestEquivalence(unittest.TestCase):
    """Speculative output must equal target-only greedy, for any config."""

    def configs(self):
        return [
            dict(draft_order=3, target_order=4, draft_smoothing=0.1, target_smoothing=0.01),
            dict(draft_order=2, target_order=3, draft_smoothing=0.1, target_smoothing=0.05),
            dict(draft_order=3, target_order=5, draft_smoothing=0.05, target_smoothing=0.01),
            dict(draft_order=1, target_order=4, draft_smoothing=0.2, target_smoothing=0.01),
        ]

    def test_identical_across_configs_prompts_lengths(self):
        for cfg in self.configs():
            draft, target = build_pair(200, 42, **cfg)
            for prompt in ["the model", "the cache", "the parser"]:
                for n in (25, 60, 90):
                    base = target_only(target, prompt, n)
                    spec = speculative_run(draft, target, prompt, n, gamma=4)
                    self.assertEqual(base["text"], spec["text"],
                                     f"cfg={cfg} prompt={prompt} n={n}")

    def test_identical_across_gammas(self):
        draft, target = build_pair(200)
        for gamma in (1, 2, 4, 8):
            base = target_only(target, "the model", 60)
            spec = speculative_run(draft, target, "the model", 60, gamma=gamma)
            self.assertEqual(base["text"], spec["text"], f"gamma={gamma}")


class TestCounting(unittest.TestCase):
    def test_baseline_counts_one_eval_per_token(self):
        _, target = build_pair(200)
        r = target_only(target, "the model", 30)
        self.assertEqual(r["target_evals"], r["emitted"])

    def test_verify_block_is_one_evaluation(self):
        _, target = build_pair(200)
        c = EvalCounter(target)
        c.verify_block("the model", ["s", "c", "o", "r"])
        self.assertEqual(c.evaluations, 1)

    def test_speculative_saves_target_evals(self):
        draft, target = build_pair(200)
        base = target_only(target, "the model", 60)
        spec = speculative_run(draft, target, "the model", 60, gamma=4)
        self.assertLess(spec["target_evals"], base["target_evals"])

    def test_acceptance_in_range(self):
        draft, target = build_pair(200)
        spec = speculative_run(draft, target, "the model", 60, gamma=4)
        self.assertGreaterEqual(spec["acceptance"], 0.0)
        self.assertLessEqual(spec["acceptance"], 1.0)
        self.assertEqual(spec["accepted"], round(spec["acceptance"] * spec["drafted"]))


class TestFixtureEvaluation(unittest.TestCase):
    def test_pinned_fixture_numbers(self):
        draft, target = build_pair()
        r = run_evaluation(draft, target)
        self.assertTrue(r["identical"])
        self.assertEqual(r["spec_target_evals"], 122)
        self.assertEqual(r["baseline_target_evals"], 480)
        self.assertAlmostEqual(r["mean_acceptance"], 0.800, places=3)
        self.assertAlmostEqual(r["target_eval_savings"], 0.7458, places=4)

    def test_determinism(self):
        d1, t1 = build_pair()
        d2, t2 = build_pair()
        self.assertEqual(run_evaluation(d1, t1), run_evaluation(d2, t2))


if __name__ == "__main__":
    unittest.main()

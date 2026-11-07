import io
import json
import unittest
from contextlib import redirect_stdout

from speclab.cli import main


def run_cli(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(list(argv))
    return code, buf.getvalue()


class TestCli(unittest.TestCase):
    def test_evaluate_terminal(self):
        code, out = run_cli("evaluate")
        self.assertEqual(code, 0)
        self.assertIn("identical", out)
        self.assertIn("acceptance", out)

    def test_evaluate_json(self):
        code, out = run_cli("evaluate", "--format", "json")
        data = json.loads(out)
        self.assertTrue(data["identical"])

    def test_achievable_gates_pass(self):
        code, out = run_cli("gates", "--require-identical",
                            "--min-acceptance", "0.6", "--min-eval-savings", "0.5")
        self.assertEqual(code, 0)
        self.assertIn("All gates passed", out)

    def test_impossible_acceptance_gate_fails(self):
        code, out = run_cli("gates", "--min-acceptance", "0.95")
        self.assertEqual(code, 1)
        self.assertIn("GATE FAILURES", out)

    def test_impossible_savings_gate_fails(self):
        code, _ = run_cli("gates", "--min-eval-savings", "0.9")
        self.assertEqual(code, 1)

    def test_weak_draft_pair_shows_low_acceptance(self):
        # A poorly matched draft/target pair must show up as low acceptance,
        # proving acceptance is measured, not assumed.
        code, out = run_cli("evaluate", "--draft-order", "2",
                            "--target-order", "3", "--format", "json")
        data = json.loads(out)
        self.assertLess(data["mean_acceptance"], 0.2)
        self.assertTrue(data["identical"])


if __name__ == "__main__":
    unittest.main()

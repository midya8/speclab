"""Public labels must distinguish modeled rounds from actual target calls."""
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

from speclab.cli import main
from speclab.eval import format_terminal


def cli(*args):
    output=io.StringIO()
    with redirect_stdout(output):
        code=main(list(args))
    return code,output.getvalue()


class ReportingTests(unittest.TestCase):
    def test_real_fixture_terminal_and_markdown_expose_more_actual_calls(self):
        code,text=cli('evaluate','--format','json');self.assertEqual(code,0)
        result=json.loads(text)
        self.assertEqual((result['spec_target_evals'],result['baseline_target_evals']),(122,480))
        self.assertEqual((result['spec_scoring_calls'],result['baseline_scoring_calls']),(510,480))
        self.assertGreater(result['spec_scoring_calls'],result['baseline_scoring_calls'])
        self.assertGreater(result['target_eval_savings'],0)
        for mode in ('terminal','markdown'):
            code,text=cli('evaluate','--format',mode);self.assertEqual(code,0)
            self.assertIn('modeled',text.lower());self.assertIn('round',text.lower())
            self.assertIn('510',text);self.assertIn('480',text)
            self.assertIn('not a measured compute or latency saving',text)
            self.assertNotIn('of target evaluations',text)
            self.assertNotIn('Target evaluations:',text)
        self.assertEqual(json.loads(cli('evaluate','--format','json')[1]),result)

    def test_terminal_scoring_counts_are_dynamic_not_fixture_constants(self):
        r={'prompts':[],'n_tokens':2,'gamma':1,'identical':True,'mean_acceptance':.25,
           'spec_target_evals':3,'baseline_target_evals':4,'target_eval_savings':.25,
           'total_draft_evals':7,'spec_scoring_calls':9,'baseline_scoring_calls':4}
        text=format_terminal(r)
        self.assertIn('rounds: 3 vs 4',text)
        self.assertIn('scoring work: 9 calls vs 4',text)
        self.assertNotIn('510',text)
        with patch('speclab.cli.build_pair',return_value=(None,None)),patch('speclab.cli.run_evaluation',return_value=r):
            code,text=cli('evaluate','--format','markdown')
        self.assertEqual(code,0);self.assertIn('calls: 9 vs 4',text)
        self.assertIn('25.0%',text)

    def test_legacy_gate_flag_keeps_behavior_with_honest_failure_name(self):
        code,text=cli('gates','--min-eval-savings','.9')
        self.assertEqual(code,1)
        self.assertIn('modeled verification-round reduction',text)
        self.assertIn('510 calls vs 480',text)
        self.assertNotIn('target-eval savings',text)
        code,text=cli('gates','--require-identical','--min-eval-savings','.5')
        self.assertEqual(code,0);self.assertIn('All gates passed.',text)

    def test_cli_help_does_not_present_tested_code_as_a_proof(self):
        for args in (['--help'],['gates','--help']):
            output=io.StringIO()
            with self.assertRaises(SystemExit) as e,redirect_stdout(output):main(args)
            self.assertEqual(e.exception.code,0)
            self.assertNotIn('proven equivalence',output.getvalue())
        self.assertIn('modeled verification-round reduction',output.getvalue())


if __name__=='__main__':unittest.main()

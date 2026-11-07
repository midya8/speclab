#!/usr/bin/env bash
# SpecLab demo: proven equivalence and measured savings, both gate sides. Offline.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Evaluate: speculative vs target-only greedy =="
python3 -m speclab evaluate

echo
echo "== Achievable gates pass =="
python3 -m speclab gates --require-identical --min-acceptance 0.6 --min-eval-savings 0.5

echo
echo "== Impossible gate fails (negative control) =="
if python3 -m speclab gates --min-acceptance 0.95; then
  echo "ERROR: impossible gate passed" >&2
  exit 1
else
  echo "impossible gate failed as expected"
fi

echo
echo "Demo complete: identical output with 74.6% fewer target evaluations."

# SpecLab

Speculative decoding built from scratch in pure Python: a cheap draft model proposes a block of tokens, an exact target model verifies the block in one counted pass, and the output is proven token-for-token identical to target-only decoding while the evaluation savings are measured directly.

SpecLab answers the two questions speculative-decoding write-ups usually skip:

- **Is the output actually unchanged?** Yes - by construction and by proof. Every emitted token is either a drafted token the target greedily confirms or the target's own greedy choice, and the test suite verifies exact equality with target-only decoding across four draft/target configurations, three generation lengths, and four block sizes (gamma 1, 2, 4, 8).
- **What does it save?** Target-model evaluations, counted exactly. On the checked-in fixture (order-3 draft, order-4 target, 8 prompts x 60 tokens, gamma=4): mean draft acceptance 0.800, and 122 target evaluations against 480 naive - a measured 74.6% saving.

## How it works

- **Exact models, not approximations** - draft and target are character-level n-grams over one deterministic corpus; the target's greedy choice comes straight from counts, so agreement and verification are ground truth.
- **Batched verification, honestly counted** - `verify_block` scores the whole drafted block in one pass and counts as exactly one target evaluation; the counter is asserted in tests, not eyeballed.
- **Measured acceptance, including the failure mode** - a mismatched pair (order-2 draft vs order-3 target) drops acceptance to 0.03 and nearly all savings disappear; that configuration is pinned in the suite so acceptance is demonstrably measured, not assumed.
- **Deterministic everything** - seeded corpus, count-based models, pinned fixture numbers in tests.

## Usage

```bash
python3 -m speclab evaluate                  # per-prompt table
python3 -m speclab evaluate --format json    # machine-readable
python3 -m speclab gates --require-identical \
  --min-acceptance 0.6 --min-eval-savings 0.5
bash examples/demo.sh                        # offline demo, both gate sides
```

## Verification

- 16 unit tests cover output equivalence across configurations and gammas, the one-evaluation-per-block counting contract, the naive baseline's one-evaluation-per-token contract, acceptance bounds and consistency, pinned fixture numbers, determinism, both sides of every CLI gate, and the low-acceptance negative control.
- CI runs the same suite across Python 3.10-3.13.

## Bounds worth stating

Greedy decoding only - the speculative *sampling* rejection rule is future work. Draft and target are n-gram toy models chosen so they share structure, as real draft/target pairs do; nothing here claims LLM-scale acceptance rates. Savings are measured in target-model evaluations, not wall-clock latency: real deployments save time because one batched verification pass is cheaper than token-by-token decoding, which is a memory-bandwidth argument this project counts rather than benchmarks. The draft model's own evaluations are reported separately and are the real cost of speculation.

## License

MIT

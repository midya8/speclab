# SpecLab

Speculative decoding built from scratch in pure Python: a cheap draft model proposes a block of tokens, a target model verifies the block, and a greedy-construction invariant keeps the output identical to target-only decoding - with the invariant stated as an algorithm property and the implementation verified token-for-token on the tested configurations, and verification rounds counted exactly.

SpecLab answers the two questions speculative-decoding write-ups usually skip:

- **Is the output actually unchanged?** The greedy construction has an exact algorithm invariant: every emitted token is either a drafted token the target greedily confirms or the target's own greedy choice, so ideal greedy speculative decoding equals target-only greedy decoding. Separately from the invariant, the checked-in implementation is verified against it on four draft/target configurations, three generation lengths, and four block sizes (gamma 1, 2, 4, 8) - finite tests of this code, not a proof of it.
- **What does it save?** Verification rounds, counted exactly: one modeled round per drafted block versus one round per token naive. On the checked-in fixture (order-3 draft, order-4 target, 8 prompts x 60 tokens, gamma=4): mean draft acceptance 0.800, and 122 rounds against 480 naive - a 74.6% modeled reduction. This is a verification-round count, not a measured reduction in model computation: this implementation scores drafted positions sequentially inside each round, and its actual scoring work on the fixture is 510 calls versus 480 naive.

## How it works

- **Exact models, not approximations** - draft and target are character-level n-grams over one deterministic corpus; the target's greedy choice comes straight from counts, so agreement and verification are ground truth.
- **Verification rounds, honestly labeled** - `verify_block` counts each drafted block as one modeled verification round (the batched pass a real deployment runs) while also counting the actual sequential scoring calls inside it; both numbers are reported, and both counting contracts are asserted in tests.
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

- 14 unit tests cover implementation equivalence with the greedy invariant across configurations and gammas, both counting contracts (one modeled round per block, actual scoring calls tracked separately, naive one round per token), acceptance bounds and consistency, pinned fixture numbers including actual scoring work, determinism, both sides of every CLI gate, and the low-acceptance negative control.
- CI runs the same suite across Python 3.10-3.13.

## Bounds worth stating

Greedy decoding only - the speculative *sampling* rejection rule is future work. The equivalence claim is an algorithm invariant of the ideal greedy construction plus finite implementation tests on the listed configurations, not a proof of this code. Draft and target are n-gram toy models chosen so they share structure, as real draft/target pairs do; nothing here claims LLM-scale acceptance rates. The headline saving is a modeled verification-round count: real deployments save wall-clock because a batched verification pass costs little more than decoding one token (a memory-bandwidth argument this project does not benchmark), while this implementation's actual sequential scoring work (510 calls vs 480 naive on the fixture) is reported alongside the round count. The draft model's own evaluations are reported separately and are the real cost of speculation.

## License

MIT

# SpecLab

Speculative decoding built from scratch in pure Python: a cheap draft model proposes a block of tokens, a target model verifies the block, and a greedy-construction invariant keeps the output identical to target-only decoding - with the invariant stated as an algorithm property and the implementation verified token-for-token on the tested configurations, and verification rounds counted exactly.

SpecLab answers the two questions speculative-decoding write-ups usually skip:

- **Is the output actually unchanged?** The greedy construction has an exact algorithm invariant: every emitted token is either a drafted token the target greedily confirms or the target's own greedy choice, so ideal greedy speculative decoding equals target-only greedy decoding. Separately from the invariant, the checked-in implementation is verified against it on four draft/target configurations, three generation lengths, and four block sizes (gamma 1, 2, 4, 8) - finite tests of this code, not a proof of it.
- **What does it save?** Verification rounds, counted exactly: one modeled round per drafted block versus one round per token naive. On the checked-in fixture (order-3 draft, order-4 target, 8 prompts x 60 tokens, gamma=4): mean draft acceptance 0.800, and 122 rounds against 480 naive - a 74.6% modeled reduction. This is a verification-round count, not a measured reduction in model computation: this implementation scores drafted positions sequentially inside each round, and its actual scoring work on the fixture is 510 calls versus 480 naive.

## v0.3: choose lookahead, including no speculation

![Exact prefix policy and full-strength baselines](assets/lookahead-policy.svg)

An exact **oracle** planner now chooses draft length 0..3 from each emitted
prefix. It retains every Bellman action value, transition and nonnegative slack,
then independently evaluates the full selected-policy sequence/count law. All
fixed lengths, including target-only action 0, are eligible baselines.

On the four-token switching fixture, the optimal assigned cost is `23/10`
versus best fixed `47/20`, a difference of `1/20` hypothetical units. Charge
actual target **positions** instead of rounds and the optimizer chooses no
speculation everywhere. The same target output law is preserved. Unreachable
prefix decisions are shown but not charged as executed rounds.

**47 tests** include exhaustive enumeration of all 12 tiny prefix policies for
25 model pairs under three cost objectives, 72 full independent joint-law
comparisons, 36 size/horizon/gamma-limit cases, and a 115-cell data-bound figure.
CI verifies all seven modes from an installed 12-module wheel outside checkout.
[Model, Bellman proof, full baseline table and prior adaptive-lookahead work](LOOKAHEAD.md).

```bash
python -m speclab.policy demo
python -m speclab.policy optimize examples/policy-input.json
python examples/render_policy.py --check
```

The planner knows every target/draft conditional. Planning, table evaluation
and policy lookup overhead are not modeled. It chooses a length before drafting,
not a confidence threshold while drafting; no measured latency, learned
controller, real-model or novel adaptive-decoding algorithm claim.

## v0.2: exact stochastic sampling, not only greedy equivalence

![Exact stochastic laws and decoder counts](assets/stochastic-audit.svg)

A separate finite autoregressive-table engine now implements sampled drafts,
`min(1,p/q)` acceptance and normalized positive-residual correction. It retains
complete proposal/first-rejection/bonus witnesses, exact full-sequence laws,
joint sequence/count laws and a procedural integer-ticket sampler. The
[stochastic model, proof, sources and bounds](STOCHASTIC.md) explain why same
output law is not same-seed output, and why target fallback after rejection
can be biased: the prefix-dependent control has TV `141/1024`, while the
corrected law has TV `0` across every displayed sequence.

The four-token mismatch averages `75/32` modeled verification rounds but
`179/32` target positions and `13/4` draft calls. An explicit hypothetical
parallel-cost scenario can improve while a position-score-work scenario gets
worse. These are assigned costs, not measured GPU throughput or FLOPs.

The v0.2 base suite contains 34 tests. Independent references cover 225 one-step
distribution pairs, 2,187 prefix-table pairs, 36 horizon/vocabulary/gamma
configurations, and exact RNG-ticket enumeration of the procedural sampler
for all 25 binary quarter-grid pairs at horizon 2. The SVG binds all 96 data
cells and complete metadata. The v0.2 installed ten-module wheel parity covers the
original two command modes plus three new stochastic modes outside checkout.

```bash
python -m speclab.stochastic demo
python -m speclab.stochastic audit examples/stochastic-input.json
python -m speclab.stochastic sample examples/stochastic-input.json --seed 19
python examples/render_stochastic.py --check
```

Finite supplied tables only: 1..3 tokens, horizon 1..4, gamma 1..3. No EOS,
learned model, numerical kernel, hardware speed or real-LLM quality claim.
The original greedy implementation and demo below remain unchanged.

## Original greedy n-gram demo

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

This original n-gram evaluation is greedy decoding only. The separate v0.2 finite-table stochastic sampler is documented above and does not change these original numbers. The equivalence claim is an algorithm invariant of the ideal greedy construction plus finite implementation tests on the listed configurations, not a proof of this code. Draft and target are n-gram toy models chosen so they share structure, as real draft/target pairs do; nothing here claims LLM-scale acceptance rates. The headline saving is a modeled verification-round count: real deployments save wall-clock because a batched verification pass costs little more than decoding one token (a memory-bandwidth argument this project does not benchmark), while this implementation's actual sequential scoring work (510 calls vs 480 naive on the fixture) is reported alongside the round count. The draft model's own evaluations are reported separately and are the real cost of speculation.

## License

MIT

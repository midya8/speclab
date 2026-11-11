# Stopping is not censoring

SpecLab v0.4 adds an exact finite audit of **within-proposal** stopping. The v0.3 oracle planner chooses an entire draft length before the round starts. This experiment instead permits decisions while the draft is being built. The original greedy decoder, stochastic sampler and oracle planner remain unchanged.

The browser runs a separate BigInt rational implementation. It does not load precomputed reports. Python uses `Fraction`. The independent verifier takes a different enumeration route: generate every full maximum-length draft path, marginalize unused suffixes, and expand full decoding states forward.

## A two-token counterexample you can check by hand

Let the target distribution be `p=(1/2,1/2)` and the draft be `q=(3/4,1/4)` at every prefix. There are two output positions, maximum draft length one, and threshold `1/2`.

- **Keep and verify:** sample A or B from q. If B is sampled, stop drafting but keep B for verification. Accepted mass on A is `(3/4)*(2/3)=1/2`; accepted mass on B is `1/4`. A rejection contributes another `1/4` of B through the residual. First output is exactly `(1/2,1/2)`.
- **Censor and forget:** erase sampled B, verify only the retained path against the original q, and draw an ordinary target bonus if the path is empty. A's accepted mass is still `1/2`, and the erased B path adds `(1/4)*(1/2)=1/8` of A. First output is now `(5/8,3/8)`.

The complete two-position target law is `AA=AB=BA=BB=1/4`. The wrong control produces `AA=AB=5/16`, `BA=BB=3/16`. Its **full-sequence total variation is exactly 1/8**. This is selection bias, not floating-point error or sampling noise.

This wrong control is deliberately defined here. It is **not attributed to Transformers, DISCO, AdaEDL or SpecDec++**. A stopping rule can be safe without matching either of this lab's safe heuristics. Censoring could also be corrected by a suitably re-derived selected proposal law; that different algorithm is not implemented here.

## Five precisely specified rules

| Rule | Decision | What gets verified |
|---|---|---|
| `fixed` | Always draft to the cap | Every generated proposal |
| `before_max` | Before sampling, stop if `max(q)<threshold` | Previously generated proposals |
| `after_keep` | After drawing x, stop if `q(x)<threshold` | Previous proposals **and x** |
| `after_censor` | After drawing x, stop if `q(x)<threshold`, erase x | Only previous proposals, still using unconditioned q; intentionally wrong |
| `no_speculation` | Never draft | One direct target sample per round |

Equality does not stop. `after_keep` uses the **sampled token's** probability, not top-1 confidence. `before_max` uses the maximum of the current proposal distribution. They are not interchangeable implementations of a library setting.

Every mode reserves one remaining output position for a rejection correction or all-accepted bonus. Thus a round with r remaining positions drafts at most `min(max_gamma,r-1)` proposals. This keeps the existing SpecLab finite-horizon convention. There is no EOS, variable completion length or tokenizer mismatch.

### Why keeping the stop token works in this model

For a fixed prefix, write `a(x)=min(1,p(x)/q(x))` when `q(x)>0`. The accepted proposal mass is `q(x)a(x)=min(p(x),q(x))`. The rejected mass is `1-sum_x min(p(x),q(x))`. Normalized positive residual `(p-q)+` contributes `p(x)-min(p(x),q(x))`. These sum to p(x). Zero-q tokens are never proposed; they can appear through the residual.

A before-next decision depends only on the already available prefix and q distribution, not an unseen next sample. If it stops, the next output is sampled from p. If it continues, the corrected next proposal also has law p.

An after-keep decision may inspect the sampled token because that token still participates in exactly the same acceptance/correction rule. Whether its acceptance is followed by another proposal or a target bonus only changes how the *next* output is obtained. Given each history where another token must be emitted, either construction has next-token law p. Stop decisions never inspect future draft samples or the target's acceptance random variables. Induction over the finite remaining horizon gives the full target sequence law.

This argument does not claim arbitrary selection-dependent pruning is safe. Conditioning on the identity of a token and then deleting it breaks the original q proposal calculation. The censoring counterexample shows the failure without relying on a theorem assertion in the implementation.

## Evidence schema

`stopping.audit` returns:

- The complete target and draft tables, including conditionals unreachable under the target.
- Every stopped proposal leaf: sampled path, retained path, exact mass, draft-distribution call count, censored count, stop reason and chronological decisions.
- Every nonzero verification branch: proposal-leaf ID, accepted prefix length, first rejection index or bonus marker, residual/bonus categorical distribution, selected replacement, emitted tokens, exact branch mass and operation-count vector.
- Full final sequence law, full joint operation-count law, and **joint sequence/count law**, not only expected work.
- Independent direct-target product law and full-sequence total variation for each rule.
- Unweakened fixed-length baselines at caps 1, 2 and 3, plus target-only. A baseline is still shown when a selected threshold makes it look better than stopping.

The producer uses stopped-tree recursion and cached suffix-law composition. `stopverify.py` does not import it: full maximum-length paths are enumerated and grouped by observed stopped leaf, then a forward state frontier composes the final law. The verifier compares **every field, list order, scalar value and JSON type**, including keys that might otherwise look merely explanatory. Native model parsing and exact rational primitives are shared. Independence is in the probability enumeration and composition, not in the input grammar.

The browser has a third BigInt implementation and is compared against every field of the Python evidence, not just a TV summary. Its editable complete conditional table recomputes all laws. Input edits invalidate displayed and downloadable results immediately. Downloaded reports are replayable by the Python verifier.

## Count semantics, not a speed benchmark

| Counter | Charge |
|---|---|
| `verification_rounds` | One target call per round |
| `target_positions` | Retained proposals plus one bonus/correction position per round, including scores wasted after an early rejection |
| `draft_calls` | One draft-distribution evaluation for each sample, plus a before-next check that decides not to sample |
| `sampled_proposals` | All drawn draft tokens, including a censored token |
| `retained_proposals` | Proposals submitted to target verification |
| `censored_proposals` | Drawn tokens deliberately erased by the wrong control |
| `accepted_proposals` | Verified retained proposals that survive rejection sampling |
| `rejection_rounds` | Rounds that emit a residual correction |
| `bonus_tokens` | All-accepted target bonus, including direct-target rounds |

Every count vector, not merely its expectation, satisfies:

```text
target_positions = verification_rounds + retained_proposals
sampled_proposals = retained_proposals + censored_proposals
horizon = accepted_proposals + rejection_rounds + bonus_tokens
```

The before-next rule pays for a distribution even when it refuses to sample. The direct-target rule pays no draft calls. No prefix inspection, Python enumeration, independent verification or BigInt overhead is presented as a model operation. There is no claim of measured latency, GPU utilization, throughput, deployment efficiency or quality improvement. A lower draft-call count can coexist with more target rounds. Operation counts are intentionally not collapsed to a convenient speedup number.

## Controls and reproduction

The 12 hand-specified controls cover sample censorship bias; changed work with unchanged law; prefix-dependent confidence; identical models that still become biased under censorship; disjoint support; equality at the threshold; threshold zero; threshold one with paid unused lookups; a one-token horizon; three-token residuals; mass `2^-128`; and a zero-probability proposal.

```sh
python -m unittest discover -s tests -v
node --test tests/test_stopping.cjs tests/test_stopping_browser.cjs
python -m speclab.stopping_cli audit examples/stopping-input.json > /tmp/report.json
python -m speclab.stopping_cli verify examples/stopping-input.json /tmp/report.json
python -m speclab.stopping_cli demo > /tmp/stopping-fixtures.json
python examples/check_stopping.py
```

Checked-in complete inputs and per-report SHA-256 values freeze all 12 reports without committing megabytes of redundant generated JSON. The check script regenerates the reports, compares the frozen hashes and independently verifies every field.

The real-Chrome test visits all 12 full reports, all five rules, every reachable round prefix, every stopped leaf and every verification branch; checks exact rendered sequence cells; performs trusted text insertion with invalidation and restoration; downloads actual Blob files and independently replays them; and checks 320, 390 and 1280 pixel layouts. Optional `STOPPING_SCREEN_DIR` captures full-page evidence. `STOPPING_URL` runs the same checks on a deployment.

Input is a JSON object with exactly `target`, `draft`, `max_gamma`, `threshold`. Models use the existing finite-table grammar: 1-3 distinct ASCII alphanumeric tokens, horizon 1-4, every shorter prefix present exactly once, normalized nonnegative rational vectors. Maximum draft length is 1-3. Threshold is in [0,1]. Exact values use integer or rational/decimal strings with at most 512-bit components. Browser numeric JSON integers must be safe JavaScript integers; use strings for larger values. The CLI refuses duplicate keys (including escaped duplicates), noninteger JSON numbers, nonfinite numbers, files over 8 MB and nesting over 48. Browser editing uses typed rational fields rather than a permissive JSON editor.

## Relationship to prior work

Primary sources establish rejection correction and the existence of dynamic stopping methods. The finite proof, counterexample, exact enumerators, evidence schema, independent verifier and browser here are educational engineering, not a new research claim or a reproduction of their reported performance.

1. Leviathan, Kalman and Matias, *Fast Inference from Transformers via Speculative Decoding*. Section 2 and Appendix A.1 give the acceptance/residual identity and all-accepted bonus. https://arxiv.org/html/2211.17192v2
2. Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*. The modified rejection-sampling proof recovers the target distribution. That paper uses p for draft and q for target, the opposite of this code. https://arxiv.org/html/2302.01318
3. Hugging Face, *Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models*. The official article says that after a confidence-triggered stop, generated draft tokens are sent for verification. It is historical implementation context, not a claim about today's library defaults or a byte-for-byte reproduction here. https://huggingface.co/blog/dynamic_speculation_lookahead
4. *AdaEDL: Early Draft Stopping for Speculative Decoding of Large Language Models via an Entropy-based Lower Bound on Token Acceptance Probability*. Section 2 separates top-1 confidence from learned predictors. Section 3 calls its entropy expression an **approximate** bound. This lab does not implement that approximation. https://arxiv.org/html/2410.18351
5. *SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths*. Sections 4.2-4.3 model partial draft states and stop/continue actions with a learned acceptance head; the stopped candidate list goes through target verification. This lab has no learned head or optimal stopping claim. https://arxiv.org/html/2405.19715
6. Mamou et al., *Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models*. Section 3 uses a classifier after each generated draft token, with top-k probabilities, entropy and position. Its greedy benchmark results are not results of this lab. https://arxiv.org/html/2405.04304v5

No real language model, private prompt, external API, pretrained artifact, paid resource or private dataset is required.

# Three-kernel test: K_fixed and K_live (K_avg awaiting approval)

Plan and outcome map: `llm/KERNEL_TEST_PLAN.md` (commit fe65958, before any kernel was computed). Code:
`llm/kernel_test.py`; post-hoc checks `llm/kernel_hidden_state.py` and `llm/kernel_estimator_check.py`. Outputs:
`results/llm/kernel_test/`. No API calls.

Simulation: the 5 live graphs per point, the live starting states, R = 400 runs per kernel. The simulator is
`surrogate.run` with only its update rule swapped for a table lookup; its measurement code is unchanged.
- Known-answer self-test: with the A3 kernel, this simulator reproduces `surrogate.run` (all |z| < 2.2).
- The self-test caught one bug in the new code before any result: the no-contacts arms looked up configurations
  they cannot see. Fixed by making the nc kernel depend on s0 only, as the plan says.

## Pre-registered verdicts

| kernel | P1 | P2 | P3 | P4 | P1–P4 passing | reproduces? |
|---|---|---|---|---|---|---|
| K_fixed (Stage 1 battery kernel) | miss (persistence z −1.9) | pass | miss (\|m\| z −2.4) | pass | 2 | no |
| K_fixed-A3 (committed surrogate) | miss | pass | miss | miss | 1 | no |
| **K_live** (n0 = 20) | pass | pass | miss (persistence z +1.9) | miss (persistence z +1.8) | **2** | **no** |
| K_live, n0 = 5 / n0 = 100 | | | | | 2 / 2 | no / no |

P5: K_live passes; K_fixed misses \|m\| (z +2.1); A3 misses by 14–20 SD. The nc arms: K_live passes m and misses
persistence (z +1.7, +2.3). K_fixed (= A3 fields-only, since there is no k = 0 battery) misses by 5–15 SD.

**Outcome-map branch: "K_live fails → the live system is not Markov in configuration; investigate hidden state
before claiming anything."** No headline claim is made.

## Investigation of the K_live failure (post-hoc, read-only)

- **The miss is one-directional and modest.** Live persistence is above the K_live prediction at 5 of 7 points:
  - z: P3 +1.9, P4 +1.8, P5 +1.6, P2-nc +1.7, P4-nc +2.3; P1 −0.8, P2 −0.2;
  - live flip rate is below the prediction at all 7 (z −0.1 to −1.8);
  - in absolute terms the persistence excess is 0.01–0.04.
- **Not the estimator.** Known-answer test: 20 synthetic "live" datasets per point, generated from a known kernel
  and pushed through the identical pipeline.
  - Mean z is within ±0.6 (persistence −0.6 to +0.5, mostly negative); coverage of the 90% intervals is 75–100%.
  - A fixed point-estimate kernel and the unsmoothed raw kernel give the same live z-pattern.
- **No hidden state detected at the per-step level** (chain-cluster bootstrap 95% CIs):

  | candidate | result |
  |---|---|
  | provider caching | identical prompts repeat with Stage 1-sized noise (logit SD 0.05–0.16; P1 0.11 vs Stage 1 0.03); bit-identical replies 20–57% |
  | wall-clock drift | ≈ 0 on repeated prompts (largest P4-nc +0.03 ± 0.01 logit; all others ≤ 0.005) |
  | time inhomogeneity: P(stay), measured minus warm-up sweeps | −0.004 to +0.002; every CI contains 0 |
  | history dependence: P(stay) after the agent's last update changed it vs not | −0.001 to +0.002; every CI contains 0 |
  | within-chain lag-1 residual correlation | −0.011 to +0.010 (only P5's CI excludes 0: +0.010 [0.001, 0.021]) |

- **Where that leaves it:**
  - Per step, the live transitions match a stationary kernel in (c, s0) to within about ±0.002 in P(stay).
  - Yet the collective trajectories sit consistently on the high-persistence, low-flip side of what that kernel
    produces. Stouffer z for persistence over the 7 points is ≈ +3.1.
  - The candidates named in the outcome map (provider caching, within-chain correlation) do not show up.
  - What remains is a dependence too small to see per step but visible collectively, e.g. on the graph
    neighbourhood beyond (c, s0), or on correlations between agents updated close together in time across
    chains. I have not tested these.
- Per the map, nothing is claimed until this is resolved.

## Amplification (per point; per-step weighted by live visit frequency)

| point | K_fixed vs K_live: mean \|ΔP\| | mean \|Δlogit\| (nats) | KL (nats) | collective, K_live sim vs K_fixed sim (SD units; primary measures) | A3 vs K_live: mean \|ΔP\| / \|Δlogit\| | collective, live vs A3 (SD units) |
|---|---|---|---|---|---|---|
| P1 | 0.010 | 0.22 | 0.003 | \|m\| −0.1, persistence −1.1 | 0.019 / 0.72 | −2.0, +3.2 |
| P2 | 0.008 | 0.16 | 0.001 | m +0.2, persistence −0.4 | 0.010 / 0.19 | −1.3, −0.6 |
| P3 | 0.002 | 0.07 | 0.011 | \|m\| +0.5, persistence **−3.4** | 0.016 / 0.67 | −5.2, +5.3 |
| P4 | 0.008 | 0.36 | 0.002 | m −0.9, persistence −1.2 | 0.013 / 0.66 | +3.6, +4.6 |
| P5 | 0.035 | 0.71 | 0.011 | \|m\| +1.1, persistence +0.5 | 0.123 / 1.63 | +13.9, +20.4 |
| P2-nc | — | — | — | (no battery; K_fixed = A3 fields-only) | 0.062 / 1.29 | −9.4, +14.6 |
| P4-nc | — | — | — | | 0.075 / 1.10 | +5.3, +11.7 |

Reading:
- **The battery kernel is close to the live kernel per step** (|ΔP| 0.002–0.010 at P1–P4). It averages four
  presentations (2 mappings × 2 option orders), each with its own seeded contact order, so the large per-call
  order effects largely average out.
- Its collective miss is mostly small, but per-step-invisible differences can still be amplified: at P3, |ΔP| =
  0.002 per step becomes 3.4 SD in persistence.
- **Most of the original Stage 3 miss is the parametric A3 surrogate, not the battery.** Its per-step error is
  0.2–1.6 nats against the live kernel and becomes 3–20 SD collectively. The single-agent data held the
  information; the fitted functional form did not carry it. At P5 in particular, the A3/MF fit says disorder and
  both empirical kernels say ordering.
- **The no-contacts arms are the k = 0 extrapolation failing** (5–15 SD). There is no battery cell to compare.

## K_avg: cost estimate (not run)

Every (c, s0) seen live (neutral: plus its relabelled counterpart). Up to 8 distinct contact orders × 2 option orders
× 2 letter mappings (political, workplace); logprob endpoints 1 call per prompt, P5 4 samples per prompt.
Priced at the actual Stage 3 cost per call (`results/llm/kernel_test/K_avg_cost_estimate.csv`):

| point | configurations | prompts | calls | est. cost |
|---|---|---|---|---|
| P1 qwen-122b | 68 | 764 | 764 | $0.03 |
| P2 qwen-9b | 70 | 1,536 | 1,536 | $0.03 |
| P3 llama-70b | 57 | 1,360 | 1,360 | $0.21 |
| P4 nemo-12b | 58 | 1,376 | 1,376 | $0.003 |
| P5 gpt6-luna | 10 | 64 | 256 | $0.004 |
| **total** | | | **5,292** | **≈ $0.28** |

Options that stay under $1:
- 2 replicates per logprob prompt, for a noise floor: +$0.27.
- 30 samples per P5 prompt instead of 4, as in Stage 1, since P5 has only 4–12 prompts per configuration: +$0.03.

---

# Update 2026-10-12: neighbourhood test and K_avg

## Neighbourhood and concurrency test (free; post-hoc; `llm/kernel_neighbourhood.py`, `neighbourhood.json`)

Residual of every live update relative to K_live(c, s0):

- **Wider neighbourhood: no dependence.** Slopes on the 2-hop state, the number of contacts that changed since the
  agent's previous update, and the time since that update: all |slope| ≤ 0.002 per SD, every CI contains 0. The one
  exception is P2-nc, 2-hop, at +0.0003, which is negligible.
- **Concurrency: one small effect, at one provider.** Calls served in the same 5 s window share a common offset only
  at Io Net (Nemo, P4/P4-nc): F = 1.16, permutation p < 0.001, intra-window correlation 0.003, lag-1 correlation of
  window means 0.10. Alibaba, Parasail and CoreWeave show no effect (p 0.22–0.74). This cannot explain the
  persistence excess, which also appears at P3, P5 and P2-nc.
- **Visited configurations: not unusual.** In the measured sweeps, the mean K_live flip probability of the
  configurations the live trajectories visited is within −1.2 to +1.9 SD of K_live-simulated trajectories.
- **Conclusion:** no mechanism is found in the logs. The modest persistence excess of the live trajectories over
  every Markov kernel stays unexplained. The remaining test is the paid one: chains rerun strictly one at a time.

## K_avg (approved; 11,992 calls, $0.58; amendment 1, f6b1922)

- **Run:** 0 errors, 1 invalid reply, every call on its pinned provider; cache uploaded to the private HF dataset and
  verified.
- **Corrections to amendment 1's text:** the call count is 11,992, not ≈ 7,500 (the doubled logprob calls were
  under-counted); the cost matches the estimate.
- **Analysis bugs fixed before any reading:** the P5 Beta-draw table crashed (an unnamed series); the
  "updates on fallback" metric used `isclose`, which matches P3's near-0/1 probabilities. Fallback is now tracked
  explicitly and is 0% at every point: every live configuration was covered.

### Per step: the order-marginalised battery recovers the live kernel

Weighted by live visit frequency:

| point | K_avg vs K_live: \|ΔP\| / \|Δlogit\| (nats) | K_fixed vs K_live: \|ΔP\| / \|Δlogit\| |
|---|---|---|
| P1 | 0.004 / 0.05 | 0.010 / 0.22 |
| P2 | 0.002 / 0.03 | 0.008 / 0.16 |
| P3 | 0.001 / 0.03 | 0.002 / 0.07 |
| P4 | 0.003 / 0.07 | 0.008 / 0.36 |
| P5 | 0.013 / 0.65 | 0.037 / 0.65 |

### Collectively

Pre-registered rule, live graphs and starts, R = 400:

| point | measure | live | K_avg 90% interval | inside | z |
|---|---|---|---|---|---|
| P1 | \|m\| | 0.208 | [0.214, 0.396] | no | −1.8 |
| P1 | persistence | 0.778 | [0.758, 0.816] | yes | −0.5 |
| P2 | m | 0.561 | [0.505, 0.614] | yes | −0.1 |
| P2 | persistence | 0.603 | [0.568, 0.641] | yes | −0.1 |
| P3 | \|m\| | 0.101 | [0.095, 0.127] | yes | −1.0 |
| P3 | persistence | 0.9908 | [0.974, 0.9906] | no (by 0.0002) | +1.6 |
| P4 | m | 0.936 | [0.923, 0.951] | yes | −0.2 |
| P4 | persistence | 0.897 | [0.854, 0.903] | yes | +1.2 |
| P5 | \|m\| | 0.596 | [0.300, 0.596] | yes | +1.7 |
| P5 | persistence | 0.660 | [0.396, 0.606] | no | +2.4 |

**K_avg: 2 of P1–P4 (P2, P4) → does not reproduce.** The same count as K_live and K_fixed.

- **K_avg and K_live make the same collective predictions.** At P1–P4, every primary measure's mean differs by at
  most 0.9 SD of the K_live spread (P5: −0.9 for |m| and −2.35 for persistence).
- So the order-marginalised one-shot battery predicts the population as well as the empirical live kernel does.
  Both miss the live realisation in the same places (P1 |m|, P3 persistence), by 1.6–1.8 SD.

## Where this stands on the outcome map

- **K_live fails**, so the first branch governs: no claim until the gap between the live trajectories and any
  Markov kernel in (c, s0) is explained.
- Free checks so far (caching, drift, history, lag correlation, neighbourhood, concurrency) find nothing.
- What the data show, recorded without a headline:
  1. **The parametric A3 surrogate causes most of the Stage 3 miss:** 0.2–1.6 nats per step, 3–20 SD collectively.
  2. **The fixed-order battery is close per step and is amplified only modestly** (≤ 3.4 SD, at P3).
  3. **The order-marginalised battery reproduces the live kernel per step (≤ 0.004 in P at P1–P4)** and makes the
     same collective predictions as the empirical live kernel.
  4. **What remains is a small persistence excess of the live runs over every Markov kernel, with no identified
     cause.**

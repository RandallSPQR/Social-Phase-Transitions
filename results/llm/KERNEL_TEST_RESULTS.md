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

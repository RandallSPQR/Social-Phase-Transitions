# Residual test: results so far (test 1 complete; test 2 awaiting approval)

Plan, predictions and interpretation map: `llm/RESIDUAL_TEST_PLAN.md` (commit 9c759f7, before either test ran). Code:
`llm/residual_heterogeneity.py` (a, b), `llm/residual_heterogeneity_c.py` (c and robustness). Outputs:
`results/llm/residual/`. No API calls.

## Test 1: heterogeneity / mover-stayer

**a) Nothing agent-specific is sent (prediction confirmed).**
- All 140,005 Stage 3 prompts are byte-identical to the deterministic render of (framing, configuration, own
  position, letter mapping, option order, listed contact order): 0 differ.
- Each point has exactly one request body apart from the message.
- The template has no names, IDs or fixed labels. The cache tag is a local key and is never sent.
- So the only things that differ between agents are their configuration, own position and the per-call random
  presentation.

**b) Between-agent variance (pre-stated test: labels shuffled within point, 1,000 permutations).**

| point | agent: z (p) | graph position: z (p) |
|---|---|---|
| P1 | +2.33 (0.013) | +3.40 (0.001) |
| P2 | −1.01 (0.84) | −0.83 (0.80) |
| P3 | −1.94 (0.99) | −1.15 (0.89) |
| P4 | +2.16 (0.019) | +1.40 (0.085) |
| P5 | +0.41 (0.34) | +0.62 (0.25) |
| P2-nc | −1.27 (0.91) | −1.15 (0.88) |
| P4-nc | +2.74 (0.005) | +2.78 (0.005) |

Fisher combined p: agent 0.008, graph position 0.003. **This contradicts the pre-registered prediction**, so c) was
triggered.

**Robustness of b) (post-hoc, labelled).** The pre-stated null shuffles labels across all rows of a point. But an
agent keeps revisiting the same configurations, and the residual spread differs between configurations.
Re-shuffling labels only within (point, c, s0) cells, at both the smoothed (n0 = 20) and the raw (n0 = 0) kernel:

- P1: z +0.2 to −0.1, p 0.42–0.51 (gone);
- P4: z +1.0 to +1.2, p 0.12–0.16 (gone);
- P2, P3, P5, P2-nc: p 0.30–0.90;
- **P4-nc: z +1.8 to +2.4, p 0.012–0.044 (remains).** This is the Io Net no-contacts arm. Io Net is also the one
  provider whose concurrent calls share a small offset (`KERNEL_TEST_RESULTS.md`).

Most of the pre-stated signal is therefore agents clustering in configurations, not agent-level heterogeneity.
P4-nc keeps a small effect.

**c) Heterogeneous kernel (pre-stated; run because b) was significant).**
- Per-agent logit offsets on P(stay) ~ N(0, τ²), with τ from b)'s statistic: P1 0.10, P4 0.11, P5 0.11,
  P4-nc 0.06, P2/P3/P2-nc 0.
- Simulated on the live graphs and starts, R = 400.

| Stouffer over P1–P4 + P2-nc + P4-nc | homogeneous K_live | heterogeneous K_live |
|---|---|---|
| persistence (live − sim) | +2.72 | +2.84 |
| flip rate | −2.77 | −2.83 |

**Heterogeneity does not explain the excess** (pre-stated criterion: heterogeneous Stouffer < 1.645 while homogeneous
≥ 1.645; not met). Offsets of this size barely move the simulated persistence, and the per-point z-values are
essentially unchanged (P3 +1.9 → +1.8, P4 +1.8 → +1.9, P4-nc +2.3 → +2.5).

## Where this stands on the interpretation map

Test 1 reads "no". The remaining branches depend on test 2:
- excess repeats → unexplained non-Markov persistence, reported as an open anomaly;
- excess does not repeat → the original misses were realisation noise, and the narrowed headline applies.

Test 2 has not been started. Cost options are in the plan.

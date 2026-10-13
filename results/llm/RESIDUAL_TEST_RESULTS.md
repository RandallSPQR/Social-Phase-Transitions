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

## Identical-request dependence test (amendment 1 §4; free; run before any paid call)

**How each live choice was produced.**
- Logprob endpoints (P1–P4, P2-nc, P4-nc): the provider returns the first-token top-5 logprobs, and the provider's
  sampled token is ignored. The choice is a local draw, A if `rng.random() < pA`, with the chain's Python
  `random.Random` (Mersenne Twister) seeded by SHA-256 of ("stage3-chain", point, graph, replica).
- P5: the provider-sampled reply (T = 1), parsed to one letter.

**Result** (`llm/residual_identical_requests.py`, `results/llm/residual/identical_requests.json`). Repeat pairs:
2,622–15,017 per point.

| point | pairs | earlier choice was "stay" | excess (ii), pre-stated [95% CI] | excess (iii), selection-corrected [95% CI] |
|---|---|---|---|---|
| P1 | 4,559 | 99.2% | +0.0055 [−0.0002, 0.0105] | −0.0000 [−0.0042, 0.0040] |
| P2 | 2,622 | 98.5% | **+0.0223 [0.0128, 0.0310]** | −0.0009 [−0.0067, 0.0049] |
| P3 | 5,400 | 100% | +0.0001 [−0.0004, 0.0007] | −0.0002 [−0.0008, 0.0004] |
| P4 | 6,759 | 99.2% | **+0.0054 [0.0018, 0.0089]** | +0.0026 [0.0004, 0.0046] |
| P5 | 7,684 | 96.9% | +0.0012 [−0.0033, 0.0056] | −0.0010 [−0.0048, 0.0025] |
| P2-nc | 15,017 | 99.7% | **+0.0225 [0.0199, 0.0257]** | +0.0024 [0.0005, 0.0044] |
| P4-nc | 14,308 | 98.1% | **+0.0237 [0.0188, 0.0281]** | +0.0014 [−0.0020, 0.0046] |

- **The pre-stated expectations (i) and (ii) are biased.** A pair exists only if the agent's whole prompt repeats,
  including its own position, which almost always means its earlier choice was "stay" (97–100% above). The pair is
  therefore selected on the earlier choice.
  - Under independence the later call agrees with probability P_later(earlier letter), i.e. its probability of
    staying, not P1·P2 + (1 − P1)(1 − P2).
  - (iii) conditions on the observed earlier choice and is the valid test. The correction is post-hoc and labelled.
- **Selection-corrected, there is no dependence.** Excess is between −0.001 and +0.001 at five points. P4 and P2-nc
  show +0.0026 and +0.0024. Both are logprob endpoints, where the agreement comes from local, independently seeded
  draws, so this is luck in the realised draws, not a provider mechanism. It matches the realised-flip deficit
  against the calls' own probabilities found earlier (`neighbourhood.json`, N3).
- **Re-simulation** (pre-stated trigger fired under (ii); `llm/residual_repeat_rule.py`). Generous rule: when an
  agent's configuration and own position repeat after a stay, P(stay) rises by the measured excess.
  - With the biased (ii) excess the rule overshoots wildly: live-vs-sim persistence Stouffer −13.4, flip rate +24.5.
    So the (ii) excess is incompatible with the data.
  - With the corrected (iii) excess: persistence Stouffer +1.74 (homogeneous +2.72), flip rate −0.90. It does not
    reproduce the excess (still ≥ 1.645), and the part it absorbs is draw luck at two logprob points.
- **Outcome map → "no excess"**: option B runs with sentinels as approved, with provider drift as the leading
  remaining hypothesis. Realisation noise in the original local draws contributes part of the excess at P4 and
  P2-nc; the replicate's fresh draws address that.

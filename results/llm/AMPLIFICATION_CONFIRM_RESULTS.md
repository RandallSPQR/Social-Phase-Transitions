# Confirmatory amplification test (corrected rule): results, Steps 4–5

Plan: `llm/AMPLIFICATION_CONFIRM_PLAN.md` (76aad78) + amendments 1–3. Runners and analysis were committed before the
results existed: `llm/confirm_step45.py` (a2-runner commit) and `llm/confirm_analyse.py`. Steps 1–3:
`results/llm/AMPLIFICATION_CONFIRM_STOP2.md`.

**The Stage B verdict stays "refuted".**
- Cause: the growth-ratio comparison against the controls is ill-defined when a control's amplification is ≈ 0. P2's
  majority-shift amplification was 0.002–0.010.
- This phase tested a corrected rule on fresh graphs and seeds. No Stage B run is reused.

**Framing:**
- **Whether a population has a tipping point at all is model-dependent:**
  - 2 of 4 confirmatory models have one in the swept range: P1 Qwen-122B and Gemini 3.1 Flash-Lite;
  - GPT-4o-mini and Gemma-31B do not;
  - this is only knowable at the population level.
- The T axis is **decision noise**, not a realistic sampling temperature.
- Stage 3 ran at T = 1, deep in P1's ordered phase (T/T_c ≈ 0.5), where real eval errors had small collective effects
  (+0.004 in |m|). The headline concerns what happens as a population approaches a tipping point.
- Random-regular d = 4 graphs only.

## 1. Verdicts (confirmatory: P1 fresh, Gemini)

All at T_c, N = 1,600, per-step error δ = 0.1 nats, in |m| per nat.

**Rules:**
- **A (decisive):** D = A(model) − max(A(P2), A(P4)); pass iff D ≥ 0.10 and the 95% CI excludes 0.
- **B:** N-growth G = A(1600)/A(100), tested only where A ≥ 0.10 at both N; pass iff the CI lower bound is > 1.
- **C:** peak within ±10% of T_c.

| model | shape | A(model) | A(P2) | A(P4) | D [95% CI] | **A** | G [95% CI] | B | peak T | C |
|---|---|---|---|---|---|---|---|---|---|---|
| **P1** (T_c 2.01) | stay-bias | 0.31 | 0.08 | 0.16 | 0.15 [0.09, 0.22] | **pass** | 1.36 [0.96, 2.08] | fail | 1.93 | pass |
| | majority-shift | 1.10 | 0.005 | 0.31 | 0.79 [0.68, 0.90] | **pass** | 1.74 [1.44, 2.11] | pass | 1.93 | pass |
| | random noise | 2.18 | 0.15 | 0.09 | 2.03 [1.96, 2.10] | **pass** | 11.7 [9.2, 14.4] | pass | 2.01 | pass |
| | real: K_fixed − K_avg (λ 1) | 0.62 | — | — | (no comparable control shape) | — | 1.73 [1.43, 2.12] | pass | 1.93 | pass |
| **Gemini** (T_c 1.83) | stay-bias | 1.35 | 0.09 | 0.15 | 1.20 [1.12, 1.28] | **pass** | 3.43 [3.02, 3.94] | pass | 1.83 | pass |
| | majority-shift | 1.30 | 0.003 | 0.27 | 1.03 [0.94, 1.13] | **pass** | 3.75 [3.15, 4.61] | pass | 1.83 | pass |
| | random noise | 1.33 | 0.16 | 0.10 | 1.17 [1.10, 1.25] | **pass** | 9.5 [7.8, 12.1] | pass | 1.83 | pass |

**Per model:**
- **P1:** amplification at the tipping point exceeds the controls for **3 of 3** synthetic shapes. B passes for 2 of 3
  (stay-bias fails: its G CI includes 1). C passes for all.
- **Gemini:** **3 of 3** on A, B and C.
- **Real fixed-order battery error (P1):** grows with N (G 1.73) and peaks at T_c.
- **Against the plan's predictions** (§5): P1's A passes as predicted for all shapes. Its B was predicted to pass for
  every shape, but stay-bias missed (CI lower bound 0.96).
- **Close margin:** P1 stay-bias D = 0.152 passes, but its CI lower bound (0.09) is below the 0.10 margin. The rule
  requires the point estimate ≥ 0.10 and the CI > 0, so the pass is close.

**Persistence (secondary measure):**
- P1: A passes for 2 of 3 (stay-bias D 0.058 is below the margin).
- Gemini: A passes for 3 of 3.
- B mostly not met: G ≈ 1.2–1.9, many CIs include 1.
- C passes for all synthetic shapes.
- Details: `results/llm/confirm/results/step4.json`.

### Negative control: GPT-4o-mini (real model, no transition; amendment 3.2)

- **|m|: "below floor everywhere".** Over all 26 band temperatures, max A(1600) is 0.075 (stay), 0.089 (majority) and
  0.096 (noise), all under the 0.10 floor. So no growth test applies.
- **Bounded:** max/median over T = 1.8–2.3.
- **The test models beat it at their T_c:**
  - P1: D = 0.28 / 1.05 / 2.13;
  - Gemini: D = 1.32 / 1.25 / 1.27;
  - every CI excludes 0.
- **Persistence:** A crosses the floor for stay and majority, but there is **no upward N-growth anywhere**. A few CIs
  exclude 1 from **below**, e.g. G 0.54 [0.37, 0.87]: amplification shrinks with N. The literal prediction "CI
  includes 1" therefore fails there, in the direction opposite to growth.

### Step 5: the low-T freezing edge of P1 (decision-noise rescaling, not verified sampling temperature)

- **Verdict: finite-time effect.**
- At 4× run length (W = M = 2,000), **no shape** has a T where the growth CI lower bound is > 1.
- Majority-shift growth at 1× (G 3.4 [1.3, 39] at T = 0.61) disappears at 4×: amplification at N = 100 falls below
  the floor everywhere.
- Stay-bias max G is 0.51 → 0.64; noise is 1.49 → 2.30, but its CIs include 1.
- The Stage B G of 12–22 at T ≈ 0.6–0.8 was a finite-run-length effect, not a second dynamical transition.

## 2. Exploratory (labelled; not confirmatory)

**GPT-6-luna** (T_c 0.49; temperature refused by the endpoint, so its T axis is a **defined decision-noise rescaling,
not verified sampling temperature**):
- Rule A passes for 3 of 3 shapes (D 0.20 / 0.42 / 1.11). C passes only for noise. B is tested only for
  majority-shift: G 2.41 [1.70, 3.65].
- Its N = 1,600 curve is flat across the band, so there is no clear peak.

**Gemma-31B** (no transition in the swept range; frustrated/glassy below T ≈ 3.2):
- **(a) Run length:** the hot/cold |m| gap at N = 1,600 shrinks with run length.
  - T = 2.0: 0.28 / 0.21 / 0.16 at 1× / 2× / 4×.
  - T = 2.74: 0.11 / 0.04 / 0.01.
  - T ≥ 2.96: ≤ 0.02 throughout.
  - Under the A3 rule, start-dependence **persists** at T ≈ 2.0–2.3 (gap > 0.02 and not halved) and dissolves above
    T ≈ 2.7.
- **(b) Outcome sensitivity** (which basin a population reaches, sign of m):
  - A 0.1-nat per-step error changes the basin in **24–51%** of paired runs from identical starts.
  - A different dynamics seed alone changes it in **43–60%**.
  - There is no consistent trend with N. The undecided share (|m| < 0.05) rises with N at low T, to 0.55 at T = 1,
    N = 1,600.
  - **Reading:** in Gemma's glassy regime the final collective state is close to a coin flip set by the stochastic
    path. A small per-step error moves outcomes about as much as intrinsic noise does. It is not a sharp,
    N-amplified effect.

## 3. Labels and caveats

- **"Defined decision-noise rescaling, not verified sampling temperature"** applies to: Step 5, luna's T_c and band,
  the P2/P4 controls at luna's T_c, and GPT-4o-mini's low-T range. Reason: 1/T fails at T = 0.5 beyond reference noise
  for every logprob model checked.
- **Qwen-122B:** narrow slope miss at T = 1.5 (CI 0.68–1.08 vs 0.67), though within reference noise.
- **GPT-4o-mini:** borderline selection. Its dated alias, the same weights, failed the temperature probe.
- **Gemma:** the fingerprint gate was corrected to the PREREG form after it fired (owner decision).
- **Spend for the phase:** API $3.33, GPU $0.91, simulations free.

Figures: `results/llm/confirm/results/fig_step4_amplification.png`, `fig_ruleA.png`;
`results/llm/confirm/fig_binder_step3.png`.

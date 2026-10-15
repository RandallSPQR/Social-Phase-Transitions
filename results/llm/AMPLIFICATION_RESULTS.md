# Amplification curve near the transition: results

Plan: `llm/AMPLIFICATION_PLAN.md` (committed 18f619e, before any run). Amendment 1 (eef27e6) was committed after
stage A and before any perturbation run. Code: `llm/amplification_curve.py`, `llm/amplification_analyse.py`.
Outputs: `results/llm/amplification/`. Simulation only, no API calls.

## Transition (stage A2, base kernels)

- **P1 (Qwen-122B neutral, ρ = 0.25): T_c = 1.93 (95% CI 1.92–1.96).** This is the U(400)–U(1600) Binder crossing.
  The checks agree: U(100)–U(400) crosses at 2.02, and the scaled m² at 2.02.
- **Below T ≈ 0.65 the dynamics are frozen.** At N = 1,600, hot-start |m| is 0.07–0.45 while cold-start |m| is 0.99.
  This is non-ergodic dynamics, not a disordered phase.
- **P2 and P4 show no crossing in [1, 3].** Both are ordered by their content field (fig. 1).
- **The Stage 3 temperature (T = 1) is at T/T_c = 0.52, deep in the ordered phase.**
- **Stage B:** W = M = 500 sweeps (hot/cold gap ≤ 0.017) and R = 400 paired runs per setting.

## Pre-registered decisive test (P1, |m|, δ = 0.1 nats, T* = 1.93)

| shape | G = A(1600)/A(100) | H1 (CI > 1) | G / G_P2 | G / G_P4 | H2 |
|---|---|---|---|---|---|
| stay-bias | 3.0 [2.3, 4.3] | yes | 2.9 [2.2, 4.2] | 3.1 [2.3, 4.4] | yes |
| majority-shift | 3.4 [2.8, 4.2] | yes | 0.55 [0.05, 3.8] | 3.5 [2.9, 4.4] | **no** |
| random noise | 5.5 [4.7, 6.7] | yes | 4.9 [4.1, 5.9] | 5.4 [4.6, 6.6] | yes |

**Verdict as pre-registered: refuted on H2.** H1 holds for all three shapes. H2 fails for majority-shift, because the
ratio against P2 has a CI that includes 1.

**What drives the H2 failure** (post-hoc reading, labelled as such):
- P2's majority-shift amplification at T* is close to zero. A = 0.002 at N = 100 and 0.010 at N = 1,600; P1's is
  1.94 at N = 1,600.
- With a near-zero denominator, P2's G (6.2 [0.9, 68]) and the ratio are uninformative.
- The data do not show P2's growth matching P1's. They show that the ratio statistic cannot decide this comparison.
  The pre-registered verdict stands as written.

**Persistence (separate verdict, not the headline): refuted on H2.**
- H1 holds for all three shapes: G = 1.6, 1.8 and 3.1.
- H2 fails for majority-shift (vs P2: 1.3 [0.35, 2.5]) and for noise (vs P2: 1.5 [0.8, 3.3]).

### Absolute amplification at T* (|m| per nat; equilibrium protocol)

| shape | P1 N=100 | P1 N=400 | P1 N=1600 | P2 N=1600 | P4 N=1600 |
|---|---|---|---|---|---|
| stay-bias δ 0.1 | 0.29 | 0.55 | 0.86 | 0.08 | 0.16 |
| majority-shift δ 0.1 | 0.57 | 1.34 | 1.94 | 0.01 | 0.30 |
| random noise δ 0.1 | 0.46 | 1.61 | 2.54 | 0.15 | 0.08 |
| K_fixed − K_avg (λ 1) | 0.50 | 0.78 | 1.27 | 0.09 | 0.13 |
| A3 − K_avg (λ 1) | 0.33 | 0.43 | 0.36 | 0.10 | 0.47 |

## Stated expectations (reported, not decisive)

- **(i) Peak at T_c: holds.** argmax_T A(T, 1600) = 1.93 for all three shapes.
- **(ii) Bounded away from T_c: fails.**
  - Above T_c, random noise keeps G ≈ 4–6. Noise breaks the ±m symmetry like a field; that is a finite-size effect of
    |m|, and A itself falls 2.5 → 0.06.
  - Near the freezing edge (T ≈ 0.6–0.8), G reaches 12–22: a second, dynamical tipping point.
- **(iii) Controls bounded: fails.** P4's G is huge at T ≤ 0.4 (A at N = 100 ≈ 0, saturated order), and P2's
  G rises to 2–3 at T ≥ 3. Both come from near-zero denominators.
- **(iv) Persistence: same pattern, weaker.** Verdict above.
- **(v) Real shapes at T*: K_fixed holds; A3 shows no growth.**
  - K_fixed − K_avg: G = 2.5 [2.2, 2.9] for P1, against 0.94 (P2) and 0.92 (P4).
  - A3 − K_avg: G = 1.07 [1.02, 1.13], no meaningful growth.
- **(vi) Linear response: holds.** The median A(0.1)/A(0.03) away from T_c is 1.01.

## Placement of the real eval errors at T = 1 (Stage 3)

At T = 1, P1 is deep in the ordered phase and amplification of |m| is small.

| run | per-step error ε (nats) | Δ|m|, N = 100, equilibrium | Δ|m|, N = 100, Stage 3 protocol |
|---|---|---|---|
| K_fixed | 0.48 | +0.004 | −0.011 |
| A3 | 1.71 | −0.034 | +0.030 |

The same K_fixed error shape at T_c (ε = 0.13) moves |m| by +0.07 at N = 100 and by **+0.17 at N = 1,600**. That is
about 40× the T = 1 effect per run, and about 140× per nat.

> **Correction (2026-10-15).** The "40×" and "140×" figures compared N = 1,600 at T_c with N = 100 at T = 1, so they
> mixed the temperature effect with N-dependence. At the same N:
> - N = 100: +0.068 vs +0.004 per run (about 16×), and 0.50 vs 0.009 per nat (about 56×);
> - N = 1,600: +0.169 vs −0.002, where the T = 1 effect is indistinguishable from zero; per nat 1.27 vs 0.004
>   [0.000, 0.017], so at least about 75×.
>
> Source: `amplification.csv`, equilibrium protocol, `fixed`, λ = 1.

At T = 1, the controls' real errors are of a similar or larger size per nat. For example, P2's A3 gives A = 0.27, and
P4's A3 gives Δ|m| = −0.044. In P1 at T = 1, persistence carries more of the effect than |m| does: K_fixed gives
+0.03 and A3 gives −0.08 (Stage 3 protocol, N = 100).

## Reading

The amplification is sharply state-dependent:
- At P1's order–disorder point, per-step errors are amplified into |m| roughly 10–100× more than at T = 1.
- The amplification grows with N at that point (3–5.5× from N = 100 to 1,600), for synthetic and real (fixed-order
  battery) error shapes alike.
- The controls do not show this.

The pre-registered rule nonetheless returns **refuted**. One of six control comparisons is statistically
uninformative, because the control's amplification is near zero. The "bounded elsewhere" expectations also failed in
two identifiable places:
- symmetry-breaking noise in the disordered phase;
- the low-T freezing edge, which is itself a second tipping point.

Stage 3 ran at T/T_c ≈ 0.5, far from the tipping point. That is consistent with the modest collective consequences
of the measured kernel errors there.

Caveats:
- The 1/T rule is assumed, not verified, for Qwen-122B and Nemo-12B.
- Results are for random-regular d = 4 graphs and the K_avg kernels.

Figures: `fig1_binder.png`, `fig2_amplification_{abs_m,persistence}.png`, `fig3_growth_{abs_m,persistence}.png`,
`fig4_placement_{abs_m,persistence}.png`.

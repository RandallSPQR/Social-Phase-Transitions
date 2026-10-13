# Amplification curve near the transition: plan and pre-registered predictions (committed before any run)

Owner spec 2026-10-13. The replay audit passed (`results/llm/stage3_rep1/replay_audit/`), so this is the next step.
Simulation only, no API calls. Code: `llm/amplification_curve.py` (simulation) and `llm/amplification_analyse.py`
(analysis, written and committed before stage B results exist). Outputs: `results/llm/amplification/`.

The headline claim under test: small per-step errors in an agent's evaluated response are amplified into collective
errors, *especially near tipping points*. This needs a transition axis and system size, not only per-point gaps.

## 1. Control axis, kernels, systems

- **Points.**
  - Test point: **P1** (Qwen-122B, neutral framing, ρ = 0.25).
  - Controls: **P2** (Qwen-9B, political, ρ = 0.5) and **P4** (Nemo-12B, workplace, ρ = 0.25).
  - Why the controls should show no sharp transition: P2's ρ = 0.5 leaves no net ferromagnetic coupling, and both
    controls carry a content field that rounds any transition.
- **Base kernel at temperature T:** K_avg (the order-marginalised battery, `k_avg_calls.csv`).
  - Every per-prompt letter logit is divided by T. The two reps of a prompt are averaged first.
  - The result is then marginalised over mapping, option order and contact order exactly as at T = 1.
  - Configurations never seen live fall back to the Stage 1 battery, tempered the same way.
  - At T = 1 the tables equal the committed K_avg, K_fixed and A3 tables (checked: max |ΔP| ≤ 2·10⁻¹⁶).
- **The 1/T rule is an assumption here.**
  - It was validated on Qwen-9B (EXECUTION_LOG entry 7, step 1b).
  - It is assumed, not verified, for Qwen-122B (P1) and Nemo-12B (P4). The GPT-4o-mini χ² test failed (entry 1).
- **Graphs:** random-regular, d = 4, rival ties with probability ρ (ε = 0).
  - N = 100: the five Stage 3 graphs, with fingerprints asserted.
  - N = 400 and 1600: five graphs each, from `make_graph` with seeds `amp-graph|point|N|g`, built with networkx 3.7.
  - All fingerprints are written to `graph_fingerprints.json`.
- **Dynamics:** random-sequential updates, P(+1) = σ(L[c, s0]), random 50/50 start.
  - **Equilibrium protocol (primary):** 200 warm-up sweeps, then 200 measured sweeps.
    - |m|, m² and m⁴ are averaged over the measured sweeps.
    - Persistence is the overlap with the end-of-warm-up state, averaged over the first 10 measured sweeps (the
      Stage 3 window length).
    - Flip rate is recorded.
  - **Stage 3 protocol (secondary; used for placement in §5):** 10 warm-up and 10 measured sweeps, as Stage 3.
  - Equilibration check, reported: |m| in the first vs second half of the measured window at T_c, N = 1600.

## 2. Locating T_c (stage A, base kernels only)

- **Coarse grid:** 15 geometric points, T from 0.2 to 5.0. R = 200 runs per (point, N, T).
- **Binder cumulant:** U = 1 − ⟨m⁴⟩ / 3⟨m²⟩². Each moment is time-averaged within a run, then averaged over runs.
- **T_c** is the crossing of U(400) and U(1600), by linear interpolation in log T.
  - The 100–400 and 100–1600 crossings are reported too.
  - 95% CI from 200 run-bootstraps.
- **The final grid** is fixed mechanically from P1's T_c:
  - 9 geometric points over [T_c/1.6, 1.6·T_c];
  - plus the coarse points outside that band;
  - plus T = 1.
  - The same grid is used for all three points.
- **If P1 shows no U(400)–U(1600) crossing on the coarse grid:**
  - The prediction cannot be tested as specified. Report that and stop.
  - No redefinition of the transition after the fact.

## 3. Perturbations at each (point, N, T) (stage B)

All perturbations act on the tempered base logit table L_T. Per-step size is measured in delivered nats.

- **(a) Synthetic**, size δ ∈ {0.01, 0.03, 0.1, 0.2, 0.5} nats:
  - **stay-bias:** L + δ·s0;
  - **majority-shift:** L + δ·sign(h), where h = (n_aP − n_aQ) − (n_rP − n_rQ) is the socially implied direction;
    ties get 0;
  - **random per-configuration noise:** L + δ·ξ_k, with ξ ~ N(0, 1) per table entry. There are 3 fixed shapes
    (k = 0, 1, 2), the same at every δ, T and N.
- **(b) Real error shapes**, scaled by λ ∈ {0.25, 0.5, 1}:
  - L + λ·(logit K_x,T − L_T), for K_x = K_fixed (the fixed-order Stage 1 battery) and K_x = A3 (the fitted trait
    summary, η/T);
  - λ = 1 is exactly that kernel at T.
- **Per-step size:** ε = Σ_c w_c |ΔL_c|.
  - w_c is the configuration visit share under the base kernel at that (point, N, T).
  - For stay-bias, ε = δ.
- **Runs:** R = 400 per setting. Run r uses graph r mod 5.
- **Common random numbers:** run r of every setting uses the same seed, so the start, the site sequence and the
  uniforms are shared. Δ is therefore a paired difference.
- **Resumable:** each (stage, protocol, point, N, T) unit is saved as soon as it finishes.

## 4. Measure

- **Amplification:** A = |Δ⟨X⟩| / ε, for X = |m| and X = persistence.
  - Δ⟨X⟩ = mean over runs of (X_perturbed − X_base), using the paired runs.
  - Absolute units, which is primary. SD units (Δ / SD of the base runs) are reported as secondary.
  - Noise shapes: A = mean over the 3 shapes of |Δ_k| / ε_k.
- **CIs:** 95% percentile bootstrap over runs (1,000 resamples; paired settings resampled together).
- **N-growth ratio:** G(T) = A(T, 1600) / A(T, 100). Its CI comes from independent bootstraps of the two N.

## 5. Pre-registered prediction and decision rule

- **Prediction:** amplification peaks near T_c and grows with N there. Away from T_c, and at P2 and P4, it stays
  bounded in N.
- **Decisive test:** P1, X = |m|, equilibrium protocol, δ = 0.1 nats, at T\* = the grid point nearest T_c (in log T).
  - **H1 (N-growth near T_c):** G(T\*) has a 95% CI lower bound > 1 for at least 2 of the 3 synthetic shapes.
  - **H2 (not shared by the controls):** for each shape that meets H1, G_P1(T\*) / G_ctrl(T\*) has a CI lower
    bound > 1 against **both** P2 and P4.
    - The control's G is computed on the same T.
    - The ratio's CI comes from the joint bootstrap.
- **Verdicts:**
  - Supported: H1 and H2 both hold.
  - **Refuted if the N-growth near T_c is absent (H1 fails) or matches the controls (H2 fails).**
- **Stated expectations, reported but not decisive:**
  - (i) The P1 peak: argmax_T A(T, 1600) lies within ±1 grid step of T\* for the H1 shapes.
  - (ii) Away from T_c: P1 G(T) < 2 for T ≤ T_c/1.6 or T ≥ 1.6·T_c.
  - (iii) P2 and P4: G(T) < 2 at every T.
  - (iv) Persistence shows the same pattern. The same H1 and H2 rules are applied to it and given a separate
    verdict, which is not the headline.
  - (v) The real error shapes at λ = 1 show the same N-growth near T_c.
  - (vi) Linear response: at δ ≤ 0.1, A is roughly independent of δ away from T_c.
- **Not a test:** whether T = 1 (Stage 3) lies near T_c. That is placement (§6), not a prediction.

## 6. Placement of the real eval errors

- At **T = 1** (Stage 3's temperature: the letter is drawn locally from the T = 1 logprobs), for each point and N:
  - the per-step size ε and the collective Δ of K_fixed − K_avg and A3 − K_avg at λ = 1;
  - both protocols;
  - placed on the synthetic Δ-vs-ε curves.
- Also reported: where T = 1 sits relative to T_c (T/T_c with its CI), and the measured K_fixed and A3 per-step gaps
  at T = 1 from the kernel test, for reference.

## 7. Figures (Tufte style)

1. Binder U(T) for N = 100, 400, 1600, as small multiples per point. T_c is marked.
2. A(T) for |m|, as small multiples (rows: shape; columns: point). One line per N, direct labels, T_c and T = 1
   marked.
3. G(T) per point and shape, with CIs, and the reference lines G = 1 and G = 2.
4. Placement at T = 1: |Δ| vs ε on log–log axes. Synthetic curves as lines, real error shapes as labelled points.

Ink: no gridlines, no boxes, range frames, direct labels, a single accent colour for P1.

## 8. Deviations

Any change to the above is logged in `llm/EXECUTION_LOG.md` as a deviation before the affected result is looked at.
Compute is about 2 CPU-hours on this container (benchmark: 3 / 11 / 25 ms per equilibrium run at N = 100 / 400 /
1600). If the container is reclaimed, the run resumes per unit.

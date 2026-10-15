# Gemma-31B frozen, start-dependent regime: exploratory mechanism test (plan; committed before any run)

Owner request 2026-10-15. **Exploratory, labelled; it cannot change any confirmatory verdict.** Simulation only, no API
calls. Code: `llm/gemma_mechanism.py`. Outputs: `results/llm/confirm/gemma_mechanism/`.

## Setting (same as the A3.3 run-length check)

- Kernel: Gemma-31B neutral K_avg (`confirm_sim.table("gemma", T)`).
- Graphs: the fresh `confirm-graph` set, 5 per N, ρ = 0.25, N = 100, 400, 1,600.
- **T ∈ {2.0, 2.163, 2.3392}**, the A3.3 grid points in 2.0–2.3, where start-dependence persisted at 4×.
- **Starts:** hot (random 50/50) and cold (all A). R = 50 runs per start, run r on graph r mod 5. Seeds
  `gemma-mech|kernel|N|T|W|start|r`; the same seed index is used across kernels.
- **Run lengths:** 1× = 1,000 warm-up + 1,000 measured sweeps; 4× = 4,000 + 4,000.
- **Start-dependence gap:** |mean |m|(hot) − mean |m|(cold)|, with |m| time-averaged over the measured sweeps (the A3.3
  definition).
- **Start-dependence persists at T** if the N = 1,600 gap at 4× is > 0.02 **and** ≥ half its 1× value (the A3.3 rule).

## Kernels

1. **Original:** K_avg as measured, logits ÷ T.
2. **Rival knockout:** rival positions carry no information. For each configuration (allies A, allies B, rivals A,
   rivals B) and own position, P is replaced by its binomial(½) average over every split of the same number of rivals
   between A and B. The number of contacts and rivals is unchanged; rivals' positions no longer matter.
3. **Saturation knockout:** an additive logit fitted to the T-tempered K_avg table at each T. The features are
   intercept, allies-A − allies-B, rivals-A − rivals-B, and own position. The fit is by cross-entropy over the 70
   configurations with equal weights. There are no higher-order terms; the ally, rival and inertia weights are the
   measured ones (fitted). Coefficients are reported.

## Glass signatures (original kernel only, same T, N = 1,600 and 400; W = 1,000 + 1,000, R = 50)

- **Two-replica overlap, identical starts:** two runs from the same random start with different dynamics seeds;
  |q| = |mean_i s_i^a s_i^b| at the final sweep. Reported with |m| of each replica.
- **Two-replica overlap, independent starts** (the standard Edwards–Anderson set-up), as a reference.
- **Excess unsatisfied ties:** time-averaged fraction of unsatisfied directed ties (ally with a different position, or
  rival with the same position) minus the **frustration floor of the same graph**. The floor is the near-zero-
  temperature Ising quench of `stage2.floor_run` (β = 6, 200 + 50 sweeps), run on each of the 5 graphs.

## Pre-stated reading

- **Frustration-driven** only if: the rival knockout **removes** start-dependence (it does not persist at any of the 3
  T) **and** the saturation knockout **does not** (it persists at ≥ 1 T).
- **Higher-order (saturation)-driven** if the reverse holds.
- **Both remove it:** no single mechanism; each ingredient is necessary.
- **Neither removes it:** neither frustration nor saturation alone. Inertia or another feature remains untested;
  mechanism not identified.
- If the original kernel does not reproduce persistence at 4× in this rerun, that is reported first, and the
  knockouts are read against it.
- **Glass signatures are descriptive only.** A large |q| from identical starts with small |m|, and an excess of
  unsatisfied ties well above the floor, are consistent with glassy freezing but do not decide the mechanism.

Report either way.

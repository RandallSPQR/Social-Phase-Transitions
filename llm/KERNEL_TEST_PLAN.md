# Three-kernel test: plan (committed before any kernel is computed)

Owner request 2026-10-12, after the Stage 3 harness audit (`results/llm/STAGE3_AUDIT_RESULTS.md`). Question: is the
Stage 3 miss an **eval mismeasurement** (the one-shot battery measured the per-step response under one fixed
presentation, while live agents see randomised presentations) or evidence that **single-agent evals do not certify
populations**? Implementation: `llm/kernel_test.py`. Outputs: `results/llm/kernel_test/`. K_fixed and K_live are
computed from existing data (no API calls). K_avg needs new calls and waits for the owner's approval of the cost
estimate.

## Kernels

A kernel is a table P(+1 | c, s0) for each point: the probability that an updating agent ends in position P (+1),
given its contact configuration c = (n_allyP, n_allyQ, n_rivalP, n_rivalQ) and its own position s0. The framing is
fixed by the point. Everything else in the presentation (contact order, letter mapping, option order) is
randomised per call in the live protocol, so the kernel is the average over those. On a random-regular graph with
d = 4 there are 35 configurations × 2 own positions (P5, ρ = 0: 5 × 2); in the no-contacts arms only s0.

1. **K_fixed**, primary form: the **Stage 1 battery kernel**. Mean replicate P(choose P) of the perm-0 battery prompt
   (one seeded contact order per cell), averaged over the 4 presentations (2 mappings × 2 option orders). In
   neutral framing, mapping 1 is the relabelled battery cell, as in `stage3.check_prompt`:
   P = 1 − P_bat(A | swapped counts, −s0). P5 uses the 30-sample shares.
   - Side by side, **K_fixed-A3**: the committed Stage 2 surrogate, averaged over L and O (P5 adds its
     first-contact voter weight). Its draws are taken as in Stage 2, one bootstrap draw per graph.
   - Battery uncertainty: logprob endpoints use the measured values as fixed (repeat SD ≤ 0.16 nats). For P5,
     each battery share is drawn from Beta(1 + x, 1 + n − x) per run.
   - The nc arms have no battery cell at k = 0, so K_fixed there is the committed fields-only rule (as Stage 3).
2. **K_live**: the empirical kernel from the 140,000 applied live updates.
   - Per call, the probability of +1: for logprob endpoints the call's renormalised P(letter) mapped to the
     position (pA under mapping 0, 1 − pA under mapping 1); for P5 the sampled outcome. A configuration's raw
     estimate is the mean over its calls (presentations are randomised, so this is the marginal).
   - **Smoothing (hierarchical, pre-set):** posterior mean (Σ p + n0·π_c) / (n + n0), with n0 = 20 pseudo-calls. The
     prior π_c comes from a soft-label logistic fit on the same live calls with terms 1, M_a, M_r, s0, s0·M_a,
     s0·M_r (nc arms: 1, s0).
   - Sensitivity: n0 = 5 and n0 = 100. Configurations never seen live get π_c.
   - Reported: per-configuration counts, and the share of simulated updates landing on configurations with
     n < 20 live calls.
   - Uncertainty: a cluster bootstrap over the 10 chains of a point (resample chains, re-estimate the kernel),
     with one bootstrap kernel per simulated run.
   - In-sample by construction: it tests whether the live system is a Markov chain in (c, s0) with these transition
     probabilities, not out-of-sample prediction.
3. **K_avg** (not run until approved): a new one-shot battery for the P1–P5 endpoints, covering every (c, s0) that
   occurs in the live logs. Each configuration gets up to 8 distinct random contact orders (all distinct orders when
   fewer exist) × 2 option orders × 2 letter mappings (political, workplace). In neutral framing, mapping 1 is
   another configuration's relabelled prompt, so only the option order varies.
   - Same request bodies as Stage 1 (logprob endpoints: 1 call per prompt; P5: 4 samples per prompt).
   - The kernel is the mean over these presentations, with no smoothing.
   - Cost estimate from the actual Stage 3 cost per call, reported before any call.

## Simulation (identical for every kernel)

- The live Stage 3 graphs (5 per point) and the **live initial states** of each replica.
- Random-sequential updates with replacement, N updates per sweep, 10 warm-up + 10 measured sweeps.
- The simulator's own measurement code: `surrogate.run`, with only its update rule replaced by a table lookup on
  (c, s0) and its random start replaced by the live start. The measurement lines run verbatim, as in audit B10.
- R = 400 runs per kernel and point. Each run covers all 5 graphs × 2 replicas and yields one mean-of-5 per
  measure.
- **Interval** = 5th–95th percentile of the 400 mean-of-5 values (90%), conditional on the live graphs and starts.

**"Reproduces"** (the owner's definition: same pass rule and intervals as Stage 3, on the live graphs): the observed
Stage 3 mean of 5 lies inside the kernel's interval for both primary measures at a point. A kernel reproduces the
live system if this holds at ≥ 3 of P1–P4. P5 and the nc arms are reported, not counted.

## Outcome map (pre-registered, owner's wording)

- **K_live fails** → the live system is not Markov in configuration. Investigate hidden state (provider caching,
  within-chain correlation) before claiming anything.
- **K_live reproduces live, K_avg fails** → properly marginalised one-shot evals still mispredict the population:
  headline supported.
- **Both reproduce** → one-shot evals certify populations only when they marginalise over every presentation detail
  the agent is sensitive to; per-step-invisible eval errors are amplified collectively (narrowed headline).
- (Not in the owner's map; recorded so it is not decided after the fact: K_live reproduces and K_avg is not yet run →
  the question stays open, and only the K_fixed vs K_live comparison and the amplification numbers are reported.)

## Amplification (per point)

- **Per-step discrepancy** of K_fixed vs K_live, weighted by how often each (c, s0) occurs live:
  - mean |ΔP| (probability);
  - mean |Δ logit| in nats (clipped at ±6);
  - mean KL(K_live ‖ K_fixed) in nats.
  - The same against K_fixed-A3.
- **Collective discrepancy** in SD units of the K_fixed simulated mean-of-5, per measure:
  - z_obs = (live observed − K_fixed mean) / K_fixed SD;
  - z_kernel = (K_live sim mean − K_fixed mean) / K_fixed SD.
- Reported next to each other per point, primary measures first.

# Social phase transitions: handoff

Owner: Randall. Prepared in a Claude (Cowork) session, 2026-10-07. Next phase: the LLM-agent experiment, run from Claude Code with an OpenRouter key.

## 1. The idea in one paragraph

Five OpenAI proofs from September 2026 (in `papers/`) suggest a chain from local interaction to collective structure:

- **Rokhlin multiple mixing.** Along a single time axis, pairwise decorrelation forces decorrelation at every order. This is the null model: time alone can't build collective structure, so the interaction *network* is necessary.
- **Percolation at criticality and nonuniqueness.** The network either has macroscopic connectivity or it doesn't, and the transition is continuous. On expanding (nonamenable) graphs there is a phase with many permanent giant clusters.
- **Mézard–Parisi for diluted spin glasses.** On a sparse connected network with mixed-sign ties, you get consensus, disorder, or frozen, incompatible "camps" (a spin glass).

The research question is how robust these phases are when ties are **non-reciprocal**, and whether **LLM agents** behave like the spins the theory assumes.

Caveat: all five proofs are unrefereed. They were public for less than a day when this work began. The Z^d percolation result had prior public AI and Lean work: Leder (Aug 2026, anthropics/formal-math) and work prompted by Bou-Rabee (Sept 5).

## 2. Model

Viana–Bray ±J Ising model on Erdős–Rényi graphs:

- N agents, mean degree c, spins (positions) s_i ∈ {−1, +1}.
- A fraction ρ of ties are negative (antagonistic).
- Heat-bath (logit) dynamics at noise T: P(s_i = +1) = 1 / (1 + exp(−2 h_i / T)), where h_i = Σ_j J_{i←j} s_j.

Analytic cavity-method transition lines (J = 1):

- consensus (ferromagnet): c (1−2ρ) tanh(1/T) = 1
- glass: c tanh²(1/T) = 1, which reduces to the percolation threshold c = 1 as T → 0
- multicritical point at tanh(1/T) = 1−2ρ = 1/√c (for c = 4: ρ = 0.25, T = 1.820)

Non-reciprocity ε: with probability ε, the reverse direction J_{j←i} is redrawn independently (still negative with probability ρ). ε = 0 is the symmetric model. For ε > 0 there is no energy function and no Gibbs measure, so only the dynamics are meaningful.

## 3. What has been run and what it showed

### 3a. Symmetric validation (`sim/run_validation.py`, `results/validation_crossings.csv`, figures 1–3)

Method:

- Parallel tempering with 16–20 temperatures and two replicas per graph.
- Measurements taken in the last log-bin (second half of 2^11–2^13 sweeps).
- Transition located at the crossing of N^{2/3}⟨q²⟩ (glass) or N^{1/2}⟨m²⟩ (consensus) for the two largest sizes.
- 95% CI from 1,000 bootstrap resamples over graphs.
- The Binder cumulant is a secondary estimator. It agrees but is noisier.

| Transition | Theory T | Sim T (N = 1000/2000) | 95% CI |
|---|---|---|---|
| consensus ρ=0 | 3.915 | 3.917 | 3.85–4.01 |
| consensus ρ=0.1 | 3.093 | 3.099 | 3.04–3.19 |
| consensus ρ=0.2 | 2.254 | 2.156 | 1.99–2.28 |
| glass c=3 | 1.519 | 1.528 | 1.45–1.58 |
| glass c=4 | 1.820 | 1.813 | 1.74–1.87 |
| glass c=6 | 2.307 | 2.233 | 2.10–2.32 |

All six are inside their CIs. ρ=0.2 (near the multicritical point) and c=6 (too few sizes) sit low and are the loosest; larger N should tighten them. Equilibration was checked by comparing the last two log-bins: flat everywhere except a slight drift at the lowest T for c=6, N=500, which does not affect the crossing.

### 3b. Asymmetry sweep (`sim/run_asym.py`, `results/asymmetry_sweep.csv`, figure 4 left)

Setup: c = 4, plain heat-bath dynamics, 2,000-sweep warm-up, then 2,000 measured sweeps. Persistence = overlap of each replica with its own configuration at the end of warm-up.

**Glass (ρ=0.5):**

| T | persistence at ε=0 | ε=0.1 | ε=0.2 | ε=0.35 |
|---|---|---|---|---|
| 1.0 | 0.51 | 0.28 | 0.11 | ≈0.01 |
| 1.3 | 0.33 | — | — | ≈0 |

There is no size dependence across N = 1k, 2k and 4k. So frozen camps melt quickly under non-reciprocity, consistent with Crisanti–Sompolinsky (1987, dense SK). This is not new physics.

**Consensus (ρ=0.1, T=2.0):** robust. |m| goes from 0.62 to 0.55 as ε goes from 0 to 1.

### 3c. Two-party polarization (`sim/balanced.py`, `sim/balanced_ctrl.py`, `results/balanced_polarization.csv`, figure 4 right)

Two-party polarization is a structurally balanced network: + ties within a party, − ties across. It is gauge-equivalent to a ferromagnet, not a glass. N = 2000, 20 graphs per point.

Non-reciprocity erodes it, but redrawing reverse ties also lowers average loyalty. The control uses symmetric ties with ρ = ε/4, matching the average directed coupling.

At T = 2.0:

| ε | non-reciprocal |m| | control |m| |
|---|---|---|
| 0.2 | 0.71 | 0.74 |
| 0.5 | 0.24 | 0.54 |

At ε = 0.2 lost loyalty explains most of the drop; at ε = 0.5 non-reciprocity itself adds real erosion. These runs are small and suggestive only.

### Known limitations

- Persistence is relative to the measurement window: the ε = 0 glass also ages.
- The consensus/glass boundary is not computed.
- The nonuniqueness phase (many permanent giants) is an infinite-graph phenomenon and is invisible on finite random graphs. Do not claim it from simulations.

## 4. Next phase: LLM agents (the interesting part)

Question: do LLM agents behave like Ising spins? If not, how do they differ: non-reciprocal weighting, non-additive (higher-order) responses, built-in bias? What does the model then predict about the collective phases they can form?

### Stage 1: response-function battery (cheap, the core result)

Each agent holds position A or B. A prompt shows the agent its k neighbors' current positions and each tie's type (ally / rival). The agent picks A or B.

**Battery:**

- k from 1 to 6, all ally/rival × A/B neighbor configurations (deduplicated by symmetry where valid). Include repeated identical neighbors and mixed cases.
- Record P(A) from logprobs where available. Otherwise sample (about 30–50 per configuration) at fixed temperature.

**Fits, per model × framing:**

1. Additive logit: logit P(A) = β (w_ally Σ_ally s_j − w_rival Σ_rival s_j) + h. The test is whether w_rival ≠ w_ally (built-in non-reciprocity or negativity bias) and h ≠ 0 (label bias).
2. Add pairwise interaction terms, e.g. s_j s_k for two allies or ally × rival. Test whether they are significant, using likelihood-ratio tests and held-out log-loss. Significant interactions mean non-additive, higher-order responses (the Rokhlin thread at the agent level).
3. Effective noise: the fitted β as a function of sampling temperature.

**Required controls:**

- **A/B label swap.** Rerun with labels exchanged. A preference for one label is a hidden external field.
- **Neighbor order randomized.**
- **Framings:** neutral (A/B, "agrees with you / opposes you"), political, workplace. Report the fit per framing and flag framing sensitivity.
- **Seeds and repeat consistency.** Check that the same prompt gives the same distribution.
- **Pre-registration.** Write hypotheses and analysis code before running: H1 w_rival ≠ w_ally; H2 h ≠ 0; H3 interactions significant; H4 β increases with model size.

### Stage 2: surrogate prediction

Plug each fitted response rule into the simulator. Generalize `asym.py` to arbitrary per-tie-type weights, bias h and interaction terms. Predict the phase (consensus / camps / disorder / churn) at chosen (c, ρ, temperature). **Write the predictions down before Stage 3.**

### Stage 3: live LLM networks (small, expensive)

- N ≈ 50–100 agents, c = 4, a few hundred asynchronous updates.
- Run at 3–4 points chosen from Stage 2 to discriminate between phases.
- Compare against the surrogate simulator **at the same N** (finite-size matched), using multiple graph seeds.
- Measures: |m|, two-replica overlap q (two runs from different random starts on the same graph), and persistence.

### Model selection

The priority is logprob access, which makes the response function exact and roughly 50× cheaper than sampling.

- **Workhorses:** 3–4 open-weight families (e.g. Qwen, Llama, Gemma, Mistral) at about 8B and about 70B.
- **Spot checks:** one cheap frontier model each from Anthropic, OpenAI and Google, on a reduced battery. Anthropic models don't expose logprobs, so use sampling there.
- On OpenRouter, logprob support varies **by provider**, not just by model. Verify per endpoint and pin providers.
- Check current prices and estimate the cost of each stage before running.

## 5. Repo layout and how to run

```
sim/        simulator and analysis (run scripts from inside sim/)
  vbsim.py            symmetric model + parallel tempering (numba)
  asym.py             non-reciprocal heat-bath dynamics
  run_validation.py   symmetric validation sweep (~3.5 h on 2 cores)
  run_asym.py         asymmetry sweep (~17 min)
  balanced*.py        two-party polarization + matched control
  analyze.py          crossings, bootstrap CIs, equilibration checks
  plots.py, plot_asym.py   Tufte-style figures -> figures/
data/       raw validation output (.npz per tag/size); data/asym/asym.npy
results/    CSV summaries of every run above
figures/    fig1–fig4
logs/       run logs
papers/     the five OpenAI PDFs
```

Dependencies: numpy, scipy, numba, matplotlib. Charts follow Tufte style: off-white background, serif type, direct labels, no legends, CI and n always shown, a source note under each chart.

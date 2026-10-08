# Pre-registration: Stage 1, LLM response-function battery

Owner: Randall. Drafted 2026-10-08, committed before any battery data was collected (provider probes only,
see `llm/LINEUP.md`). Prompt templates and the battery (`llm/battery.py`) and the analysis code
(`llm/analysis.py`) are committed after this file and before the pilot; their state at the commit
preceding the main runs is the frozen version. Any change after this commit is logged under **Amendments** with date and reason.

## 1. Question

Do LLM agents choosing between two positions respond to their network neighbours like heat-bath Ising spins
on a signed graph — logit P(choose +1) = Σ_j J_ij s_j + h with fixed per-tie-type couplings — and if not,
how do they deviate?

## 2. Units and claims

- The unit of every claim is a **(model, provider endpoint)** pair, pinned (`provider.order=[tag]`,
  `allow_fallbacks=false`, `require_parameters=true`). Probes showed the same weights on different
  providers giving logits up to 5 nats apart, so no claim is made about "a model" in general.
- Analysis cells are (endpoint × framing). Framings: **neutral**, **political**, **workplace**.

## 3. Battery

Each prompt is a single user message (no system message). It describes the scenario, the agent's own current
position (except in the no-own control arm), and lists every contact **individually** as a bulleted line,
"An ally holds A" / "A rival holds B" (with a short content gloss in content framings). Counts are never
stated, and lines are bullets, not numbers, so the prompt never hands the model k.

- **Configurations.** The neighbour multiset over 4 types {ally, rival} × {P, Q}, for k = 1…6:
  Σ C(k+3, 3) = **209** multisets. Permutation symmetry is the only deduplication. Global spin-flip pairs are
  both kept, because they identify the external field.
- **Own position** s₀ ∈ {P, Q} → 418 *cells* (multiset × own position).
- **Neighbour order.** Each prompt uses a random permutation, seeded by the SHA-256 of its identity.
  Identical replicates reuse the permutation, so replicate variation is pure endpoint noise.
- **Label controls.**
  - *Mapping* L: which content is lettered A. P = support / adopt, Q = oppose / keep. In neutral the
    positions *are* the letters, so the mapping is fixed.
  - *Order* O: which letter is listed first, in both the scenario and the answer instruction.
- **Arms per endpoint.**

  | arm | cells | variants | prompts |
  |---|---|---|---|
  | neutral, own position shown | 418 | O × 2 | 836 |
  | neutral, no own position (control) | 209 | O × 2 | 418 |
  | political | 418 | L × O = 4 | 1,672 |
  | workplace | 418 | 4 | 1,672 |

  Every prompt is sent **twice** (identical body).
- **Order-sensitivity subset.** For the neutral-own arm, each cell with k ≥ 3 gets 2 extra permutations,
  under the O variant chosen by hash.
- **Decoding.**
  - Workhorses: logprobs at temperature 1 only, `max_tokens=1`, `top_logprobs=5`. Probes found that
    providers disagree on whether returned logprobs are pre- or post-temperature, so the logprob battery
    is never run at T ≠ 1.
  - Frontier spot checks (no logprobs): 30 samples per prompt at T = 1, reasoning disabled,
    neutral + political arms, k ≤ 4.
- **Pilot.** Llama-3.1-8B, about 50 cells, using replicate indices ≥ 100. Pilot data is **exploratory only**
  and never enters a confirmatory analysis. Because of the separate replicate indices, the cache can never
  serve a pilot response to a main run.

## 4. Measurement and exclusions

- **P(A).** Sum the probability of first-token top-5 entries whose whitespace-stripped text is exactly `A`,
  and likewise `B`, then renormalise over {A, B}. Leak = 1 − (p_A + p_B) is recorded per call. If A or B is
  missing from the top 5, its logprob is set to the 5th-ranked logprob (an upper bound) and the call is
  flagged as censored.
- **Leak exclusion.** A prompt is excluded if either replicate has leak > 5%. Exclusion counts are reported
  per endpoint × arm. If more than 10% of prompts in an arm are excluded, that arm is reported but flagged.
- **Endpoint noise gate.** Repeat SD = RMS over prompts of |y₁ − y₂|/√2, where y is logit P(A) clipped to
  ±6. Any endpoint with repeat SD > **0.3 nats** is excluded and replaced by its pre-listed backup
  (`llm/endpoints.py`), which must also pass. If no endpoint passes, that model is dropped.
- **Sampling (frontier).** A reply is valid if, after stripping whitespace, `*`, quotes and a trailing `.`,
  it is `A` or `B`, or starts with `A`/`B` followed by a non-letter. Invalid replies count as leak, with the
  same 5% rule per prompt.
- **Outcome.** Analysis uses y = P(choose content P) = P(A) if P is lettered A, else 1 − P(A). The two
  replicates are averaged on the probability scale.

## 5. Models (estimation)

Spin coding: s = +1 for P, −1 for Q. Define:

- M_a = Σ_{allies} s_j and M_r = Σ_{rivals} s_j
- L = +1 if P is lettered A, else −1
- O = +1 if P's letter is listed first, else −1

**Additive (A0):** logit y = b_a M_a − b_r M_r + γ s₀ + h_C + h_L L + h_O O.

- In neutral, P ≡ letter A, so h_C and h_L collapse into one letter field **h**.
- In the no-own arm γ is absent.
- β := (b_a + b_r)/2, w_ally := b_a/β, w_rival := b_r/β, so w_a + w_r = 2. Only the products βw are
  identified, so this normalisation is a definition, not an estimate.

**Pairwise (A1 = A0 + block P):** Q_aa = Σ_{j<k allies} s_j s_k, Q_rr (rivals), Q_ar (ally × rival pairs),
s₀M_a, s₀M_r (the last two only when s₀ is shown).

- Note: in the logit these products are *even* under a global flip. They are configuration-dependent
  fields, i.e. three-spin couplings s_i s_j s_k in an energy.

**Cubic (A2 = A0 + block C):** the flip-odd three-neighbour sums T_aaa, T_rrr, T_aar, T_arr.

- Within a type these are equivalent to cubic terms in the counts. They capture majority/unanimity
  non-linearity that respects flip symmetry.

**Degree scaling (A3, H5):** logit y = k^(−α) (b_a M_a − b_r M_r) + γ s₀ + fields, with k = number of
contacts. α = 0 is Ising summation; α = 1 is averaging (DeGroot/voter-like).

**Fitting.**

- Workhorses: soft-label cross-entropy (fractional logit), Σ [y log σ(η) + (1−y) log(1−σ(η))], via L-BFGS.
- Frontier: binomial likelihood on the sample counts.

**Uncertainty.** Cluster bootstrap with clusters = cells (multiset × own position). Each draw resamples
cells with replacement and keeps all of a cell's variants and replicates. B = 1,000; 95% percentile CIs;
two-sided bootstrap p-values (2 × min tail share, floored at 1/B).

## 6. Hypotheses and tests

Each hypothesis is tested in every endpoint × framing cell. **Holm** correction at family-wise α = 0.05
is applied across all model × framing cells, separately for each hypothesis (and for each H3 block).

- **H1: valence asymmetry, w_rival ≠ w_ally** (equivalently b_r ≠ b_a). Test: bootstrap p for b_r − b_a.
  Report w_r/w_a with CI.
  - *Scope:* this does **not** test non-reciprocity. In a homogeneous population the same weights apply at
    both ends of a tie, so J_ij = J_ji. See §8 for where non-reciprocity enters.
- **H2: label field h ≠ 0.**
  - Neutral: letter field h.
  - Content framings: h_L (letter) and h_C (content) as separate tests.
  - h_O (first-listed) is reported in all framings.
- **H3: non-additive responses.** Two blocks, P (pairwise) and C (cubic), each tested against A0.
  - Significance: **likelihood-ratio (F) test on the logit scale**. Use cell-mean logits clipped to ±6,
    with the null imposed by a **wild cluster bootstrap** (Rademacher weights by cell, 999 draws).
    Frontier models use the exact binomial LR test, χ²_q.
  - Relevance: an effect counts only if it is significant after Holm **and** clears the
    **minimum-effect threshold**:
    - held-out cross-entropy improves by ≥ 2% relative to A0, under 10-fold cross-validation grouped by
      **multiset** (whole configurations held out); **and**
    - |P_A1 − P_A0| ≥ 0.05 on ≥ 10% of cells (full-data fits).
  - Descriptive: residual SD of A0 on the logit scale vs the pure-error SDs (replicate noise and
    permutation noise from the order subset).
- **H4: β increases with model size.**
  - Primary comparisons, one per family:
    - Llama-3.1-8B vs Llama-3.3-70B
    - Qwen3.5-9B vs Qwen3.5-122B-A10B
    - Gemma-4-26B-A4B vs Gemma-4-31B
    - Mistral-Nemo-12B vs Mistral-Large-4
  - Test: one-sided bootstrap of β_large − β_small, with independent resamples per endpoint; Holm over
    families × framings.
  - Supported overall if significant in ≥ 3 of 4 families in the **neutral** framing.
  - **Confounds** (H4 compares endpoints, not clean size interventions):
    1. *Llama 3.1-8B vs 3.3-70B:* different post-training releases, not one release at two sizes.
    2. *Gemma:* the ladder is in **active** parameters (4B MoE vs 31B dense); total parameters are
       ~26B vs 31B. It also mixes MoE and dense.
    3. *Qwen:* dense 9B vs MoE 122B with 10B active, so the active-parameter gap is small.
    4. *Mistral:* Nemo (2024) vs Large 4 (2026) differ in generation and post-training. Large 4's
       open-weight status is unverified, and it has the highest leak.
    5. *Providers and quantisation:* only the Llama pair shares a provider at full precision. The other
       pairs mix Parasail / Alibaba / Io Net, and fp8 / bf16 / unknown quantisation.
    6. *Decisiveness:* larger models may be more confident overall. β alone cannot separate "more
       responsive to neighbours" from "lower entropy everywhere", so |h| and γ are reported alongside.
- **H5: summation vs averaging.** Estimate α (model A3) with a bootstrap CI.
  - α ≠ 0 (CI excludes 0) is a departure from Ising summation.
  - "Averaging" is declared if the CI excludes 0 and contains 1.

## 7. Secondary and exploratory (no confirmatory claims)

- **Inertia γ:** reported with CI in every cell. Heat-bath spins have γ = 0. γ > 0 slows the dynamics and
  shifts the effective temperature.
- **Own-position control:** b_a, b_r, h from neutral-no-own vs neutral-own (difference with CI). This shows
  whether stating the agent's position changes how it weighs neighbours.
- **Framing sensitivity:** pairwise differences of b_a, b_r, h, γ across framings, with bootstrap CIs.
  Flagged if a CI excludes 0.
- **Order effects:** an extra weight on the first- and last-listed contact.
- **Temperature (effective noise):** sampled sweep at τ ∈ {0.5, 0.7, 1.0, 1.5} on Llama-3.1-8B and
  Qwen3.5-9B (40 cells × 40 samples). Prediction under pure softmax sampling: β(τ) = β(1)/τ, i.e. slope −1
  of log β on log τ.
- **Bridge:** GPT-4o-mini fitted from logprobs and from 30-sample data. This checks that the sampling
  protocol recovers the logprob response function.
- **Provider robustness:** Llama-3.3-70B on Parasail fp8 vs CoreWeave fp16, neutral arms.

## 8. Stage 2/3 plan additions (registered now, run later)

The surrogate simulator generalises `sim/asym.py` to per-receiver couplings J_{i←j} = b_{type(i), tie(i←j)},
plus fields h, γ, any surviving interaction terms, and degree scaling k_i^(−α). Non-reciprocity
(J_{i←j} ≠ J_{j←i}) can then enter three ways, all to be simulated and, where feasible, run live:

- **(a) Mixed-model populations.** Agents from endpoints with different (b_a, b_r) share a graph. A tie
  between a high-gain and a low-gain agent is asymmetric even if both perceive it the same way.
- **(b) Directed tie perceptions.** i sees j as an ally while j sees i as a rival (the ε-redraw of
  `asym.py`). This is implemented in prompts by giving each agent its own view of each tie.
- **(c) Degree-induced asymmetry.** If α > 0, J_{i←j} ∝ k_i^(−α), so on heterogeneous-degree graphs
  ties are asymmetric even in a homogeneous population.

Phase predictions from Stage 2 are written down before any Stage 3 run.

## Amendments

(none yet)

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

**Amendment 1 (2026-10-08, before any battery data; motivated only by synthetic tests in `llm/selftest.py`).**

1. *H3 significance test replaced.*
   - **Problem:** the logit-scale F test needs clipping (±6). Strong responders exceed that at k = 5–6,
     and the clip then acts like curvature. On synthetic *additive* data the cubic block was rejected in
     37% of full-size datasets at nominal 5%.
   - **New primary test:** a cluster-robust **score test** on the cross-entropy fit (efficient score of
     the block, clusters = cells), with p-values from a **wild score bootstrap** (Kline & Santos 2012;
     Rademacher by cell, 999 draws). It is asymptotically equivalent to the LR test and needs no clipping.
   - **Calibration on synthetic null data:** rejection at α = 0.05 was 0.05/0.08 (full size, 60 sets),
     0.07/0.05 (pilot size, 150 sets) and 0.03/0.02 (very strong responders), for the pairwise/cubic
     blocks. Power: 100% at λ = 0.1 on Q_aa.
   - The logit-scale F (wild cluster bootstrap) is retained as a **secondary** statistic.
   - For sampled endpoints the same score test is primary, because the binomial LR test ignores
     between-prompt (order) variation and over-rejects. The binomial LR is secondary.
   - Holm, the minimum-effect threshold and grouped-CV held-out CE are unchanged.
2. *Estimand made explicit (this definition came out of the synthetic coverage check, not from any LLM
   data).* All coefficients are those of the **order-averaged (marginal) response
   function**: the cross-entropy fit to probabilities, with neighbour order randomised. Per-prompt noise
   attenuates these relative to a latent per-prompt logit (by ~2% at noise SD 0.3). The marginal function
   is what governs network dynamics in which neighbour order is random.
   - Bootstrap 95% CIs cover the marginal truth at 0.90–0.97 (30 synthetic full-size datasets). Against
     the latent truth, β coverage was 0, because of the attenuation.
3. *Clarification.* Confirmatory fits use the main battery (perm 0). Order-subset prompts (perm 1, 2) are
   used only for permutation-noise SD and the exploratory order-effect model.
4. *H5 interpretation caveat.* With a planted pairwise interaction, A3 overestimated α (1.30 vs true 1.0),
   because M_a² terms are partly collinear with k-scaling. α is interpreted as summation/averaging only
   where H3 is negative for that cell; otherwise it is reported as descriptive.

**Amendment 2 (2026-10-08, after pilot 1, before any main run). Answer instruction changed.**

- **Problem.** Pilot 1 (Llama-3.1-8B, instruction v1: "Answer with a single letter: A or B.") put 58–69% of
  first-token mass on starting an explanation ("Since", "To", "Given"). The leak **rose with k**: 24% at
  k = 1, 55% at k = 3, 94% at k = 5–6. Every prompt failed the 5% leak rule. Renormalising over {A, B} under
  that leak would bias exactly the k-dependence tested by H5.
- **New instruction (v2):** "Which position do you take now? Reply with exactly one letter, A or B, and
  nothing else." This is the wording used in the provider probes.
- **Check** (`llm/probe_instruction.py`, `results/llm/catalog/instruction_leak_2026-10-08.csv`): 2 cells
  per k on 4 endpoints. Maximum leak under v2 was 1.1% (Mistral Large 4, k = 6), versus 57% under v1.
  Llama-8B went from 19–95% to 0.0%.
- Nothing else in the battery changed. Pilot-1 data are kept in
  `results/llm/stage1/pilot/llama-8b_instruction-v1.csv` and are not analysed further.

**Amendment 3 (2026-10-08, owner-requested, before any main run).**

- **(a) Comprehension check.** This is a separate battery and is not conformity-based (`battery.comprehension_items`,
  `llm/comprehension.py`).
  - Neutral framing, own position shown, both letter orders, k = 1–6. Each prompt has the same scenario and
    contact list as the main battery, then one factual question:
    - "Which position do you currently hold?" (836 prompts)
    - "Which position do most of your allies currently hold?" (640 prompts; strict majorities only)
    - "Which position do most of your rivals currently hold?" (640 prompts; strict majorities only)
  - One call per prompt, logprobs at T = 1.
  - A prompt is correct if the renormalised P(correct letter) > 0.5; a prompt with leak > 5% counts as
    incorrect.
  - **Pass iff accuracy ≥ 90% on each of the three question types.**
  - Endpoints below the threshold are labelled "does not parse task". They are excluded from H4 and from
    every confirmatory Holm family, and their social fits are reported descriptively only.
  - Run on every workhorse and on the GPT-4o-mini bridge.
- **(b) Temperature.**
  - Sampling at temperature T with no top-p/top-k truncation is softmax(z/T). For a two-option choice this
    gives logit P_T = logit P₁ / T, so **β(T) = β(1)/T**, computed analytically.
  - Applicability is classified per endpoint (`llm/temp_check.py semantics`): pre- vs post-temperature
    logprobs. The analytic rule is claimed for endpoints returning pre-temperature logprobs with no
    truncation parameters sent (we send no top_p/top_k); elsewhere it is flagged as assumed.
  - **Verification on GPT-4o-mini** (`temp_check.py verify`):
    - 20 prompts from its neutral battery with P₁ ∈ [0.1, 0.9], chosen in hash order; 50 samples each at
      T = 0.5 and T = 1.5.
    - Pass if, at each T, the 95% CI of the logistic slope of samples on logit₁ contains 1/T, and the
      Pearson χ² (df = 20) has p > 0.01.
  - **The planned sampled temperature sweep** on Llama-8B and Qwen-9B (40 cells × 4 T × 40 samples) is
    **cancelled** if verification passes. If it fails, the owner is asked before any sweep.
- **(c) Interpretation.**
  - **β is largely a sampling knob:** a deployment's temperature rescales it as 1/T. Cross-model differences
    in β at T = 1 therefore mean little for collective phases. The substantive, model-specific quantities
    are:
    - valence asymmetry w_rival/w_ally
    - inertia γ
    - degree scaling α
    - interaction structure
    - the fields h_C, h_L and h_O
  - **H4 has a comprehension confound:** a model that misreads the task has a small or negative β regardless
    of size, so "β rises with size" can reflect parsing rather than social responsiveness. The
    comprehension gate addresses part of this. Any H4 result is reported next to the comprehension
    accuracies.
- **(d) Stage 3 requirement.** Option order (which letter is listed first) is randomised **per call** in live
  networks. Otherwise h_O acts as a global external field. Content fields (e.g. workplace h_C) are
  real-scenario preferences and are kept as measured external fields in Stage 2 predictions.
- **(e)** See the note on Amendment 1 item 2.
- **Wording-robustness arm** (`neutral_own_w2`), with wording "Having considered your contacts, which position
  do you choose? Reply with exactly one letter, A or B, and nothing else."
  - Same 836 prompts and replicates as `neutral_own`; secondary.
  - Coefficient differences vs `neutral_own` are reported with bootstrap CIs, and flagged if a CI
    excludes 0.
- **Spot-check sampling** keeps the pre-registered reduced battery (neutral + political, k ≤ 4, 30 samples,
  T = 1). The spot checks run only if total spend after the workhorse fits is ≤ $12. Spend is the larger of
  the summed per-call costs in the cache and the OpenRouter key usage.

**Amendment 4 (2026-10-09, owner-requested; committed before any GPU run). Self-hosted Gemma-4 in bf16.**

- **Why.** Both Gemma-4 provider endpoints failed the 0.3-nat repeat-noise gate (Parasail 0.70, CoreWeave
  0.35, Io Net 0.304), and the Novita backup returned no logprobs on 77% of calls. Self-hosting removes
  provider jitter and gives logprobs and activations from the same computation.
- **New endpoints: `gemma-26b-local` and `gemma-31b-local`.**
  - Weights: `google/gemma-4-26b-a4b-it` and `google/gemma-4-31b-it`, bf16, HF `transformers`, one RunPod
    GPU pod (`llm/local_gemma.py`, `llm/pod_run.sh`).
  - Prompts: the identical battery (`llm/battery.py`, instruction v2), i.e. all five arms, the
    order-sensitivity subset and the comprehension battery. Each is wrapped with the **official Gemma chat
    template** (`tokenizer.apply_chat_template(..., add_generation_prompt=True)`), user turn only.
  - **One forward pass per prompt** (batched, left-padded); the next-token distribution is read at the last
    position. P(A) uses the Stage 1 rule: top-5 tokens whose stripped text is `A`/`B`, renormalised, with
    leak and censoring recorded. The exact full-vocabulary version (all token ids whose stripped decode is
    `A`/`B`) is recorded alongside, as secondary.
  - **Two replicates** per prompt, run in different batch orders and padding, so batch-composition
    numerics are measured. The 0.3-nat gate applies unchanged.
- **Confirmatory, reported regardless of outcome.**
  1. **Comprehension check and noise gate** exactly as in Stage 1. Only endpoints that pass both enter
     the analyses below.
  2. **H1–H5** with the same analysis code. The confirmatory set is **extended** to the passing local
     Gemma endpoints. Holm families are recomputed over all extended cells. The original 12-cell Stage 1
     results stand as reported; both versions are reported.
  3. **H4 in the Gemma family** (26B-A4B local → 31B local; one-sided bootstrap, Holm over the 3 framings)
     is reported in its own right. The overall H4 rule ("≥ 3 of 4 families") is unchanged. With Llama
     (comprehension) and Mistral (noise gate) excluded, at most 2 families are testable, so the overall
     verdict cannot become "supported" under this amendment.
  4. **Local-vs-provider comparison.** Pairs: local 31B vs Io Net 31B; local 26B vs Parasail 26B; local
     26B vs CoreWeave 26B.
     - For each pair × arm (neutral_own, neutral_noown, political, workplace) and each coefficient
       (b_a, b_r, γ, h / h_C / h_L, h_O, β, α), compute a **paired bootstrap** of provider − local: the
       same cell-resampling indices for both endpoints, B = 1,000.
     - **Tolerance (same as the GPT-4o-mini bridge):** a coefficient "agrees" if its 95% paired-bootstrap
       CI contains 0.
     - **Verdict per pair:** "jitter averages out" if ≥ 90% of coefficients agree and the attenuation test
       below does not reject; "jitter biases estimates" otherwise.
     - **Attenuation test:** regress provider coefficient estimates on local ones across all (arm,
       coefficient) entries, excluding β (a sum of b_a and b_r) and α (a different model). Through the
       origin, weighted by 1/(se²_provider + se²_local), with a cell-bootstrap CI for the slope.
       Prediction from measured jitter σ: κ = (1 + πσ²/8)^(−1/2), i.e. 0.92 Parasail, 0.98 CoreWeave,
       0.98 Io Net. Report whether the CI contains 1, and whether it contains κ.
     - **Confound note:** providers may also differ in quantisation (Io Net's is unreported), chat-template
       handling and serving stack. A disagreement shows that provider estimates differ from bf16 HF
       inference, not that jitter caused it.
- **Exploratory annex (labelled; cannot alter H1–H5).**
  - **Activations:** last-token residual-stream hidden states at 6 layers (≈ 1/6, 2/6, …, 6/6 of depth),
    replicate 1 only, for the main battery (perm 0 and the order subset) and the comprehension battery.
    Stored as fp16 `.npy` with a prompt index.
  - **Probes, per layer:** L2 logistic regression, 10-fold CV grouped by multiset, evaluated by held-out
    AUC.
    - Targets: (a) ally-majority side, (b) rival-majority side (strict majorities), (c) own position.
    - **Controls:** a letter-count probe (number of "A" tokens among the ally, rival and own lines); shuffled
      labels; and probes trained on neutral and tested on political.
  - **"Represented but not used" test:** compare rival-majority decodability with the fitted rival effect
    b_r (and ally with b_a).
    - Supported descriptively if the rival target is decodable (AUC ≥ 0.9 at some layer, and above the
      letter-count control) while |b_r| is small relative to b_a.
    - Plus: the cosine between the rival-majority probe direction at the last layer and the unembedding
      readout W_U[A] − W_U[B].
  - **SAE:** feature lookup only if an existing SAE release covers Gemma-4 (the owner's setup); otherwise
    skipped.
- **Cost control.** A GPU-hour and $ estimate goes to the owner, and no pod is rented without the
  owner's OK. Pods self-terminate when the batteries finish. Pod IDs and spend are logged in
  EXECUTION_LOG.

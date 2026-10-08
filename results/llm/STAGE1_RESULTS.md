# Stage 1 results: LLM response functions (2026-10-08)

Pre-registration: `llm/PREREG.md` (Amendments 1–3, all made before any main-battery data). Post-data events
and judgment calls: `llm/EXECUTION_LOG.md`. Per-cell numbers: `stage1_cells.csv`. H4: `stage1_H4.csv`.
Secondary contrasts: `stage1_secondary.csv`. Bridge check: `bridge_comparison.csv`.
Figures:
- `figures/llm_stage1_comprehension.png`
- `figures/llm_stage1_invariants.png`
- `figures/llm_stage1_H4.png`
- `figures/llm_pilot_llama8b.png`

Total spend: $9.56 (OpenRouter key usage). The per-call cache sum agrees.

## Which endpoints are confirmatory

| endpoint | comprehension (own / ally maj. / rival maj.) | pooled repeat SD | status |
|---|---|---|---|
| Llama-3.3 70B @ CoreWeave fp16 | 1.00 / 0.96 / 0.99 | 0.088 | **confirmatory** |
| Qwen3.5 9B @ Parasail | 1.00 / 0.96 / 0.96 | 0.037 | **confirmatory** |
| Qwen3.5 122B-A10B @ Alibaba | 1.00 / 1.00 / 0.99 | 0.034 | **confirmatory** |
| Mistral Nemo 12B @ Io Net fp16 | 1.00 / 0.98 / 0.94 | 0.137 | **confirmatory** |
| Llama-3.1 8B @ CoreWeave | 0.99 / 0.78 / 0.72 | 0.026 | does not parse task |
| Mistral Small 24B @ Parasail | 0.98 / 0.18 / 0.10 | 0.04 | fails by leak: reasons before answering majority questions |
| Gemma-4 26B-A4B @ Parasail / @ CoreWeave | 1.00 / 1.00 / 1.00 | 0.703 / 0.354 | noise gate failed; no passing backup |
| Gemma-4 31B @ Io Net | 1.00 / 1.00 / 1.00 | 0.304 | noise gate failed; Novita backup returned no logprobs on 77% of calls |
| Mistral Large 4 @ Mistral | 1.00 / 0.98 / 0.97 | >0.30 | noise gate failed; no backup |

**Confirmatory set:** 4 endpoints × 3 framings = 12 cells. Holm correction is applied within each
hypothesis family.

## Hypotheses

- **H1 (valence asymmetry, w_rival ≠ w_ally): supported in 9 of 12 cells, all in one direction: rivals
  weigh less than allies.**
  - In most cells rivals are close to ignored, and for both Qwens they mildly *attract* (b_rival < 0).
  - Symmetric in neutral framing for Llama-70B (b_rival/b_ally = 1.15) and Nemo.
  - Not significant for Llama-70B political (Holm p = 0.078).
  - Scope: this is valence asymmetry, not non-reciprocity (Amendment 3c).
- **H2 (label fields): supported.**
  - The letter field is significant in 10 of 12 cells; the content field h_C in all 6 content cells.
  - Content direction varies by model:
    - Llama-70B prefers *oppose* and *keep*.
    - Qwen and Nemo prefer *adopt* in workplace.
  - The listed-first field h_O is mostly negative (−0.2 to −0.66: the second-listed letter is
    favoured). Option order must therefore be randomised per call in Stage 3 (Amendment 3d).
- **H3 (non-additive responses): supported in 2 of 12 cells (Llama-70B political and workplace, pairwise
  block).**
  - Held-out cross-entropy improves by 21% and 13%; ≥ 10% of cells move by ≥ 0.05.
  - The pairwise block is significant in 11 of 12 cells, but the other 9 fail the pre-registered minimum
    effect.
  - The cubic block is supported nowhere.
  - The Llama-70B effect is dominated by s₀·M_rival (rivals who agree with you push you toward the default
    content). Its sign holds under both letter mappings, so it is a content × social interaction, not a
    letter artefact.
- **H4 (β rises with model size): not supported. Only 1 of the 4 families was testable.**
  - Qwen 9B → 122B: β rises in all three framings (Holm p = 0.003).
  - Llama is excluded (8B fails comprehension). Gemma and Mistral are excluded (noise gate).
  - Descriptively, β is higher for the larger model in every family.
  - By Amendment 3c, β is a sampling knob and is confounded with comprehension.
- **H5 (sum vs average, α): departs from summation in 7 of 12 cells.**
  - Neutral framing: α ≈ 1 (averaging) for Llama-70B (1.42, CI 1.13–1.96), Nemo (0.99) and Qwen-9B (1.05).
  - Qwen-122B sums in every framing (α ≈ 0).
  - α is not interpretable for the two Llama-70B content cells, where H3 is positive.

## Secondary and exploratory findings

- **Inertia appears once positions have content.**
  - γ differs between political and neutral framing for all 15 endpoints.
  - In neutral, γ is about 0 or negative (agents lean toward switching), except Qwen-122B (γ = 2.2).
  - In political and workplace, γ = 1.1–7 (agents hold their position). In neighbour-equivalents, γ/β
    is often 5–25.
- **Order (primacy) effects are large and not exchangeable.**
  - Llama-70B, neutral framing: the logit SD across neighbour permutations of the same cell is 2.95 nats.
  - Mean P(choose P) is 0.76 when the first-listed contact pulls toward P and 0.25 when it pulls away.
  - This is voter-model-like: it copies one (random) neighbour, consistent with α ≈ 1.
  - The order-averaged response function is the estimand that matters for random-order networks
    (Amendment 1.2), but a voter-vs-logit comparison is worth adding as an exploratory analysis.
- **Wording** ("Having considered your contacts, which position do you choose?") shifts β, b_ally or γ
  modestly in about half the endpoints (largest: Mistral Large γ −1.04, Qwen-122B γ −0.81). α is unchanged
  in all 11.
- **Hiding the agent's own position** changes b_rival in 10 of 12 endpoints. Llama-70B's rival weight
  rises from 0.69 to 1.30.
- **Provider robustness.** Fits are robust across providers even where per-call jitter fails the gate.
  - Llama-70B, CoreWeave vs Parasail: b_ally 0.605 vs 0.607, γ −0.252 vs −0.256.
  - Gemma-26B, Parasail vs CoreWeave: b_ally 1.56 vs 1.59, γ −0.53 vs −0.60.
- **Bridge.** GPT-4o-mini sampled (30 per prompt) vs logprobs on the identical k ≤ 4 battery: every
  coefficient agrees (all |z| < 0.72; paired-bootstrap CIs include 0, except political β at the edge).
  This validates the sampling protocol. The leak check passed: sampled frequencies deviate from the
  logprob P by more than binomial noise (MSE 0.0019 vs 0.0008), as per-call jitter predicts.
- **Spot checks** (sampled; neutral and political, k ≤ 4):
  - **GPT-6-luna** is the most Ising-like: β 0.46, rivals oppose (b_rival/b_ally 0.65), small inertia
    (γ 0.47), additive (no H3), α ≈ 0.95.
  - **Gemini 3.1 Flash-Lite** follows allies strongly, ignores rivals, and switches away from its own
    position when only rivals are present.
  - **Claude Haiku 5.5** follows allies, rivals mildly attract in neutral, and it has strong inertia.
  - Political fits for Haiku and GPT-4o-mini sampled are near-separable (most prompts come out 30/30), so
    their CIs are wide.
- **Temperature (Amendment 3b): pre-registered verification FAILED.**
  - The χ² rejects at both temperatures. The slopes match 1/T (CIs contain 2.0 and 0.67).
  - The exploratory diagnostic attributes most, but not all, of the misfit to reference-logit jitter
    (0.25 nats).
  - No sampled temperature sweep has been run (owner decision).

## How LLM agents differ from Ising spins (input to Stage 2)

1. **Field from neighbours.** Most models give ally ≫ rival weight. Rivals are often ignored or mildly
   attractive, rarely repulsive. So a "rival" tie behaves closer to no tie than to J < 0.
2. **Inertia.** It is strong and framing-dependent: a self-coupling that the heat-bath has no term for.
3. **Degree scaling.** Several models average their neighbours (α ≈ 1), so their coupling is ∝ 1/k. This
   induces non-reciprocity on heterogeneous-degree graphs (PREREG §8c).
4. **External fields.** Letter, order and content fields are present. Order must be randomised in live
   runs; content fields are real.
5. **Additivity.** Mostly additive; the exception is Llama-70B in content framings.

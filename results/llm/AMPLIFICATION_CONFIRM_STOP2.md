# Confirmatory amplification phase: stop point 2 (Steps 1–3)

Plan: `llm/AMPLIFICATION_CONFIRM_PLAN.md` (76aad78), amendments 1 (32daeac) and 2 (guardrail re-pin). Step 4 has
**not** been run.

**Spend:**
- API: $3.33, about 137,000 calls.
- GPU: $0.91 over two A100 pods. The first stopped at the fingerprint gate.

**Framing for everything below:**
- The T axis is **decision noise**, not a realistic sampling temperature. Gemma's invalid-reply rate is 15% at T = 2.
- Results are for random-regular d = 4 graphs only.

## Model selection (amendment 1)

- **Selected logprob model: `openai/gpt-4o-mini`** (rank 20 of 28, pinned `openai`).
  - Repeat SD 0.296 (gate 0.3), leak 0.
  - κ = 0.89 [0.48, 1.28] (temperature honoured).
  - Comprehension 1.00 / 1.00 / 0.99.
- **Why the 19 higher-ranked candidates failed:**
  - Repeat SD above 0.3: DeepSeek v4.x on fp8 providers (0.35–0.98), Llama-4-Maverick, gpt-4o variants.
  - Errors: provider 400 errors (DeepSeek v3.2, R1); gpt-oss makes reasoning mandatory.
  - Temperature not honoured: Llama-3.2-3B (κ −0.16).
  - Comprehension failed: Llama-3.1-8B (ally 0.78, rival 0.72).
  - The first-party `deepseek` provider is blocked by the account's data-policy guardrail. Amendment 2 re-pinned it.
- **Caveat:** rank 19, `gpt-4o-mini-2024-07-18`, is the same model on the same provider. It **failed** the
  temperature probe (κ 0.12 [−0.22, 0.43]). The probe's 5 × 40-sample κ test is noisy, so the selection is
  borderline. The selected model is also the pre-stated fallback.

## Step 1: is the 1/T temperature scaling valid?

| model | role | T = 0.5 slope (1/T = 2) | χ² p | T = 1.5 slope (1/T = 0.67) | χ² p | pre-registered verdict | χ² vs reference noise (diagnostic) |
|---|---|---|---|---|---|---|---|
| Qwen-122B (P1) | test | 2.34 [1.89, 2.94] | 0.0005 | 0.86 [0.68, 1.08] | 0.08 | **fail** | T = 0.5 above noise; T = 1.5 within |
| Nemo-12B (P4) | control | 1.79 [1.37, 2.31] | 0.002 | 0.54 [0.22, 0.86] | 0.30 | **fail** (T = 0.5) | T = 0.5 above noise; T = 1.5 within |
| GPT-4o-mini | confirmatory | 1.51 [1.16, 1.94] | 1e−25 | 0.60 [0.43, 0.76] | 0.30 | **fail** (T = 0.5) | T = 0.5 above noise; T = 1.5 within |
| Gemini 3.1 Flash-Lite | confirmatory | sampled κ check: κ = 1.13 [0.96, 1.31], GOF p 0.20 | | | | **pass** | — |
| Gemma-31B (local) | confirmatory | holds by construction (softmax(logits/T), no top-p/top-k) | | | | n/a | — |
| GPT-6-luna | exploratory | temperature **refused** (404 under require_parameters) | | | | **unverifiable** | — |

**Reading** (post-hoc, labelled):
- The 1/T rule fits within noise on the **noise side (T = 1.5)** for every logprob model.
- It misfits on the **sharpening side (T = 0.5)**, beyond reference noise.
- Every located T_c is above 1 except luna's (0.49).

### Invalid replies per temperature

| model | T = 0.5 | T = 1.0 | T = 1.5 | T = 2.0 |
|---|---|---|---|---|
| Qwen-122B | 0% | 0% | 0% | — |
| Nemo-12B | 0% | 0% | 9.2% | — |
| GPT-4o-mini | 0% | 0% | 0% | — |
| Gemini | 0% | 0% | 0% | — |
| Gemma-31B | — | 0% | 0.15% | 15.3% |
| GPT-6-luna | — | 0% | — | — |

## Step 2: K_avg batteries and comprehension

- **Prompt set:** 768 prompts per model (70 configurations, ≤ 8 contact orders × 2 option orders). Both letter
  mappings are covered by neutral relabelling, asserted in code.
- **Comprehension** (pass ≥ 90% per type): all pass.

| model | own | ally majority | rival majority |
|---|---|---|---|
| Gemma | 1.00 | 1.00 | 1.00 |
| Gemini | 1.00 | 1.00 | 0.998 |
| GPT-4o-mini | 1.00 | 1.00 | 0.99 |
| GPT-6-luna | 1.00 | 1.00 | 0.997 |

- **Gemma-31B (local, batch size 1, deterministic):**
  - fingerprint vs Amendment 4: 0.234 (PREREG gate form; owner decision after the first pod stopped at 0.335 in the
    coded form);
  - **replicates bit-identical**;
  - 210/768 prompts censored by the top-5 rule. A sensitivity run with exact full-vocabulary P(A) gives the same T_c
    result.
- **GPT-4o-mini:** 768 × 2 repeats, 0 invalid.
- **Gemini and GPT-6-luna:** 768 × 50 samples each, 0 invalid.

## Step 3: T_c per model

Fresh graphs and seeds.

| model | result | T_c [95% CI] | U(100)–U(400) | scaled m² | Step 4 W, R |
|---|---|---|---|---|---|
| **P1 Qwen-122B (fresh)** | transition | **2.01 [1.96, 2.07]** (Stage B: 1.93) | 1.92 | 1.99 | 500, 400 |
| **Gemini 3.1 Flash-Lite** | transition | **1.83 [1.81, 1.86]** | 1.84 | 1.86 | 500, 400 |
| **GPT-4o-mini** | **no transition in the swept range** | — | — | — | — |
| **Gemma-31B** | **no transition in the swept range** | — | — | — | — |
| GPT-6-luna (exploratory) | transition | 0.49 [0.47, 0.57]; across 20 Beta kernel draws 0.45 [0.35, 0.57] | 0.44 | 0.46 | 1000, 200 |

**Diagnostics** (post-hoc, labelled; they do not change the pre-registered results):
- **GPT-4o-mini:** never ordered. The highest U(1600) is 0.47, at T = 0.2, and |m| ≤ 0.22. Hot and cold starts agree
  everywhere. U grows with N at T ≲ 0.45, a weak low-T ordering tendency that falls below the rule's ordered
  threshold (U ≥ 0.5).
- **Gemma-31B:** frustrated and glassy.
  - At every T ≤ 3.16, hot and cold starts disagree, so the rule marks the region as frozen. Cold starts decay from
    all-up to |m| ≈ 0.44; hot starts stay near 0.05.
  - A partially ordered window around T ≈ 2.5–3.2 reaches U(1600) = 0.50 at T = 3.16, with a hot/cold gap of 0.04.
  - Consistent with Gemma's strong rival response (b_r/b_a 0.71 at ρ = 0.25).
- **GPT-6-luna:** T_c lies on the sharpening side (T < 1), where its temperature axis is unverified and 1/T misfits
  for the other models.

**Step 4 as planned would therefore run on** P1 (fresh) and Gemini (confirmatory), plus GPT-6-luna (exploratory). It
would not run on GPT-4o-mini or Gemma, whose result is "no transition in range".

Figure: `results/llm/confirm/fig_binder_step3.png`.

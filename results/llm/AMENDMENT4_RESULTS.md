# Amendment 4 results: self-hosted bf16 Gemma-4

Pre-registration: `llm/PREREG.md`, Amendment 4 (committed before any GPU run). Execution record: `llm/EXECUTION_LOG.md`
entries 23–34. Claims concern these two checkpoints served by HF `transformers` in bf16 on one A100, nothing wider.

## Data

- Endpoints `gemma-26b-local` (`google/gemma-4-26b-a4b-it`, MoE) and `gemma-31b-local` (`google/gemma-4-31b-it`,
  dense). The identical Stage 1 battery (instruction v2, 5 arms + order subset + comprehension), official chat
  template, one forward pass per prompt, two replicates with different batch composition. Run `e3zan9ygnmqe45`
  (≈ 40 min, ≈ $1.16); every output uploaded and verified as it was produced.
- **Deviations from the amendment text** (both decided before the run, entry 23): batches are padding-free (prompts
  grouped by exact token length) instead of left-padded, because bf16 padding changed logprobs by up to 0.84 nats on
  the real models; activations are from the first replicate.
- Validity: local and provider P(A) on the same 6,214 prompts correlate at 0.987–0.993 (logit scale); comprehension
  100% for both local models.

## 1. Comprehension and noise gate (confirmatory)

| endpoint | comprehension | pooled repeat SD | gate (≤ 0.30) |
|---|---|---|---|
| gemma-31b-local | 100% (passes) | **0.114** | **pass** |
| gemma-26b-local | 100% (passes) | **0.592** | **fail** |

The 26B MoE's bf16 results move with batch composition (per arm 0.56–0.65 nats; the dense 31B 0.08–0.15). Local
self-hosting removed the provider jitter for the dense model but not for the MoE.

## 2. H1–H5 with the extended confirmatory set (confirmatory)

The confirmatory set grows from 12 to **15 cells** (adds `gemma-31b-local` × neutral, political, workplace). Holm
families recomputed over all 15. Outputs: `results/llm/stage1_*_amendment4.*`. The original 12-cell results stand
as reported in `STAGE1_RESULTS.md`.

`gemma-31b-local`:

| cell | b_a | b_r [95% CI] | H1 (Holm) | H3 pairwise | H3 cubic | α [95% CI] |
|---|---|---|---|---|---|---|
| neutral | 1.30 | 0.93 [0.79, 1.11] | supported (0.015) | no (0.84) | **supported (0.015)** | 1.05 [0.74, 1.51] |
| political | 2.41 | −0.58 [−0.91, −0.34] | supported (0.015) | no (0.27) | **supported (0.015)** | 0.13 [−0.06, 0.33] |
| workplace | 1.54 | 0.24 [0.14, 0.36] | supported (0.015) | no (0.27) | no (0.36) | 0.67 [0.30, 1.10] |

- **H1** (rivals weighted less than allies): 12 of 15 cells (original cells unchanged).
- **H3 cubic**: the two Gemma-31B positives that the gate-relaxed sensitivity analysis found on the jittery provider
  endpoint (EXECUTION_LOG entry 6) **replicate** on the clean local endpoint (neutral, political). Pairwise: no.
  No original H3 verdict changes.
- **H2**: letter field significant in all 3 local cells; content field not significant (political Holm 0.85,
  workplace 0.25). Re-Holm over the extended content family moves **Qwen-122B political** from 0.034 to 0.102
  (no longer significant); every other original H2 verdict is unchanged.
- **H5**: in the neutral and political cells H3 cubic is supported, so α is not interpretable there; workplace α =
  0.67 [0.30, 1.10] departs from summation.

## 3. H4 in the Gemma family (confirmatory)

**Not testable**: the 26B local endpoint fails the noise gate, so the pre-registered local pair 26B → 31B is
excluded. The overall H4 verdict is unchanged: **not supported** (1 of 1 testable family significant in neutral;
≥ 3 of 4 required).

## 4. Local vs provider (confirmatory)

Paired bootstrap (B = 1,000; both endpoints refitted on the same resampled cells), 29 coefficients per pair (4 arms),
attenuation slope through the origin over 21 entries (β, α excluded). `llm/compare_local.py`;
`results/llm/amendment4/local_vs_provider_*`.

| pair | coefficients agreeing | slope [95% CI] | κ predicted from jitter | pre-registered verdict |
|---|---|---|---|---|
| Io Net 31B vs local 31B | 22 / 29 (76%) | 0.99 [0.94, 1.03] | 0.98 | jitter biases estimates |
| Parasail 26B vs local 26B | 22 / 29 (76%) | 0.97 [0.91, 1.02] | 0.92 | jitter biases estimates |
| CoreWeave 26B vs local 26B | 22 / 29 (76%) | 0.98 [0.94, 1.02] | 0.98 | jitter biases estimates |

- Every verdict is driven by the agreement criterion; every slope CI contains both 1 and κ (no detectable overall
  attenuation).
- **Operating characteristic of the verdict rule** (measured, not pre-registered; `compare_local_null.py`): with
  identical true coefficients (20 synthetic replications, same row layout, B = 100) the rule returns "jitter
  averages out" in 80% of replications; mean agreement 94%, minimum 76%. The observed 76% in all three pairs sits at
  the bottom of that null distribution, so the disagreements are probably real, but small: the larger ones are the
  letter field h_L and order field h_O in political framing (e.g. Parasail h_L 0.32 vs local 0.74; CoreWeave h_O
  0.00 vs −0.29) and inertia γ; β agrees in 8 of 12 arm comparisons.
- The 26B comparisons set two noisy measurements against each other (local 26B fails the gate). The 31B comparison
  is the clean one.
- Confound (pre-registered note): providers also differ in quantisation, chat-template handling and serving stack;
  a disagreement shows provider estimates differ from bf16 HF inference, not that jitter caused it.

## 5. Exploratory annex: probes (labelled exploratory; cannot alter H1–H5)

L2 logistic probes on last-token residual activations, 10-fold CV grouped by multiset, AUC from the decision
function (`llm/probes.py`; `results/llm/amendment4/probes_*`). Targets in letter space.

| | 31B best AUC | 26B best AUC | letter-count control | shuffled (max) |
|---|---|---|---|---|
| ally-majority side | 1.000 | 0.994 | 0.960 | ≤ 0.55 |
| rival-majority side | 0.986 (layer 50) | 0.976 (layer 25) | 0.955 | ≤ 0.55 |
| own position | 1.000 | 1.000 | 1.000 | ≤ 0.53 |
| transfer neutral → political, rival | 0.75 | 0.84 | | |

- **Represented but not used** (pre-stated descriptive criterion): rival side decodable (AUC ≥ 0.9 and above the
  letter-count control) in both models.
  - 26B: fitted |b_r| / b_a = 0.07 → **pattern present** (rival majority is represented, barely moves the answer).
    Caveat: the 26B fit comes from the gate-failing endpoint.
  - 31B: |b_r| / b_a = 0.71 → pattern **absent** (represented and used).
- Readout alignment: cosine between the last-layer rival-majority probe direction and W_U[A] − W_U[B] is −0.013
  (31B) and ≈ 0.000 (26B); the rival-majority information is not on the answer readout at the last layer.
- The letter-count control is strong by construction (≈ 0.96 in letter space); the margins above it are small
  (0.02–0.04 for the rival side).
- SAE lookup: skipped (no SAE release for Gemma-4 in this setup).

## Process notes

- A container restart killed the first analysis run; the chain was rebuilt to save every unit (pair × arm, null
  replication, probe layer) as it finished and resumed cleanly after a second restart.
- A probe scoring bug (AUC from saturated probabilities, forcing transfer AUC to exactly 0.5) was found in the
  output and fixed before any probe result was kept.
- The Stage 1 comprehension summary had been overwritten by a synthetic test and committed; it was restored
  byte-identical before the analysis (entry 33).

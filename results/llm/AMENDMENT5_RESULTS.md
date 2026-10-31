# Amendment 5 results: rescue rerun of local Gemma-26B at batch size 1 (and follow-ups)

Pre-registration: `llm/PREREG.md`, Amendment 5 (committed 7705d12 before any run). Execution record:
`llm/EXECUTION_LOG.md` entries 35–36. **Every result using `gemma-26b-local-bs1` is a rescue analysis**: the direction
of H4 in the Gemma family was already known from the gate-relaxed sensitivity analysis (entry 6).

## Run

- `gemma-26b-local-bs1`: `google/gemma-4-26b-a4b-it`, bf16, batch size 1 for both replicates, deterministic kernels
  (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off). Pod `4p4a463q78clp0`,
  ≈ 2 h 28 min, ≈ $4.40; all 43 files uploaded and verified; replicate 0 was checkpointed and uploaded before replicate 1.
- **Deviation:** Amendment 5's caps (2.5 h / $6) were raised to a 3.3 h job deadline and 3.5 h / $6.50 watchdog after
  the first pod (`5407atbbj7bwit`, stopped after 8 min, ≈ $0.25) measured 1.86 forward passes/s at batch size 1,
  i.e. a run longer than its 2.2 h deadline. My 40–70 min estimate was wrong. GPU stage total ≈ $11.90 of $15.
- The two replicates are identical (max |ΔP(A)| = 0.0); no missing values; no prompt with leak > 5%; comprehension
  100%.

## Confirmatory (rescue)

- **Gate:** pooled repeat SD 0.000 → passes. At batch size 1 this measures run-to-run determinism only, not
  batch-composition numerics.
- **Extended confirmatory set: 18 cells** (12 original + 3 local 31B + 3 local 26B-bs1); Holm recomputed over 18.
  **No Amendment 4 verdict changes** (H1, H2, H3 per cell identical).
  - H1: 15 of 18 (all 3 new 26B cells).
  - H3: pairwise 3 of 18 (adds 26B workplace); cubic 4 of 18 (adds 26B neutral and workplace).
  - H2 (26B-bs1): letter and content fields significant in political and workplace (Holm 0.012–0.018); no
    letter field in neutral (0.66).
- **H4 in the Gemma family (26B-bs1 → 31B local):** β rises in every framing (one-sided bootstrap, Holm within
  family 0.003 each):

  | framing | β 26B | β 31B | 95% CI of difference |
  |---|---|---|---|
  | neutral | 0.75 | 1.11 | [0.17, 0.56] |
  | political | 0.33 | 0.92 | [0.40, 0.85] |
  | workplace | 0.30 | 0.89 | [0.44, 0.75] |

- **Overall H4: not supported** (2 of 2 testable families significant in neutral; the pre-registered rule needs
  ≥ 3 of 4, and Llama and Mistral remain excluded).

## Secondary: providers vs the clean 26B (rescue)

| pair | agreement | null percentile (B = 1000) | slope [95% CI] | κ | verdict |
|---|---|---|---|---|---|
| Parasail 26B vs local 26B-bs1 | 86% (25/29) | ≈ 5th | 0.97 [0.91, 1.02] | 0.92 | jitter biases estimates |
| CoreWeave 26B vs local 26B-bs1 | 83% (24/29) | ≈ 1st | 0.97 [0.93, 1.00] | 0.98 | jitter biases estimates |

Against the batched 26B both pairs agreed at 76%; a clean reference raises agreement to 83–86%: part of the earlier
disagreement was the noisy reference. Social coefficients (b_a, b_r, β) agree in 10 of 12 arm comparisons for both
providers; format fields 8–9 of 10.

## The batch effect (kept as data; exploratory)

Batched 26B (Amendment 4) vs batch-size-1 26B, same 6,214 prompts, logit scale (`results/llm/amendment5/batch_effect.json`):

- RMS deviation of a batched replicate from the batch-size-1 value: **0.86 nats** (unbiased: mean +0.015;
  correlation of the batched mean with the batch-size-1 value 0.990; 2.5% of prompts flip majority answer).
- RMS deviation between the two batched replicates (/√2): 0.59, the value the gate saw. The replicates share part of
  their deviation, so the replicate difference understates the batch effect.
- Roughly uniform across arms (0.83–0.89); largest at k = 1 and k = 6.
- For Stage 2, the "batched serving" jitter source should use σ ≈ 0.86 (reported next to the gate value 0.59).

## Local vs provider: recalibrated null and framing (Amendment 4 item 4, owner request)

- **Null at B = 1000, 100 replications** (`results/llm/amendment4/compare_rule_null_B1000.json`): with identical true
  coefficients the pre-registered rule returns "jitter averages out" in 77% of replications; agreement mean 95%,
  1st percentile 83%, minimum 83%; **0 of 100 replications at or below 76%.**
- **The only clean comparison is Io Net 31B vs local 31B** (the Amendment 4 26B pairs used a gate-failed reference and
  share data). Its 76% agreement lies below every null replication (p < 0.01): the differences are real.
- **What differs** (31B pair): format fields agree 7/10 (h_O in neutral and political, h_L in workplace differ);
  social coefficients agree 9/12, and all three disagreements are in **workplace**, where b_a, b_r and β are 7–9%
  lower at the provider; slope 0.99 [0.94, 1.03] (no overall attenuation).
- **Headline (qualified):** provider choice shifts prompt-format biases (h_L, h_O); the social coefficients
  (β, b_ally) agree in two of three framings and differ by 7–9% in the third.

## Exploratory (not for any headline)

**Gemma-31B cubic H3** (`results/llm/amendment5/cubic_gemma-31b-local.json`): political is dominated by T_aaa = −0.48
[−0.84, −0.37] (alone: 4.7% held-out gain) — the ally response saturates; neutral has T_rrr −0.36, T_aar −0.36, T_arr
+0.35 (all CIs exclude 0; full cubic 3.0% held-out gain). A majority-rule response (sign of M_a, M_r) is far worse than
linear (held-out CE +21% neutral, +55% political) and adds ≤ 0.5% on top of it: mild saturation, not majority rule.

**Probe follow-ups** (`results/llm/amendment5/probes_followup_*`):
- a) AUC(probe) − AUC(letter-count) with 95% group-bootstrap CI. 31B: +0.03 in neutral at layers 30–60 (CIs exclude
  0); +0.01–0.02 in political and workplace (CIs exclude 0 at some layers). 26B: never significantly above the
  control (neutral CIs include 0; political and workplace below it at middle layers).
- b) Deconfounded test (rival-majority side ≠ overall A-majority; 160 configurations per arm; an overall-A-count score
  gives AUC 0.03 there). 31B: layer 10 tracks overall A-count (AUC 0.11–0.20), layer 20 is a transition (0.55–0.60),
  layers 30–60 carry rival-specific information (0.75–0.96, rising with depth). 26B: neutral 0.85–0.92 at layers
  20–30; political and workplace 0.6–0.8.
- c) Readout baselines at the last layer (cosine with W_U[A] − W_U[B]). 31B: rival −0.013, ally −0.009, own −0.011, all
  inside the shuffled-label and isotropic bands (SD ≈ 0.013): the test cannot separate used from unused directions
  there, so the earlier "rival direction orthogonal to the readout" statement is uninformative and withdrawn. 26B:
  own +0.078 (100th percentile of shuffled-label directions), ally +0.039 (99th), rival −0.000 (1st): the directions
  that drive the answer lean toward the readout and the rival direction does not.
- Revised reading of Amendment 4's "represented but not used": for 26B the rival side is not decodable above the
  letter-count control (CI includes 0), though a deconfounded rival signal exists in neutral framing and the rival
  direction is the least aligned with the readout; for 31B the rival side is represented (above control in neutral,
  deconfounded AUC up to 0.96) and used (|b_r|/b_a 0.71).

## Stage 3 status

Complete since 2026-10-09 (EXECUTION_LOG entry 26): 140,000 live updates, $4.95; pre-stated verdict **surrogate not
supported** (both primary measures pass at 0 of P1–P4; P5 and both no-contacts arms also miss). Nothing further is
pending unless a follow-up analysis is requested.

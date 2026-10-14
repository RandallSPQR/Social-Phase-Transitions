# Confirmatory amplification test with a corrected rule, plus two new models: plan (committed before any run or API call)

Owner spec 2026-10-14. This plan fixes the rules, predictions, model-selection record and costs before anything is
run. **Stop point 1:** after this commit, no API or GPU spend until the owner says OK.

## 0. Context and what is not reused

- **The Stage B pre-registered verdict stays "refuted"** (`results/llm/AMPLIFICATION_RESULTS.md`, commit 8da7862).
  - Cause: the growth-ratio comparison against the controls (H2) is ill-defined when a control's amplification is
    ≈ 0.
  - P2's majority-shift amplification at T_c was 0.002 (N = 100) and 0.010 (N = 1,600) |m| per nat. Its growth ratio
    had CI 0.9–68, so the P1/P2 ratio (CI 0.05–3.8) could not decide.
  - Every write-up of this phase restates that verdict and cause.
- **This phase tests a corrected rule (§5) on fresh simulations.**
  - New seeds: salt `confirm`, used everywhere.
  - New graphs: seeds `confirm-graph|point|N|g`, five per (point, N), **including N = 100**. The Stage 3 graphs are
    not used.
  - Fingerprints are committed and asserted before any perturbation run (networkx 3.7, as CLAUDE.md requires).
- **No Stage B run, T_c, grid or growth number feeds any confirmatory number.**
  - P1 is re-located from scratch.
  - Stage B values appear only in "compare with Stage B" lines, labelled as such.
- **Kernels reused as data, not as results:**
  - P1, P2, P4: the committed K_avg tables (`k_avg_calls.csv`).
  - P1's fixed-order battery: the Stage 1 battery.
  - The temperature transform is the one used in Stage B: per-prompt letter logit ÷ T, then marginalised
    (`amplification_curve.kavg_T`).

## 1. Model selection (criteria fixed here, before any run)

The criteria are the owner's:
1. neutral framing with no strong content field;
2. an order–disorder transition expected from Stage 2;
3. passes the 0.3-nat noise gate, or is sampled with ≥ 30 samples per prompt;
4. 1/T verifiable;
5. lineage different from Qwen.

| criterion | Gemma-4 31B (local bf16) | GPT-6-luna (OpenRouter, pinned `openai`) |
|---|---|---|
| 1. neutral, no content field | yes: neutral_own has no h_C. Letter and option fields (h_L 0.45, h_O 0.09) cancel under the mapping/order average | yes: no h_C. h_O 0.64 cancels under the option-order average |
| 2. transition expected (Stage 2, random-regular, fitted rule, T = 1) | **likely**: ordered at ρ = 0 (x ≈ 0, \|m\| ≈ 1.0); intermediate at ρ = 0.25 (x = −0.58 [−0.81, −0.36]). Fits come from the Io Net endpoint; the local fit is near-identical (b_a 1.30, b_r 0.93) | **uncertain**: disordered at every ρ at T = 1 (x ≈ −1, b_a 0.55, b_r 0.36, β 0.46). An ordering transition can only appear below T = 1, where the frozen-dynamics region (Stage B) may intervene |
| 3. noise gate / sampling | passed in Amendment 4 (0.114 nats, batched); batch size 1 with deterministic kernels here, so replicates should be identical | sampled, **50 per prompt** (≥ 30 required) |
| 4. 1/T verifiable | by construction for local sampling with no top-p/top-k (§2.4) | **at risk**: the OpenRouter catalogue lists no `temperature` (or `logprobs`) parameter for this model, and sample-mode requests do not set `require_parameters`, so a temperature may be silently ignored. Tested first in Step 1 (§2.3) |
| 5. lineage ≠ Qwen | Google | OpenAI |

**Rules (owner's):**
- Qwen-9B is the fallback only. It replaces a chosen model only if that model **cannot be run at all** (endpoint
  gone, weights unavailable, comprehension failed), never because of its results.
- No substitution after seeing results.
- If a chosen model shows no transition in the swept range, that is its result.
- If GPT-6-luna fails criterion 4 (temperature not accepted, or the sampled check rejects 1/T):
  - it stays in the study;
  - its T axis is labelled **"defined decision-noise rescaling, unverified for this model"** in every table and
    figure;
  - its verdict is reported with that label.
  - The owner is told at stop point 2 before Step 4 runs.

## 2. Step 1: verify the 1/T rule

**2.1 Qwen-122B (P1) and Nemo-12B (P4).**
- Exactly the check that passed for Qwen-9B: `temp_check.py verify <key>`.
  - 20 prompts from the endpoint's Stage 1 neutral_own battery (perm 0, rep 0) with P₁(A) in [0.1, 0.9], taken in
    SHA-256 order.
  - 50 samples each at T = 0.5 and T = 1.5.
- Pass at each T requires both:
  - the logistic slope of the samples on the logprob logit has a 95% CI containing 1/T;
  - Pearson χ² over the 20 prompts (df 20) has p > 0.01.
- Also reported:
  - the reference-noise-adjusted χ² (`temp_diag.py` method), labelled as a diagnostic;
  - invalid replies per T.

**2.2 GPT-6-luna, logprob probe.** Two calls in lp mode with `require_parameters: true`. If logprobs come back, run
§2.1 unchanged. The catalogue predicts they will not.

**2.3 GPT-6-luna, temperature probe and sampled check.**
- **Probe:** 2 calls with `temperature` and `provider.require_parameters: true`.
  - If OpenRouter refuses the request (no endpoint supports the parameter), temperature is **not accepted**:
    criterion 4 fails and §1's rule applies.
  - Otherwise, run the sampled check.
- **Sampled check:**
  - 20 prompts from luna's Stage 1 neutral_own battery with sampled P(A) in [0.15, 0.85], in SHA-256 order.
  - **100 samples at T = 1, 50 at T = 0.5 and 50 at T = 1.5.**
  - Model: P_T(A) = σ(ℓ_i · T^(−κ)), with a free ℓ_i per prompt and one exponent κ, fitted by maximum likelihood.
  - **Pass iff:**
    - the profile-likelihood 95% CI of κ contains 1;
    - the likelihood-ratio goodness-of-fit against the saturated per-(prompt, T) model has p > 0.01 (df 39);
    - κ's CI excludes 0, so the temperature visibly acts.
- Invalid replies per T.

**2.4 Gemma-31B (local).**
- Sampling is softmax(logits / T) over the full vocabulary, with no top-p, top-k or repetition penalty. The letter
  probability renormalised over A/B is therefore σ(ℓ/T) **exactly**, and 1/T holds by construction.
- The plan states this and does not test it.
- **Garbage / invalid-token rate:** sampled replies (≤ 8 new tokens, the Stage 1 validity rule: a single A/B letter)
  at **T = 1.0, 1.5 and 2.0**. Prompts are the first 100 K_avg prompts in SHA-256 order, 20 samples each.
- Reported per T, with the exact full-vocabulary leak (mass on non-A/B first tokens) per T.

**2.5** A table of invalid-reply rates per model and temperature, from the above.

## 3. Step 2: order-averaged batteries (K_avg) for the new models

- **Configurations:** every k = 4 contact configuration on random-regular d = 4 graphs, i.e. multisets over
  {ally-P, ally-Q, rival-P, rival-Q}, × own position. That is 35 multisets × 2 = **70 configurations**.
  - Neutral framing, ρ = 0.25 as in P1, prompts rendered by `stage3.render_live` and checked by `check_prompt`.
- **Per configuration:**
  - ≥ 8 contact orders: all distinct orders if there are ≤ 8, else 8 sampled with seed
    `confirm-kavg-orders|model|c|s0`;
  - × 2 option orders;
  - × both letter mappings.
- **Letter mappings in neutral framing:** the mapping-1 prompt of (c, s0) is byte-identical to the mapping-0 prompt
  of the P↔Q-relabelled configuration (c′, −s0), which is in the set. So rendering every configuration at mapping 0
  covers both mappings of every configuration exactly. This is the same construction as `k_avg.py`, asserted in
  code by comparing the rendered strings.
- **Size:** 768 distinct prompts.
- **Gemma-31B (local):**
  - bf16, HF transformers 5.19.0 (pinned), batch size 1;
  - Amendment 5 deterministic settings: `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off,
    cuDNN deterministic, seeds 0;
  - the official chat template;
  - P(A) by the Stage 1 top-5 rule, with the exact full-vocabulary value as secondary.
  - **Two replicates** in different orders: **confirm they are bit-identical** (report max |ΔP|).
  - Environment fingerprint asserted before the run: for 20 Stage 1 prompts, P(A) at batch size 1 must agree with
    the Amendment 4 `gemma-31b-local` values within the 0.3-nat gate.
- **GPT-6-luna:** 50 samples per prompt (38,400 calls) at T = 1, reasoning off, pinned `openai`, no fallbacks.
  - Per-prompt P is the Jeffreys estimate (x + ½)/(n + 1).
  - Kernel uncertainty: 20 Beta(x + ½, n − x + ½) kernel draws, used for a T_c sensitivity check (§4).
- **Comprehension (both, before either kernel is used):** the Stage 1 comprehension battery.
  - 2,116 prompts covering own position, ally majority and rival majority.
  - Pass: ≥ 90% correct per question type.
  - **Gemma:** logprob rule (P(correct) > 0.5, leak ≤ 5%), batch size 1.
  - **GPT-6-luna:** sampled, 10 samples per prompt (21,160 calls); correct if ≥ 6 of 10 valid samples pick the
    correct letter.
  - **A model that fails comprehension cannot be run as specified.** It is reported, and Qwen-9B is the fallback
    under §1.
- **Kernel at T:** the same transform as P1, applied to the new per-prompt values (`amplification_curve.kavg_T`
  generalised to a calls file).

## 4. Step 3: locate T_c per model (P1 fresh, Gemma-31B, GPT-6-luna)

- **Coarse pass:**
  - 15 geometric points over T ∈ [0.2, 5];
  - hot (random 50/50) and cold (all +1) starts;
  - W = M = 1,000 sweeps, R = 100 per start, N = 100, 400, 1,600;
  - fresh graphs and seeds.
  - **Non-frozen T:** hot and cold mean |m| at N = 1,600 within 0.02.
  - **Edge interval:** among consecutive non-frozen coarse pairs, the one with the largest drop in U(1600) per log T.
- **No transition:** if no non-frozen coarse T has U(1600) ≥ 0.5 (never ordered), or no drop exists, the model has
  **no transition in the swept range**. That is its result, and Step 4 is not run for it.
- **Fine pass:**
  - 13 geometric points over [T_a/1.25, 1.25·T_b], where (T_a, T_b) is the edge interval;
  - same runs as the coarse pass.
  - **T_c** = the U(400)–U(1600) Binder crossing in the fine band; if there are several, the one nearest the steepest
    drop of U(1600).
  - 95% CI from 200 run-bootstraps.
  - Cross-checks: the U(100)–U(400) crossing and the scaled-m² (N^½ ⟨m²⟩) crossing.
  - If the fine band shows no U(400)–U(1600) crossing: no transition located.
- **GPT-6-luna:** T_c is also located for each of the 20 Beta kernel draws (fine pass only, R = 50). The spread is
  reported as kernel-sampling uncertainty.
- **Run length for Step 4:** the shortest W = M ∈ {200, 500, 1000} with hot/cold |m| gap ≤ 0.02 at N = 1,600 at
  every Step 4 band point. R = 400 if W ≤ 500, else R = 200.

**Stop point 2:** report the 1/T checks, invalid rates, comprehension, replicate identity and T_c per model, then
wait for the owner's OK before Step 4.

## 5. Step 4: corrected confirmatory test (pre-registered rule)

**For each test model separately:** P1 (fresh runs), Gemma-31B, GPT-6-luna. No pooling across models.

**Design:**
- **T band:** 13 geometric points over [T_c/1.3, 1.3·T_c], with T_c as the centre point (about a 4.5% step).
- **Sizes:** N = 1,600 at every band point; N = 100 at T_c.
- **Controls:** P2 and P4 at the test model's T_c, N = 1,600 and N = 100. They use their own committed K_avg and ρ,
  and fresh graphs.
- **Per-step error δ = 0.1 nats**, with these shapes:
  - stay-bias;
  - majority-shift;
  - random per-configuration noise (3 fresh fixed tables, seed `confirm-noise|k`);
  - for P1, also the **real fixed-order battery error shape** (K_fixed − K_avg at T, λ = 1).
- **Real shape for the new models:** their Stage 1 batteries exist (Gemma local; luna sampled), so their K_fixed −
  K_avg shapes are run and reported as **secondary**, outside the verdict.
- **Common random numbers within each arm.**
- **Amplification:** A = |Δ⟨|m|⟩| / ε, where ε is the visit-weighted mean |ΔL|. For noise, A is the mean over the 3
  tables. 95% percentile bootstrap over runs (1,000 resamples).

**Rule A (primary): absolute amplification beats each control.**
- For shape s, D_c = A_model(T_c, 1600) − A_c(T_c, 1600) for each control c ∈ {P2, P4}.
- Its CI comes from independent run-bootstraps of the two arms.
- **Pass for s iff, for both controls, D_c ≥ 0.10 |m| per nat AND the 95% CI of D_c excludes 0** (lower bound > 0).
- **Why 0.10:**
  - It is the same floor as rule B.
  - At δ = 0.1 it corresponds to a collective shift of 0.01 in |m| beyond the control.
  - It is well below the Stage B gaps (0.6–2.5) and above the Stage B control range at T_c (0.01–0.30, of which
    only P4 majority-shift exceeded 0.16).
  - It was chosen without the new data.

**Rule B (secondary): N-growth, tested only where amplification is non-negligible.**
- G = A(T_c, 1600) / A(T_c, 100).
- Tested only if **A ≥ 0.10 at both N**; otherwise "not tested (amplification below floor)".
- **Pass iff the 95% CI of G excludes 1 from above** (lower bound > 1).
- The control comparison of Stage B's H2 is dropped. Rule A replaces it.

**Rule C: peak location.** argmax_T A(T, 1600) over the band lies within **±10% of T_c**, i.e. in
[T_c/1.1, 1.1·T_c]. That covers the 5 central band points. Bootstrap share reported.

**Verdicts:**
- Per model and per shape: A pass/fail (decisive), B pass/fail/not tested, C pass/fail.
- Per model: "amplification at the tipping point exceeds the controls for k of n shapes". There is no further
  aggregation and no pooling across models.
- Persistence is reported with the same A/B/C rules as a secondary measure, with its own verdicts.

**Predictions (stated before running):**
- **P1:** A passes for all 4 shapes (Stage B absolute gaps were 0.7–2.5), B passes for the 3 synthetic shapes and
  the battery shape, and C passes.
- **Gemma-31B:** if a transition is located, A passes for ≥ 2 of 3 synthetic shapes. This is less certain: its
  couplings differ, and rivals are strong (b_r/b_a 0.71).
- **GPT-6-luna:** no prediction on whether a transition exists. If one is located, the same as Gemma.
- **Expected failure modes, stated in advance:**
  - random noise may beat the controls more strongly than the symmetric shapes, because it breaks the ±m symmetry;
  - majority-shift and stay-bias act through an effective-temperature shift, so their A tracks the slope of |m|(T)
    at T_c.

## 6. Step 5: the low-T freezing edge (P1)

- **Grid:** 7 geometric points over T ∈ [0.5, 0.9], hot starts.
- **Run lengths:** W = M = 500 (1×, as Stage B), 1,000 (2×) and 2,000 (4×).
- **Sizes and runs:** N = 100, 400, 1,600; shapes stay-bias, majority-shift and noise (3 tables) at δ = 0.1;
  R = 200; fresh seeds and graphs (salt `confirm-freeze`).
- **G_L(T)** = A(1600)/A(100) at length L, tested only where A(100) ≥ 0.10.
- **Rule:** call it a **dynamical transition** only if, at 4× length, all of the following hold:
  - for ≥ 2 of 3 shapes, some testable T has a G CI lower bound > 1;
  - max_T G at 4× is ≥ 0.5 × max_T G at 1× for those shapes.
- Otherwise report it as a **finite-time effect**.

## 7. Figures and framing

**Figures** (Tufte style, as Stage B):
1. Binder U(T) per model.
2. A vs distance from T_c (T/T_c), one line per N, controls ghosted, real eval errors marked.
3. Rule A differences with CIs per model and shape.
4. Step 5: G vs T per run length.

**Framing in every write-up:**
- The T axis is **decision noise**, not a realistic sampling temperature: invalid tokens rise above about T = 1.5,
  and for luna the axis may be unverified.
- Stage 3 ran at T = 1, deep in P1's ordered phase (T/T_c ≈ 0.5 in Stage B). There the real evaluation errors had
  small collective effects (+0.004 in |m|). The headline concerns what happens as a population approaches a tipping
  point.
- Results are for random-regular d = 4 graphs only.

## 8. Costs (estimates; caps are hard)

| item | calls / time | est. cost | cap |
|---|---|---|---|
| 1/T check, Qwen-122B (2,000 sampled + 40 lp at $3.7e-5) | 2,040 | $0.08 | — |
| 1/T check, Nemo-12B (at $2.5e-6) | 2,040 | $0.01 | — |
| GPT-6-luna: probes + sampled 1/T check (4,000 at $1.5e-5) | ~4,010 | $0.06 | — |
| GPT-6-luna K_avg (768 prompts × 50) | 38,400 | $0.57 | — |
| GPT-6-luna comprehension (2,116 × 10) | 21,160 | $0.32 | — |
| **API total** | ~67,600 | **≈ $1.05** | **$3.00** (budget stop in client) |
| Gemma-31B pod: A100 80GB at ~$1.79/h. Setup + 62 GB weights ≈ 20 min. K_avg 768 × 2 replicates + comprehension 2,116 + fingerprint 20 + garbage-rate generation (100 prompts × 20 samples × 3 T, ≤ 8 tokens) at batch size 1, ≈ 1.5–3 passes/s ≈ 30–50 min | ≈ 1–1.5 h | **$1.8–2.7** | **$5, 4 h hard stop** (watchdog + job deadline) |
| Simulation (Steps 3–5), this container | ≈ 3–4 CPU-h | $0 | — |

**Gemma pod process** (CLAUDE.md):
- a new `local_gemma_confirm.py` that takes an arbitrary prompt file and does batch-size-1 logprobs and short
  generation;
- run by `pod_run.sh` with `runpod_launch.py` (caps `--max-hours 4 --max-spend 5`);
- heartbeat every 90 s to HF;
- per-unit verified uploads (fingerprint, K_avg replicate 0, replicate 1, comprehension, garbage rates);
- teardown only after a verified final upload.
- Before paying: a local end-to-end test with a tiny model that exercises the heartbeat, per-unit uploads,
  verification and teardown.
- The failure record at each stage is written into the launch log first.
- The owner's other pods and the shared volume are not touched.

**Order after OK:**
1. API Steps 1–2: probes first, then checks, comprehension, then K_avg for luna.
2. Gemma pod.
3. Step 3 simulations.
4. Stop point 2.

## 9. Process and deviations

- Every background job goes through `llm/bg.sh` with PID files; no pattern kills. Saves are per unit and resumable.
- Every deviation is logged in `llm/EXECUTION_LOG.md` before the affected result is looked at.
- **Stop points:**
  1. after this commit (owner OK for spend);
  2. after Steps 1–3;
  3. after Steps 4–5: the full report, verdicts first, post-hoc readings separated.

# Stage 1 execution log (events after main data collection began)

These are not pre-registration amendments. They record how pre-registered rules were applied, plus any
judgment calls, in time order (2026-10-08).

1. **Temperature verification (Amendment 3b) FAILED as pre-registered.**
   - GPT-4o-mini, 20 prompts × 50 samples at T = 0.5 and T = 1.5.
   - Slopes 1.67 (95% CI 1.37–2.07; expected 2.0) and 0.63 (0.46–0.82; expected 0.67): both CIs contain 1/T.
   - Pearson χ²: p = 1.4e-12 (T = 0.5) and p = 0.008 (T = 1.5), so the test fails.
   - **Exploratory diagnostic** (`llm/temp_diag.py`): the reference logits themselves jitter by 0.25 nats
     between identical calls. Simulating the 1/T rule plus that noise gives 95th-percentile χ² of 57.2
     (T = 0.5) and 35.7 (T = 1.5). Observed: 63.3 and 36.1. So reference noise explains most, but not all,
     of the misfit.
   - Per the amendment, no sampled temperature sweep has been run; the owner decides.
   - The per-endpoint semantics rule labelled 6 of 11 endpoints "unclear". All of them use saturated test
     prompts (|logit| 8–15) where bf16 jitter exceeds the ±0.2 tolerance. None showed post-temperature
     (1/T) scaling.
2. **Noise gate (PREREG §4) applied at endpoint level.** Repeat SD is the RMS over all prompts of all arms.
   Results:
   - gemma-26b @ Parasail: **0.703, fail**
   - gemma-31b @ Io Net: **0.304, fail**
   - gpt4o-mini: 0.497, fail (bridge; descriptive anyway)
   - llama-70b @ Parasail: 0.404, fail (replication; descriptive anyway)
   - All others pass (0.026–0.137).
   - The first interim summary applied the gate per arm. That was a bug, fixed before any confirmatory
     output was produced.
3. **Backups for the failed Gemma primaries** were run as pre-listed (`llm/endpoints.py`).
   - **gemma-26b → CoreWeave bf16.** A 100-prompt pre-check gave 0.289, but the full battery gave
     **0.354: fail.** No further backup is listed, so gemma-26b is dropped from confirmatory analysis.
   - **gemma-31b → Novita bf16.** **77% of calls returned no logprobs** (2,676 of 3,498), even with
     `require_parameters: true`. Text answers were correct, and when logprobs were present they matched
     Io Net exactly (r = 1.00). This is not a usable logprob endpoint, and the missingness may depend on
     the prompt.
     - The run was **stopped** after the comprehension battery (its 22% "accuracy" reflects missing
       logprobs, not misreading).
     - gemma-31b is dropped from confirmatory analysis.
   - Consequence: the Gemma family is excluded from H4.
   - The Gemma fits are reported descriptively. The Parasail and CoreWeave fits of Gemma-26B agree closely
     (neutral b_ally 1.56 vs 1.59, γ −0.53 vs −0.60; political γ 5.25 vs 6.0). Llama-70B's two providers
     likewise agree (b_ally 0.605 vs 0.607, γ −0.252 vs −0.256). The gate measures per-call jitter that
     largely averages out in fits. **Whether to relax it is an owner decision; it has not been changed.**
4. **Comprehension (Amendment 3a).**
   - **llama-8b fails:** ally majority 0.78, rival majority 0.72, own position 0.99. It is labelled "does
     not parse task".
   - **mistral-small fails by the leak rule:** 1,116 of 2,116 prompts have leak > 5%. On the majority
     questions it starts to reason ("To…", "Let…", "Based…"). Mean renormalised P(correct) is 0.94 (ally)
     and 0.91 (rival), so it is not misreading. It is reported as "answers only after reasoning"; the
     pre-registered verdict (fail) stands.
   - qwen-122b: one comprehension call was lost to an Alibaba rate-limit error (2,115 of 2,116 scored).
5. **Spot checks were started before the last two workhorse runs finished**, to save wall-clock time.
   - Spend at launch: $5.18. The remaining workhorse runs (Mistral Large, Novita Gemma-31B) could cost at
     most about $2.40 at list price, so spend after the workhorses was guaranteed to be ≤ $12, the owner's
     condition.
   - Per-process caps held the worst case to about $11.50.

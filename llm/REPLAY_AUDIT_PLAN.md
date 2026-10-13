# Replicate replay audit and combined effect size: plan (committed before computing)

Owner request 2026-10-13. Free, no API calls. Code: `llm/replicate_replay_audit.py`. Outputs:
`results/llm/stage3_rep1/replay_audit/`. If every check passes, the amplification curve starts. If any check fails:
report and stop.

## Replay audit (all 70 replicate chains)

Each chain is replayed with the unchanged harness (`stage3.chain`, run_tag "rep1") on **its recorded graph**
(`results/llm/stage3_rep1/graphs/`, assignment `chain_graph.json`). The only call allowed is a cache-only lookup by
the exact request key (body + tag): a miss is a failure, and nothing is ever sent.

| check | pass criterion |
|---|---|
| R1 prompts | every prompt the replay renders equals a cached request exactly (request-key hit for 100% of calls, all attempts) |
| R2 choices | every choice is reproduced: logprob points from the cached logprobs plus the chain's seeded RNG; P5 from the cached reply. Replayed update log (agent, s0, new, attempts, mapping, order, letter) and saved states identical to the files, 0 mismatches |
| R3 counts | each chain has exactly 20 × 100 = 2,000 updates, (sweep, step) unique and complete, and 21 saved states |
| R4 restart boundaries | per chain, every resume point is listed: a cache-time gap that spans a recorded (re)start in `interruptions.jsonl`, with sweep/step and the hosts before and after. At each one: no duplicated update (the step before and after are consecutive), no skipped update, and no re-drawn update. A tag used with two different replies for the same request key counts as re-drawn; so does a replayed step that took a different cached record than the saved log. RNG continuity follows from R2 holding across the boundary, since the replay uses one unbroken RNG per chain |
| R5 provider pinning | every call used by a chain was served by the pinned provider (response `provider` = the endpoint's Stage 1 provider), and its body pins it with `allow_fallbacks: false` |
| R6 invalid / resample | counts per point of first-attempt invalid replies, resampled calls and kept-after-invalid; all consistent between cache and log |
| R7 graph fingerprint | the recorded graph of each chain is the one its logged configurations imply (re-verified at every update) |

Also reported, not pass/fail: cached replicate records **not used by any final chain**, i.e. orphaned. These are
calls from the session-container segments on the original graphs that were later redone on the pod graphs. Count
and cost.

## Combined effect size of the persistence excess (original run + replicate)

- **Per arm and run:** effect = observed persistence (mean of 5) − reference mean; SE = SD of the reference's
  mean-of-5 distribution.
  - Reference: K_avg simulated on that run's actual graphs and starts (R = 400).
  - Original run P1–P4: the kernel-test K_avg sims (`results/llm/kernel_test/sims_<point>_K_avg.csv`).
  - Original nc arms: the same k = 0 K_avg table as the replicate, simulated now on the original graphs and starts.
  - Replicate: `results/llm/stage3_rep1/reanalysis.json`.
- **Arms:** P1–P4, P2-nc, P4-nc (as the pre-registered Stouffer test); P5 reported separately.
- **Fixed-effect estimates:**
  - (a) the inverse-variance weighted mean of the raw differences, with 95% CI;
  - (b) the same on the standardised scale (mean z, SE 1/√n);
  - per run and pooled over both runs;
  - Cochran's Q for heterogeneity between the two runs.
- **Verdicts reported alongside:** the pre-registered pass/fail of each run (original: K_live fails, 2 of P1–P4;
  replicate: Stouffer Z = −0.42, excess does not repeat). P3 |q| stays marked invalid.

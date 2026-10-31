# Stage 3 harness audit: results

Plan: `llm/STAGE3_AUDIT_PLAN.md` (commit 4c3924a, before any result). Code: `llm/stage3_audit.py`,
`llm/stage3_audit_order.py`. Outputs: `results/llm/stage3_audit/`. No Stage 3 rerun, no fix, no API call: everything
was recomputed from the chain logs and the raw call cache (140,005 cached calls with exact request bodies and
replies). Post-hoc diagnostics that the plan did not list are labelled as such.

## Verdict

**The harness is sound.** Every check on the live logs passes with zero mismatches, and the surrogate mock pushed
through the unchanged harness reproduces the simulator within 0.013 on every measure. The pre-stated A rule fails
at P2 and P4-nc, but a post-hoc diagnostic traces both to the committed intervals, not to the harness: the
simulator itself, run on the same 5 graphs, falls inside them only 80–86% of the time. **The Stage 3 failure is not
explained by the harness or the simulator.** One item matters for interpretation (C): live agents react to contact
order, and the fitted surrogate has no order term.

## Pass/fail table

| check | result | evidence |
|---|---|---|
| A P1 | PASS | live seeds: \|m\| 0.401 in [0.24, 0.46], persistence 0.682 in [0.66, 0.74]; repetitions inside 35/40, 36/40 |
| A P2 | **FAIL (rule)** | live seeds: m 0.630 in [0.58, 0.66], persistence 0.637 in [0.59, 0.67]; repetitions inside **29/40** (threshold 30), 37/40 |
| A P3 | PASS | \|m\| 0.213 in [0.20, 0.37], persistence 0.936 in [0.91, 0.95]; 33/40, 34/40 |
| A P4 | PASS | m 0.882 in [0.87, 0.91], persistence 0.806 in [0.75, 0.84]; 36/40, 40/40 |
| A P5 | PASS | \|m\| 0.186 in [0.14, 0.24], persistence 0.167 in [0.11, 0.20]; 33/40, 37/40 |
| A P2-nc | PASS | m 0.411 in [0.34, 0.43], persistence 0.507 in [0.47, 0.54]; 39/40, 34/40 |
| A P4-nc | **FAIL (rule)** | live seeds: m **0.511** vs [0.52, 0.58]; repetitions inside 33/40; persistence 0.462 in [0.41, 0.48], 38/40 |
| B1 update scheme | PASS | all 70 chains: 20 sweeps × 100 updates, steps 0–99; agent draws uniform with replacement (χ² min Bonferroni p 0.03, P3; others ≥ 0.44); updates per agent per sweep match Binomial(100, 1/100); the simulator draws `randint(N)` N times per sweep, identically |
| B2 stale state | PASS | 0 of 140,005 prompts showed a neighbour state different from the replayed current state; replay reproduces all 1,400 saved sweep states; call timestamps strictly increase within every chain. Chains ran in parallel with each other, but each chain's calls were strictly sequential |
| B3 option order / letter mapping | PASS | mapping recovered from the prompt text (content gloss, or own line + replayed state in neutral) and option order from the instruction line: 0 mismatches with the logs; pA recomputed from the raw top-5: 0 mismatches; P5 letter re-parsed: 0 mismatches; applied position vs letter under the recovered mapping: 0 mismatches; exact replay of all 70 chains from the cache through the unchanged harness reproduces every log and state |
| B4 own-position feed | PASS | own line = the agent's current replayed position in 100% of prompts (an independent check in political and workplace; a consistency check in neutral) |
| B5 tie labels | PASS | ally/rival × position shown = the graph's tie signs × current neighbour states in 100% of prompts; ties symmetric on every edge (ε = 0); realised ρ per graph within the binomial 99% interval (P1/P3 0.22–0.28, P2 0.485–0.55, P4 0.21–0.255, P5 0); no contact lines in the nc arms |
| B6 prompt equivalence | PASS | `check_prompt` passes on all 140,005 prompts actually sent; mapping and option order 49.5–50.5% at each point and independent of the own position (all p ≥ 0.12); contact order uniform (all \|z\| ≤ 1.9); every request body equals `battery.body` for its endpoint (T = 1.0; max tokens 1 with logprobs and top-5, or 8 for P5; reasoning flag; provider pin) and matches 14,278–24,841 Stage 1 bodies field for field, except the message |
| B7 provider pinning | PASS | one provider per point, the same as Stage 1: Alibaba (P1), Parasail (P2, P2-nc), CoreWeave (P3), Io Net (P4, P4-nc), OpenAI (P5); `allow_fallbacks: false` in every body; no fallback |
| B8 invalid replies | PASS | first-attempt invalid: P2 4/20,000, P4-nc 1/20,000, all other points 0; every one was an empty reply, resampled valid on the 2nd attempt; kept after invalid 0; too few to test for clustering |
| B9 time alignment | PASS | both: independent 50/50 start (live initial m z: mean −0.11, var 1.11 over 70 chains), 10 warm-up + 10 measured sweeps (`stage2.jobs_stage3` and every `stage3.csv` row: 10/10), persistence reference = state at the end of warm-up |
| B10 measurement code | PASS | `surrogate.run` with only its update rule replaced by a replay of the logged updates (measurement lines executed verbatim; `replay_measure_source.py`) reproduces `measures_by_graph.csv` and `summary.csv` to 1.1e-16 |
| B11 surrogate side | PASS | each committed interval comes from 40 runs of the right endpoint, framing arm, ρ, rrg, ε = 0, N = 100, 10/10 sweeps, no explicit jitter; mode fitted (nc: fields-only); fit files and `stage3.csv` unchanged since the predictions commit 878724e; intervals recomputed to 1e-16; neutral h applied as h·L; L and O redrawn on every update |
| C P1 | PASS | observed A 0.490 vs battery 0.493; slope 0.91 [0.89, 0.94]; RMS logit, live vs battery pA: 0.83 |
| C P2 | PASS | 0.492 vs 0.492; slope 0.97 [0.94, 0.99]; RMS logit 0.40 |
| C P3 | **FLAG** | 0.507 vs 0.506; slope **0.67** [0.64, 0.70]; RMS logit 0.55 |
| C P4 | PASS | 0.502 vs 0.504; slope 0.95 [0.92, 0.98]; RMS logit 0.78 |
| C P5 | **FLAG** | 0.488 vs 0.484; slope **0.39** [0.38, 0.40] |
| C P2-nc, P4-nc | n/a | no battery cell at k = 0 |

## The failures and flags, and what they do to the Stage 3 measures

**A at P2 and P4-nc: the intervals, not the harness** (post-hoc diagnostic, `A_diag_sim_same_graphs.*`). I ran
`surrogate.run` itself on the same 5 live graphs, with the same per-graph parameter draws as the mock repetitions.

- Mock (harness) vs simulator, paired by draw: largest difference 0.013, largest \|z\| 2.3 over 42 comparisons (P1
  and P3 at 160 repetitions, the rest at 40). No detectable harness effect.
- The committed intervals resample 40 runs, each on a fresh graph; they are not conditional on the 5 graphs actually
  used. On these 5 graphs the simulator falls inside them in 319/400 (P2 m) and 343/400 (P4-nc m) runs, i.e. 80% and
  86% instead of 90%. The other primary measures: 84–95%.
- **Effect on Stage 3:** a slightly higher chance of a miss for a correct surrogate (per primary measure about
  0.80–0.95 instead of 0.90). Against the same-graph surrogate distribution, the live misses are:
  - P1 persistence +3.6 SD (\|m\| −1.5 SD);
  - P3 \|m\| −3.7 SD, persistence +5.7 SD;
  - P4 m +3.8 SD, persistence +4.3 SD;
  - P5 +12 and +20 SD;
  - P2-nc −14 and +13 SD; P4-nc +4 and +14 SD.

  These cannot come from interval coverage. **P2 is the exception:** live m 0.561 and persistence 0.603 are −1.5 and
  −1.0 SD from the same-graph surrogate (2/40 and 6/40 repetitions at or below the live values). P2's miss is
  therefore consistent with a correct surrogate on these graphs. This does not change the pre-stated verdict: P1, P3
  and P4 miss clearly, so at most 1 of 4 points could pass.

**C at P3: an artefact of near-certain battery probabilities.** Every P3 battery cell is near 0 or 1 (Llama-70B,
γ ≈ 5.8). Live calls choose the minority letter in 0.4% of calls where the battery says ≈ 0.0%. On the logit scale,
clipped at 1e-4, that compresses the slope. Per-step behaviour matches in probability (calibration-in-the-large
+0.001).

**C at P5: battery noise plus order sensitivity.** The P5 battery P(A) is a share of 30 sampled replies per cell,
so its logit is noisy and the slope is attenuated. Binned, though, there are real differences (e.g. battery 0.76 →
live 0.39 over 384 calls), and they are order effects; see the next point. P5 is a calibration point and does not
count toward the verdict.

**Contact order: the main thing the audit turned up** (post-hoc, `C_order_matched.json`). Live calls were split by
whether their contact order happened to equal the order in the Stage 1 battery prompt for that cell.

| point | same order as battery: RMS logit (live vs battery pA) | different order |
|---|---|---|
| P1 (Qwen-122B) | 0.18 (n 4,823) | 0.95 (n 15,177) |
| P2 (Qwen-9B) | 0.06 (n 3,552) | 0.44 (n 16,448) |
| P3 (Llama-70B) | 0.08 (n 3,679) | 0.60 (n 16,321) |
| P4 (Nemo-12B) | 0.35 (n 8,106) | 0.97 (n 11,894) |
| P5 (GPT-6 Luna), mean \|Δ share\| | 0.04 (n 11,416) | 0.21 (n 8,584) |

- On identical prompts, live agents gave the battery's answer: providers did not drift, and the harness fed them the
  right states.
- Contact order alone moves a single response by 0.4–1.0 nats.
- The A3 surrogate was fitted on one fixed order per cell and has no order term. In the live runs order is redrawn on
  every call, so the response varies from call to call beyond the fitted rule. Means agree (calibration-in-the-large
  within ±0.003 at every point), but the surrogate does not describe that variance.
- **Likely effect:** a mechanism the harness cannot fix and the mock cannot show. Its size in the dynamics is
  unknown, because order effects need not average out: primacy or recency of a contact behaves like a partial voter
  rule. That is a model question for the owner, not a harness fault.

**The no-contacts arms miss far more than the network points.** On the same graphs and starts, the mock gives the
committed fields-only values, while live agents show:
- P2-nc: m 0.15 (mock 0.41), persistence 0.80 (mock 0.51);
- P4-nc: m 0.64 (mock 0.54), persistence 0.70 (mock 0.45).

The k = 0 prompt changes the fields and inertia, as the addendum's extrapolation flag anticipated. No harness check
explains it (B2–B6 pass for these arms).

## Not done (by instruction)

No rerun, no fix, no new live call, no change to any Stage 3 result or verdict.

# Stage 3 harness audit: plan (committed before any audit result)

Owner request 2026-10-12. Purpose: decide whether the Stage 3 "surrogate not supported" result (EXECUTION_LOG entry 26)
could be caused by the live harness or the simulator rather than by the agents. **Constraints:** no reruns of Stage 3,
no fixes to the harness or the simulator, no live API calls. Everything below reads the committed chain logs
(`results/llm/stage3/chains/`), the raw call cache (`results/llm/cache/*.jsonl`, records tagged `stage3:<point>:<graph>:<replica>:<sweep>:<step>:<attempt>`,
each holding the exact request body and the compact response) and the Stage 1/2 files. Implementation:
`llm/stage3_audit.py` (audit-only code; it imports the harness and simulator unchanged). Outputs:
`results/llm/stage3_audit/`. Then stop and report.

Points: P1–P5, P2-nc, P4-nc; 5 graphs × 2 replicas × 2,000 updates each (140,000 updates).

## A. Known-answer test with a surrogate mock agent

- **Mock agent.** Replaces only the API call inside `stage3.chain` (its `call=` hook). It reads the prompt it is
  given, exactly as built by the harness, and answers with the fitted surrogate's own P(choice | prompt):
  - A3 points: logit P(A) = k^-α (b_a M_a − b_r M_r) + γ·own + h_C·[A is content P] + h_L + h_O·[A listed first],
    all in letter space (neutral: h is the letter-A field, no content field).
  - P5 (MF surrogate): with probability p follow the first-listed contact (copy an ally's letter, take the other
    letter from a rival), else the A3 logit.
  - Answer format as the real endpoint: first-token top logprobs for logprob endpoints (P1–P4, nc arms), a sampled
    single letter for P5. Parsing, letter sampling, state updates, invalid handling and measures are the harness's own.
  - Parameters: one cluster-bootstrap draw per graph, as Stage 2 (`stage2.param_row`, mode "fitted"); nc arms
    have no contact lines, so the social terms vanish by construction (= fields-only control, the committed prediction).
- **Runs.** (i) The live graphs and live chain seeds (same initial states, same per-call randomisation stream).
  (ii) 40 repetitions on the same graphs with fresh chain seeds and fresh draws.
- **Pass:** for every point, both primary measures of run (i) inside the committed Stage 2 90% mean-of-5 interval,
  AND across the 40 repetitions each primary measure inside its interval in ≥ 30 of 40 (a correct 90% interval
  falls below 30/40 with probability ≈ 0.5%). All six measures are reported. **A fail means the harness or the
  simulator is wrong.**

## B. Checks on the real live logs

Each check is recomputed independently from the raw cache (actual prompt text sent, raw reply) where possible,
not from the runner's own derived columns.

1. **Update scheme.** Pass: every chain has 20 sweeps × 100 updates; agents drawn uniformly with replacement
   (χ² on agent counts per chain, updates per agent per sweep ~ Binomial(100, 1/100)); identical to the simulator
   (`surrogate.run`: `randint(N)` per update, N updates per sweep, two independent replicas).
2. **Stale state.** Replay each chain from its initial state and the logged updates. Pass: the replay reproduces
   every saved sweep state; for 100% of calls the neighbour positions shown in the prompt text equal the replayed
   current positions at the time the reply was applied; cache timestamps increase within each chain (calls of a
   chain strictly sequential).
3. **Option-order / letter mapping.** From the prompt text alone, recover which letter means position P (own line +
   replayed own state in neutral; the content gloss in political/workplace), and the option order. From the raw
   reply recompute pA (logprob endpoints) or the parsed letter (P5). Pass: 0 mismatches between recovered and logged
   mapping/order/pA/letter, and 0 mismatches between the applied new position and letter → position under the
   recovered mapping. Plus an exact replay of every chain from the cache (no API calls) reproducing all logs.
4. **Own-position feed.** Pass: the own-position line of 100% of prompts equals the agent's replayed current
   position (and its last applied update, when it had one).
5. **Tie labels.** Pass: the ally/rival × position multiset in 100% of prompts equals the one implied by the graph's
   tie signs and replayed neighbour states; ties symmetric (ε = 0) on every edge; realised rival share per graph
   consistent with ρ (binomial 99% interval).
6. **Prompt equivalence.** Pass: `check_prompt` passes on 100% of the prompts actually sent (re-run on the cache);
   mapping and option order each ≈ 50% (binomial p > 0.001 per point) and independent of the own position; contact
   order uniform (share of calls whose first-listed contact type is that of the graph's first-stored neighbour vs
   its expectation); every request body equals `battery.body(...)` for the endpoint (temperature 1.0, max tokens,
   logprobs, reasoning flag, provider pin) and matches a Stage 1 main-battery body of the same endpoint in every
   field except the message.
7. **Provider pinning.** Pass: every live call served by the same provider as the endpoint's Stage 1 main calls
   (response `provider` field), every body pins it with `allow_fallbacks: false`. Reported per point.
8. **Invalid replies.** Rate per point (first attempt invalid), resamples, kept-after-invalid. Clustering by own
   position, mapping, order and contact configuration (χ² where ≥ 20 invalid). Pass: ≤ 2% at every point (the
   pre-stated flag) and no factor with p < 0.001.
9. **Time alignment.** Pass: same initial-state distribution (independent 50/50 per agent; live initial m consistent
   with Binomial(100, ½)), same warm-up (10) and measured sweeps (10), same persistence reference (state at the end of
   warm-up) in harness and simulator.
10. **Measurement code.** Recompute every live measure from the replayed trajectories with the simulator's own
    measurement code (the measurement block of `surrogate.run`, executed verbatim on the replayed states). Pass:
    agrees with `measures_by_graph.csv` and `summary.csv` to 1e-9.
11. **Surrogate side.** Pass: the Stage 2 rows behind each committed interval use the point's endpoint, framing arm,
    ρ, rrg graph, ε = 0, mode "fitted", the fit file unchanged since the predictions commit, h_L·L and h_O·O
    redrawn per call; the committed intervals are reproducible from `results/llm/stage2/stage3.csv`.

## C. Single-step calibration

For live calls whose cell (contact multiset, own position, mapping, option order; neutral mapping 1 relabelled as
in `check_prompt`) is in the Stage 1 main battery (all P1–P5 calls; nc arms have no battery cell), compare the live
choice (letter A or B) with the battery P(A) for that cell (mean over Stage 1 replicates; contact order differs).
Report a calibration table (10 bins of battery P(A), count, mean battery P(A), observed share A) per point, the
calibration-in-the-large (mean observed − mean battery) and the logistic calibration slope with 95% CI; for logprob
endpoints also the RMS logit difference between live pA and battery pA. Descriptive; flagged if
|calibration-in-the-large| > 0.03 or the slope CI excludes [0.8, 1.2] entirely.

## What a failure would mean for the Stage 3 measures

- A fails → the Stage 3 verdict is uninterpretable; the harness or simulator must be fixed before any reading.
- B1/B2/B9 → wrong dynamics class (e.g. synchronous updates or a different time window): all dynamical measures
  (persistence, flip rate, |m|) shift systematically.
- B3/B4/B5 → agents saw wrong states or had replies applied to the wrong position: biases m and |m|, destroys
  persistence/ordering.
- B6/B7 → live prompts or serving differ from the fitted single-shot conditions: per-step response differs (C shows
  how much).
- B8 → kept-after-invalid inflates persistence where invalids cluster.
- B10/B11 → reported numbers or intervals are not what the protocol defines.
- C off while A and B pass → the agents' per-step behaviour in the live run differs from the battery (provider
  drift, context), not a harness fault.

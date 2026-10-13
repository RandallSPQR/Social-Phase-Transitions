# Residual test: heterogeneity and a fresh live replicate (plan and predictions, committed before either test runs)

Owner request 2026-10-12. Question: why do the live Stage 3 trajectories show more persistence (and fewer flips)
than every Markov kernel in (configuration c, own position s0), including the empirical live kernel K_live and the
order-marginalised battery K_avg (`results/llm/KERNEL_TEST_RESULTS.md`)? No claim, and no amplification curve, until
this is resolved.

## Test 1: heterogeneity / mover-stayer (free; `llm/residual_heterogeneity.py`)

- **a) Tokens.**
  - List everything in a Stage 3 request that can vary between agents: prompt text and request body.
  - Empirical check over all 140,005 cached prompts: group by (framing, c, s0, letter mapping, option order, listed
    contact order) and count distinct prompt texts per group.
  - Confirm the request body carries nothing agent-specific (the cache tag is a local key, never sent).
- **b) Between-agent variance.**
  - Per update: residual r = P(stay | call) − K_live_stay(c, s0). For logprob endpoints P(stay | call) is the call's
    own probability; for P5 it is the sampled outcome.
  - Agent unit = (point, chain, node).
  - Statistic: between-agent sum of squares Σ_a n_a (r̄_a − r̄)².
  - Null: agent labels shuffled across updates within each point (counts per label kept), 1,000 permutations;
    one-sided p.
  - Repeated for the graph-position unit (point, graph, node), pooling the two replicas (and, for nc arms, the
    parent point's graph).
  - Combined over the 7 points by Fisher's method.
- **c) Only if b) finds variance above the null** (Fisher p < 0.05):
  - Estimate the between-agent SD τ of a logit-scale offset (method of moments from b's statistic).
  - Simulate K_live with per-agent offsets ~ N(0, τ²) fixed per node, on the live graphs and starts (R = 400).
  - Test whether it removes the persistence excess and the flip-rate deficit: Stouffer z of the original live data
    against the heterogeneous-kernel simulations, P1–P4 + nc arms.
  - "Heterogeneity explains it" = that Stouffer z for persistence is < 1.645, while against homogeneous K_live it
    is ≥ 1.645.

**Predictions for test 1.**
- a) Nothing varies between agents beyond (c, s0), the letter mapping, the option order and the contact order. The
  template has no names, IDs or labels, and each group has exactly one prompt text.
- b) For logprob endpoints a call's probability is a function of its prompt text alone. Agents differ only in which
  prompts they get, which the residual conditions on (c, s0) and averages over the random presentation. So
  between-agent and between-position variance should be at the permutation null: Fisher p > 0.05 for both units,
  and per point at most about one p < 0.05 by chance.
- c) Not triggered.

## Test 2: fresh live replicate (paid; cost estimate below; not started without the owner's OK)

- **Design.** The same 5 graphs per point and the same protocol as Stage 3 (N = 100, 2 replicas per graph, 10 + 10
  sweeps, per-call randomisation, resample-on-invalid, the same endpoints, pinned providers and request bodies).
  - Fresh starts and seeds via `stage3.chain(..., run_tag="rep1")`, with cache tags `stage3-rep1:...`.
  - The original chains still replay byte-identically after this change (checked).
  - **No concurrency:** at most one call in flight per endpoint (see the options for the wall-clock trade-off).
- **Reference kernels.**
  - P1–P5: K_avg, the order-marginalised one-shot battery, already measured.
  - P2-nc, P4-nc: a k = 0 K_avg battery: 2 s0 × 2 letter mappings × 2 option orders × 2 replicates = 16 calls per
    arm, run first. Its predictions are committed before any replicate chain. Secondary nc reference: K_live from
    the original run (numbers below).
- **Pre-registered tests.**
  - **Primary:** pooled one-sided test that live persistence exceeds the reference prediction, over P1–P4 + P2-nc +
    P4-nc. Each arm: z_i = (observed persistence − reference mean) / reference SD, simulated on the replicate's
    graphs and starts (R = 400). Z = Σ z_i / √6. **"Excess repeats" if Z ≥ 1.645** (one-sided 5%).
  - **Per point:** observed mean of 5 inside the reference 90% interval, for every measure. Flip rate is reported
    as the companion to persistence.
  - **Repeat-or-move:** whether the original misses recur: P1 |m| below the interval, and P3 persistence above it.
  - P5 is reported and does not count, as in Stage 3.

**Numeric predictions (K_avg on the replicate's graphs and starts; mean [90% interval of the mean of 5]; R = 400;
`results/llm/replicate/predictions.json`):**

| point | \|m\| | m | persistence | flip rate |
|---|---|---|---|---|
| P1 | 0.361 [0.273, 0.449] | −0.048 [−0.176, 0.075] | 0.795 [0.765, 0.825] | 0.028 [0.025, 0.032] |
| P2 | 0.577 [0.522, 0.631] | 0.577 [0.522, 0.631] | 0.614 [0.581, 0.650] | 0.059 [0.055, 0.064] |
| P3 | 0.157 [0.140, 0.176] | −0.074 [−0.092, −0.053] | 0.980 [0.970, 0.987] | 0.002 [0.001, 0.002] |
| P4 | 0.937 [0.923, 0.950] | 0.937 [0.923, 0.950] | 0.880 [0.854, 0.903] | 0.027 [0.024, 0.032] |
| P5 | 0.382 [0.238, 0.524] | 0.003 [−0.222, 0.209] | 0.477 [0.388, 0.570] | 0.135 [0.106, 0.166] |
| P2-nc (secondary, K_live) | 0.188 [0.150, 0.227] | 0.184 [0.147, 0.227] | 0.771 [0.745, 0.796] | 0.024 [0.022, 0.027] |
| P4-nc (secondary, K_live) | 0.616 [0.587, 0.650] | 0.616 [0.587, 0.650] | 0.655 [0.621, 0.682] | 0.048 [0.044, 0.052] |

## Interpretation map (owner's wording; pre-registered)

| test 1 (heterogeneity explains it) | test 2 (excess repeats, Z ≥ 1.645) | reading |
|---|---|---|
| yes | yes | Resolved: agents are persistently heterogeneous. The narrowed headline applies ("trait summaries fail; full conditional tables work"), with heterogeneity as an extra required dimension of the table. |
| no | yes | Unexplained non-Markov persistence; reported as an open anomaly. |
| either | no | The original misses were realisation noise; the narrowed headline applies. |
| yes | no | (Not in the owner's map; recorded in advance.) Realisation noise governs the verdict; the heterogeneity estimate is reported as descriptive. |

## Cost and wall-clock estimate for test 2

From the Stage 3 spend ($4.95 for 140,000 updates) and the measured sequential latency (median 1.8–2.2 s per call;
mean 1.8–2.4 s by endpoint):

| option | calls | cost | wall clock |
|---|---|---|---|
| A. Full replicate, strictly one call in flight overall | 140,000 + 32 | ≈ $4.95 | ≈ 84 h |
| B. Full replicate, one call in flight per endpoint (5 streams: Alibaba, Parasail [P2 + P2-nc], CoreWeave, Io Net [P4 + P4-nc], OpenAI) | 140,000 + 32 | ≈ $4.95 | ≈ 27 h (Parasail stream) |
| C. As B, but one replica per graph (persistence, m, flip rate need one; \|q\| is dropped) | 70,000 + 32 | ≈ $2.50 | ≈ 14 h |

- **Execution (repo rules for long jobs).**
  - Every call is appended to the on-disk cache as it returns (resume = rerun, free).
  - Each finished chain is saved.
  - Every 15 min a status file (calls done, spend, ETA per stream) and a cache snapshot are uploaded to the private
    HF dataset and verified.
  - Progress goes to stdout and the job log; budget stop at $7.
  - Background job with a PID file.
  - The container is ephemeral: a restart costs at most the calls in flight, and the run resumes from the cache.

## Amendment 1 (owner decision 2026-10-13, before any paid call): option B, power, sentinels, identical-request test

### 1. Power of option B

- Original effect size: persistence z of the original live data against K_avg (P1 −0.50, P2 −0.13, P3 +1.62, P4
  +1.17) and against K_live for the nc arms (P2-nc +1.67, P4-nc +2.32). Pooled Stouffer Z = **2.51**.
- If the replicate's true effect equals this, its Z ~ N(2.51, 1): **power to reach Z ≥ 1.645 is 0.81**.
- At 75% and 50% of that effect: 0.59 and 0.35.
- The original estimate is selected (we are testing it because it looked large), so the true power is probably
  below 0.81.
- For reference, option C would have had 0.55 / 0.38 / 0.22.

### 2. Interpretation asymmetry (added to the map)

Option B changes two things at once: it removes our own concurrency (one call in flight per endpoint), and it is a
fresh realisation.
- **Excess repeats (Z ≥ 1.645)** → a real anomaly: not realisation noise, and not caused by our concurrency.
- **Excess vanishes** → "realisation noise **or** our own concurrency; not separable". It is never reported as noise
  alone.

### 3. Provider-drift sentinels

- Per endpoint, 3 fixed prompts, issued every 5 minutes for the whole run, interleaved in that endpoint's single
  stream (so still one call in flight per endpoint). Endpoints whose chains finish early keep issuing sentinels
  until the last stream ends.
  - One prompt from the K_avg battery whose measured P(A) is closest to 0.5 for the endpoint's framing.
  - Two prompts from the two most frequent live configurations, with fixed presentations.
  - Logprob endpoints: 1 call per sentinel. P5 (sampled): 5 samples per sentinel.
- Cache tags `sentinel-rep1:<endpoint>:<prompt>:<round>:<sample>`. Expected ≈ 8,700 calls (≈ 6% extra, not 1%; with
  5-minute rounds over ≈ 27 h), ≈ $0.28.
- **Pre-registered analysis:**
  - Drift trace per endpoint: logit P(A) of each sentinel over time (P5: share A per round). The drift statistic is
    the per-round mean logit deviation from the sentinel's whole-run mean, signed toward position P, plus the
    variance of rounds over the measurement noise.
  - Correlation test: per replicate chain (each occupies a contiguous time window in its stream), the mean sentinel
    drift during the chain vs that chain's persistence and signed m (oriented to the field).
  - Spearman per endpoint, combined over endpoints by Fisher's z; two-sided p < 0.05.
  - Mechanism hypothesis: slow provider-side drift acts as a time-varying shared field.
- Monitored during the run for operational health only. No interim look at the primary test.

### 4. Identical-request dependence test (free; run before any paid call)

- **How each live choice was produced:**
  - Logprob endpoints (P1–P4, P2-nc, P4-nc): the provider returns the first-token top-5 logprobs, and the provider's
    sampled token is not used. The choice is a **local draw**, letter A if `rng.random() < pA`. `rng` is the chain's
    Python `random.Random` (Mersenne Twister), seeded by SHA-256 of ("stage3-chain", point, graph, replica). The same
    stream also draws the updating agent and the per-call presentation.
  - P5 (sampled endpoint): the choice is the **provider-sampled reply** (T = 1, max 8 tokens), parsed to one letter;
    no local draw.
- **Pairs:** within each chain, every call whose full prompt text exactly repeats an earlier call by the same agent,
  paired with that agent's most recent identical call. The agreement is the same letter.
- **Expected agreement under independence:**
  - (i) the owner's: P² + (1 − P)², with P = K_live for that configuration, expressed in the prompt's letter space;
  - (ii) the prompt-level: P1·P2 + (1 − P1)(1 − P2). Logprob endpoints use the two calls' own returned P(A). P5 uses
    P(A) of that exact prompt text, estimated from all other live calls with the same text (both calls of the pair
    left out).
  - (i) is biased upward in agreement-deficit terms, i.e. it makes (i)-excess easier to find, because repeated full
    prompts share their presentation, and E[p² + (1 − p)²] ≥ P̄² + (1 − P̄)². (ii) is the exact test, and is
    primary.
- **Test:** observed − expected agreement per point, chain-cluster bootstrap 95% CI (1,000 draws). Excess = the CI
  under (ii) lies above 0 at a point.
- **If excess:** re-simulate K_live with a repeat rule. When an agent's current prompt equals its previous one, its
  choice repeats the earlier choice with the measured excess probability, otherwise it is drawn from the kernel.
  Check whether this reproduces the persistence excess (Stouffer over P1–P4 + nc vs the homogeneous K_live sims).
- **Outcome map (owner's):**
  - excess agreement that reproduces the persistence excess → mechanism found (non-independent sampling on repeated
    requests), and the replicate becomes confirmatory;
  - no excess → run option B with sentinels as approved, with drift the leading remaining hypothesis.
- **Prediction:** no excess at the logprob endpoints, because their draws are local and seeded independently of the
  reply. P5 is the only point where provider-side sampling could carry dependence.

### 5. Execution of option B

- One stream per endpoint (Alibaba P1; Parasail P2 then P2-nc; CoreWeave P3; Io Net P4 then P4-nc; OpenAI P5), each
  strictly sequential.
- The k = 0 K_avg battery (32 calls) runs first, and its predictions for the nc arms are committed before any chain.
- $7 budget stop. Resumable: per-call cache, and a chain file is saved when the chain ends.
- Every 15 min: `status.json` (calls, spend, ETA per stream, sentinel health) and gzip copies of the five cache files
  are uploaded to the private HF dataset under `runs/stage3-rep1/` and verified by a remote listing with sizes. The
  raw cache is never rewritten while the run is live.
- Background job with a PID file. Self check-ins restart it if the container drops it.
- **The primary analysis runs once, after all chains finish.**

### Amendment 1, addendum: nc reference predictions (k = 0 K_avg battery, 32 calls, run 2026-10-13 before any replicate chain)

Primary nc reference, simulated on the replicate's graphs and starts (R = 400; `results/llm/replicate/predictions.json`,
`k_avg_nc_calls.csv`):

| arm | m | persistence | flip rate |
|---|---|---|---|
| P2-nc | 0.186 [0.145, 0.228] | 0.770 [0.746, 0.796] | 0.024 [0.022, 0.027] |
| P4-nc | 0.604 [0.572, 0.636] | 0.639 [0.611, 0.668] | 0.051 [0.047, 0.055] |

**Runner:** `llm/stage3_replicate.py`. It was proved end to end in mock mode, with no API calls: 70 chains, 5
streams, sentinels, heartbeats, an HF upload verified 148/148 files, and a restart that skips finished chains.

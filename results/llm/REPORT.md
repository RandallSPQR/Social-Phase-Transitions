# Do LLM agents behave like spins? Response functions, live networks and amplification near tipping points

**Technical report (preprint draft), 2026-10-14.** Repository `randallspqr/social-phase-transitions`, branch
`claude/peaceful-fermi-hz11cv`.

Every number below is taken from a committed results file, cited in brackets as `[file]`. Paths are relative to the
repository root. Pre-registered verdicts are always stated before any post-hoc reading, and post-hoc readings are
labelled as such. Refuted and failed results are reported in the main text.

---

## Headline summary

- **Single-agent response functions.** Across the four open-weight LLM endpoints that passed every validity gate (of nine tested), the agents are not Ising spins.
  - They weight rivals less than allies: valence asymmetry, pre-registered H1, in 9 of 12 confirmatory cells.
  - They carry strong, framing-dependent inertia.
  - They respond to label, option-order and contact-order cues.
  - They are mostly additive.
- **Live networks.** The surrogate fitted to those response functions mispredicted live 100-agent networks, which
  fails the pre-registered Stage 3 test.
  - The live harness was audited and is sound.
  - Most of the miss came from the parametric "trait summary" of each model, not from the agents.
- **What does work.** A conditional response table, averaged over the presentation details the agents are sensitive
  to (K_avg), reproduces live per-step behaviour to |ΔP| ≤ 0.004. It predicts the population as well as the live
  empirical kernel does.
  - Both still miss 2 of 4 points collectively. Per-step differences that small can be amplified into collective
    differences of several standard deviations.
- **Persistence excess.** A small excess of persistence over every Markov kernel in the original run did not repeat
  in a fresh replicate. It is reported as realisation noise or our own concurrency; the two cannot be separated.
- **Amplification near tipping points.** The first pre-registered test was **refuted**: its control comparison was
  ill-defined when the control's amplification was near zero. A corrected rule was then tested on fresh simulations.
  - It passes for **all three synthetic error shapes in both confirmatory models that have a tipping point** (Qwen3.5
    122B and Gemini 3.1 Flash-Lite).
  - A real model with no tipping point (GPT-4o-mini) stays below the amplification floor everywhere.
- **Tipping points are model-dependent.** Only **2 of 4** confirmatory models have a tipping point in the swept range
  of decision noise. That is knowable only by simulating the population, not from single-agent statistics.
- **Scope throughout:** these models, this binary task, memoryless agents, random-regular degree-4 graphs.

---

## Abstract

We ask whether large language model (LLM) agents placed on a signed social network behave like the binary spins of
statistical physics, and what that implies for evaluating them one at a time. We make five claims, each limited to
the models, binary task and graphs tested.

1. **LLM response functions are not Ising-like.**
   - In a pre-registered battery (4 confirmatory endpoints × 3 framings, Holm-corrected), rivals carry less weight
     than allies in 9 of 12 cells. It is 12 of 15 with a self-hosted Gemma-4 31B and 15 of 18 with a further rescue
     endpoint.
   - Label fields are present in 10 of 12 cells. Inertia appears whenever positions have content.
   - Non-additive responses clear the pre-registered minimum effect in 2 of 12 cells.
   - Size-dependence of the coupling (H4) is not supported.
2. **Trait-level summaries fail to predict live collective behaviour; order-averaged conditional tables succeed per
   step.**
   - A surrogate built from fitted response functions passed 0 of 4 pre-registered live points (rule: ≥ 3).
   - An order-marginalised one-shot battery (K_avg) matches the live per-step kernel to |ΔP| ≤ 0.004 at the four
     network points.
3. **Small per-step differences are amplified collectively.**
   - At one point a per-step difference of |ΔP| = 0.002 between two kernels became a 3.4-SD difference in persistence.
   - The parametric surrogate's 0.2–1.6-nat per-step error became 3–20 SD.
4. **Near a tipping point, amplification is much larger and exceeds controls.**
   - Under a corrected rule tested on fresh simulations, amplification at T_c exceeds the worst-case control by
     0.15–2.03 |m| per nat for 3 of 3 synthetic error shapes in both confirmatory models with a transition.
   - The original pre-registered version of this test was refuted, for a stated technical reason.
5. **Whether a population has a tipping point is model-dependent.** Only 2 of 4 confirmatory models have an
   order–disorder transition in the swept decision-noise range. This is a property of the population, not readable from single-agent
   statistics.

Amplification results are simulation-based: empirical kernels are run under a 1/T decision-noise rescaling that is
verified only for T > 1. The persistence excess over Markov kernels seen in one live run did not repeat (footnote-level
result).

---

## 1. Motivation

**Pairwise versus higher-order structure.** The project began from five unrefereed mathematical results released
publicly in September 2026 (PDFs in `papers/`):

- Rokhlin multiple mixing for one transformation;
- the Mézard–Parisi formula for diluted spin glasses;
- no percolation at criticality on quasi-transitive graphs;
- critical bond and site percolation on the cubic lattice;
- nonuniqueness of percolation on nonamenable graphs.

`HANDOFF.md` §1 summarises the motivating chain:
- a single time axis cannot build collective structure (multiple mixing), so the interaction network is necessary;
- on sparse signed networks, the possible collective states are consensus, disorder, or frozen incompatible camps (the
  diluted spin glass).

These results are **motivation only**:
- they are unrefereed;
- they were public for less than a day when this work began (`HANDOFF.md` §1);
- nothing in this report depends on their correctness.

**The question this report addresses.** If LLM agents are wired into a signed network, do they behave like the spins
this theory assumes? Where they do not, which differences matter for collective outcomes?

**Relation to discrete-choice models of social interaction.** The agent update rule studied here is a logit
discrete-choice model with a social term: the probability of choosing a position depends on the choices of an
agent's contacts. That is the framework of Brock and Durlauf (2001), whose mean-field analysis already predicts
multiple equilibria and tipping. Inferring such social effects from aggregate outcomes runs into Manski's (1993)
reflection problem. Here we sidestep it by measuring each agent's response function directly, with the
contacts' positions under experimental control.

The practical stake is evaluation. Most evaluations measure one agent's response to a fixed prompt. If collective
behaviour depends on details that single-agent summaries drop, and if small per-step differences are amplified near
tipping points, then single-agent evaluation can mislead about what a population will do.

---

## 2. Symmetric baseline and asymmetry sweep (known physics, reproduced)

**Model.** The baseline is the Viana–Bray (1985) ±J Ising model on Erdős–Rényi graphs (`HANDOFF.md` §2):
- mean degree c, a fraction ρ of negative ties, heat-bath dynamics at noise T;
- analytic cavity lines: consensus at c(1 − 2ρ) tanh(1/T) = 1, glass at c tanh²(1/T) = 1.

**Validation** (parallel tempering, finite-size crossings, 1,000 graph bootstraps): all six simulated transitions fall
within their 95% CIs of theory [`results/validation_crossings.csv`; `HANDOFF.md` §3a; `figures/fig3_crossings_c4.png`].

| transition | theory T | simulated T (N = 1000/2000) | 95% CI |
|---|---|---|---|
| consensus ρ = 0 | 3.915 | 3.917 | 3.85–4.01 |
| consensus ρ = 0.1 | 3.093 | 3.099 | 3.04–3.19 |
| consensus ρ = 0.2 | 2.254 | 2.156 | 1.99–2.28 |
| glass c = 3 | 1.519 | 1.528 | 1.45–1.58 |
| glass c = 4 | 1.820 | 1.813 | 1.74–1.87 |
| glass c = 6 | 2.307 | 2.233 | 2.10–2.32 |

**Asymmetry sweep** [`results/asymmetry_sweep.csv`; `HANDOFF.md` §3b; `figures/fig4_asymmetry.png`]. Non-reciprocity ε
is the probability that the reverse direction of a tie is redrawn independently.
- **Glass (ρ = 0.5, c = 4, T = 1.0):** persistence falls 0.51 → 0.28 → 0.11 → ≈ 0.01 as ε goes 0 → 0.1 → 0.2 → 0.35,
  with no size dependence across N = 1k–4k.
- **Consensus (ρ = 0.1, T = 2.0):** |m| goes only from 0.62 to 0.55 as ε goes 0 → 1.

Frozen camps melt quickly under non-reciprocity. This is consistent with Crisanti and Sompolinsky's (1987) result for dense
asymmetric spin glasses and is **not new physics**. We report it only to establish that the simulator reproduces known
results.

**Two-party polarisation** (`results/balanced_polarization.csv`) is small and suggestive only (`HANDOFF.md` §3c). It
plays no part in the claims below.

---

## 3. Stage 1: LLM response functions

### 3.1 Design (pre-registered)

- **Pre-registration:** `llm/PREREG.md`, committed e9c7e47 before any battery data. Amendments 1–3 were also committed
  before any main-battery data (8c0ab35, 9bf6c28, 483decc).
- **Task:** each agent holds position A or B and sees k = 1–6 contacts. Each contact is labelled ally or rival and
  shown with that contact's current position. The agent chooses A or B.
- **Response function:** P(A) is read from first-token logprobs (renormalised over A/B), or estimated from sampled
  replies for frontier spot checks.
- **Framings:** neutral, political and workplace. Letter mapping and option order are swapped across the battery.
- **Validity gates:**
  - a comprehension battery: 2,116 factual questions about own position and ally/rival majorities, ≥ 90% per type to
    pass;
  - an endpoint noise gate: repeat SD ≤ 0.3 nats on the logit scale.
- **Estimand:** the order-averaged (marginal) response function, fitted by cross-entropy (Amendment 1.2).
- **Hypotheses (H1–H5),** Holm-corrected across cells:
  - H1: valence asymmetry, w_rival ≠ w_ally;
  - H2: label fields;
  - H3: non-additive responses (pairwise and cubic blocks, with a pre-registered minimum effect);
  - H4: β rising with model size (supported if ≥ 3 of 4 families in neutral framing);
  - H5: summation vs averaging over contacts, α.

### 3.2 Confirmatory set

Four endpoints pass both gates [`results/llm/STAGE1_RESULTS.md`]: Llama-3.3 70B, Qwen3.5 9B, Qwen3.5 122B-A10B and
Mistral Nemo 12B. That gives 4 × 3 framings = 12 cells.

| endpoint | failure |
|---|---|
| Llama-3.1 8B | comprehension: ally 0.78, rival 0.72 |
| Mistral Small 24B | leak: reasons before answering |
| Gemma-4 26B and 31B (providers) | noise gate: 0.354–0.703 |
| Mistral Large 4 | noise gate |

Stage 1 cost $9.56.

### 3.3 Pre-registered results [`results/llm/STAGE1_RESULTS.md`]

| hypothesis | verdict | detail |
|---|---|---|
| **H1** valence asymmetry | **supported in 9 of 12** | All in one direction: rivals weigh less than allies, and for both Qwens rivals mildly *attract*. Symmetric in neutral framing for Llama-70B and Nemo. |
| **H2** label fields | **supported** | Letter field 10 of 12. Content field h_C 8 of 8 content cells (corrected from "6 of 6", EXECUTION_LOG entry 6). |
| **H3** non-additivity | **2 of 12** (Llama-70B political and workplace, pairwise) | Pairwise significant in 11 of 12 but below the minimum effect in 9. Cubic: none. |
| **H4** β rises with size | **not supported** (only 1 of 4 families testable) | Qwen 9B → 122B rises in all framings (Holm p = 0.003). Llama excluded by comprehension; Gemma and Mistral by the gate. |
| **H5** sum vs average | **departs from summation in 7 of 12** | Averaging (α ≈ 1) for Llama-70B, Nemo and Qwen-9B in neutral framing; Qwen-122B sums (α ≈ 0). |

**Amendments 4–5 (self-hosted Gemma-4).**

Amendment 4: self-hosted Gemma-4 in bf16 [`results/llm/AMENDMENT4_RESULTS.md`].
- The dense 31B passes the gate locally (0.114). The 26B MoE fails (0.592), because its bf16 outputs move with batch
  composition.
- Adding the 31B gives **15 cells**. H1 then holds in 12 of 15. H3 cubic replicates for Gemma-31B in neutral and
  political framing. No original verdict changes.

Amendment 5: a rescue rerun of the 26B at batch size 1 with deterministic kernels, labelled "rescue"
[`results/llm/AMENDMENT5_RESULTS.md`].
- Replicates are identical. The extended set is **18 cells**, with H1 in 15 of 18.
- H4 rises in the Gemma family as well (β 26B → 31B: neutral 0.75 → 1.11, CI of the difference [0.17, 0.56]).
- **Overall H4 remains not supported**: 2 of 2 testable families, but the rule requires ≥ 3 of 4.

**Gate-relaxed sensitivity** (post-data, EXECUTION_LOG entry 6). Adding the three gate-failed provider endpoints
(21 cells):
- H1: 18 of 21.
- H3 cubic: 4 cells, all on jittery endpoints. Two of them were later replicated on the clean local 31B.
- H4 would be supported (3 of 3 testable families).

The confirmatory analysis was not changed.

**Provider and batch effects** (secondary) [`AMENDMENT4_RESULTS.md`, `AMENDMENT5_RESULTS.md`]:
- Provider choice shifts prompt-format biases (letter field h_L, order field h_O).
- Social coefficients agree between provider and local serving in two of three framings, and differ by 7–9% in the
  third (Io Net vs local 31B, the only clean pair).
- Batched serving of the MoE model moves single answers by an RMS of **0.86 nats** relative to batch size 1. It flips
  2.5% of majority answers.

**Temperature.** The pre-registered 1/T verification on GPT-4o-mini **failed**: χ² rejects at T = 0.5 and 1.5, while
the slopes match 1/T (EXECUTION_LOG entry 1). Reference-logit jitter explains most but not all of the misfit.

The same check on **Qwen3.5-9B passed** at both temperatures [`results/llm/temperature_verify_qwen-9b.json`; EXECUTION_LOG
entry 7]:

| T | slope [95% CI] | expected | χ² p |
|---|---|---|---|
| 0.5 | 2.22 [1.87, 2.61] | 2.0 | 0.13 |
| 1.5 | 0.68 [0.45, 0.93] | 0.67 | 0.57 |

At T = 1.5, 99 of 1,000 samples were invalid. The 1/T rule describes the choice conditional on a valid reply.

**Exploratory annex** (probes on Gemma activations; cannot alter H1–H5).
- Rival-majority information is decodable from 31B activations: deconfounded AUC up to 0.96 at layers 30–60.
- It is also used: |b_r|/b_a = 0.71.
- For the 26B, the earlier "represented but not used" reading does not survive a CI and was revised
  [`AMENDMENT5_RESULTS.md`, probe follow-ups].
- No headline rests on probes.

### 3.4 What this means for the spin analogy

Five differences from Ising spins [`STAGE1_RESULTS.md`, "How LLM agents differ from Ising spins"]:
1. **Rival ties behave closer to no tie than to J < 0.**
2. **Inertia is strong and framing-dependent:** γ = 1.1–7 in content framings. This is a self-coupling that the
   heat-bath has no term for.
3. **Several models average their neighbours** (α ≈ 1), which induces non-reciprocity on heterogeneous-degree graphs.
4. **Letter, order and content fields are present.**
5. **Contact order matters strongly.** In neutral framing, Llama-70B's logit SD across permutations of the same cell
   is 2.95 nats, a voter-like primacy effect. Copying one randomly met contact is the recruitment dynamics of
   Kirman (1993).

Item 5 turns out to matter in Stage 3.

---

## 4. Stages 2–3: surrogate predictions and live networks

### 4.1 Stage 2: surrogate predictions [`results/llm/STAGE2_RESULTS.md`]

**Surrogate.** Each fitted response function was plugged into a simulator: the A3 logit with k^(−α) scaling, inertia γ,
content field h_C, and a first-contact voter step (copying one contact, as in Kirman 1993) where it beat the logit
held-out. The simulator reproduces the Viana–Bray results
with Ising agents.

**Findings (surrogates only):**
- The naive Ising reading of each model mispredicts in both directions. For example, in neutral framing at ρ = 0.25,
  Qwen-122B orders (|m| 0.39) while its Ising reading does not (0.09).
- Rival ties often fail to frustrate.
- Political framing freezes agents by stubbornness rather than frustration.
- Directed tie perceptions (ε) change |m| by ≤ 0.06 at Stage 3 scale.

**Pre-registered predictions** were written before any live call [`llm/STAGE3_PREDICTIONS.md`, commit 878724e; pre-data
addendum de0ff08]:
- five points on random-regular d = 4 graphs, N = 100, 5 graphs × 2 replicas, 10 + 10 sweeps;
- **P1** Qwen-122B neutral ρ = 0.25; **P2** Qwen-9B political ρ = 0.5; **P3** Llama-70B political ρ = 0.25; **P4**
  Nemo workplace ρ = 0.25; **P5** GPT-6-luna neutral ρ = 0 (calibration);
- plus no-contacts arms P2-nc and P4-nc.
- **Rule:** the surrogate is supported if both primary measures fall inside its 90% interval at ≥ 3 of P1–P4.

### 4.2 Stage 3: live networks — pre-registered failure

- **Run:** 70 chains, 140,000 live updates, $4.95. Invalid replies ≤ 0.02% at every point (EXECUTION_LOG entry 26).
- **Pre-stated verdict: the surrogate is not supported.** It passed both primary measures at **0 of P1–P4**. The
  calibration point P5 and both no-contacts arms also missed.

Observed values (mean of 5 graphs) against the surrogate's 90% interval:

| point | primary measure 1 | primary measure 2 |
|---|---|---|
| P1 | \|m\| 0.21 [0.24, 0.46] | persistence 0.78 [0.66, 0.74] |
| P3 | \|m\| 0.10 [0.20, 0.37] | persistence 0.99 [0.91, 0.95] |
| P4 | m 0.94 [0.87, 0.91] | persistence 0.90 [0.75, 0.84] |
| P5 | \|m\| 0.60 [0.14, 0.24] | persistence 0.66 [0.11, 0.20] |

At P1, P2, P4 and P5 the observed values also fell outside both control intervals: the naive Ising reading and
fields-only.

### 4.3 Harness audit [`results/llm/STAGE3_AUDIT_RESULTS.md`; plan 4c3924a]

Before interpreting the failure we audited the live harness from the call cache, without rerunning anything.

**Every check on the live logs passed with zero mismatches.** The checks covered:
- update scheme;
- stale state;
- option order and letter mapping;
- own-position feed;
- tie labels;
- prompt equivalence (all 140,005 prompts);
- provider pinning;
- invalid replies;
- time alignment;
- measurement code, reproduced to 1.1e-16.

A mock surrogate pushed through the unchanged harness reproduced the simulator within 0.013 on every measure.

- **Pre-stated mock check A:** failed at P2 and P4-nc, by rule.
- **Cause, post-hoc:** the committed intervals were not conditional on the five graphs actually used. On those graphs
  the simulator itself falls inside the intervals only 80–86% of the time.
- **This does not rescue the surrogate.** Against same-graph distributions, the live misses are 3.6–20 SD at P1, P3, P4
  and P5. Only P2's miss is consistent with a correct surrogate.

**Contact order** was the main audit finding (post-hoc):
- On calls whose contact order matched the battery prompt's, live answers equal the battery's: RMS logit 0.06–0.35.
- On other orders they differ by 0.44–0.97 nats.
- The fitted surrogate had no order term.

---

## 5. Kernel test: trait summary vs order-averaged table vs live kernel

**Pre-registration:** plan and outcome map committed before any kernel was computed (fe65958; amendment f6b1922)
[`results/llm/KERNEL_TEST_RESULTS.md`].

A *kernel* here is the full table P(+1 | contact configuration c, own position s₀). We compare four:

| kernel | what it is |
|---|---|
| **A3** | the committed parametric surrogate (a "trait summary") |
| **K_fixed** | the Stage 1 battery table, one fixed contact order per cell |
| **K_live** | the empirical live kernel |
| **K_avg** | a new order-marginalised battery: ≤ 8 contact orders × 2 option orders × both letter mappings per configuration; 11,992 calls, $0.58 |

Each kernel is simulated on the live graphs and starts (R = 400), with the measurement code unchanged.

**Pre-registered verdicts (P1–P4 passing; rule ≥ 3):**

| kernel | P1–P4 passing | reproduces live? |
|---|---|---|
| K_fixed | 2 | no |
| A3 | 1 | no |
| K_live | 2 | no |
| K_avg | 2 (P2, P4) | no |

The outcome-map branch that governs is "K_live fails → the live system is not Markov in configuration; investigate
before claiming anything". This triggered the residual tests (§6).

**Per-step agreement with K_live** (live-visit-weighted):

| point | K_avg \|ΔP\| / \|Δlogit\| | K_fixed \|ΔP\| / \|Δlogit\| | A3 \|ΔP\| / \|Δlogit\| |
|---|---|---|---|
| P1 | 0.004 / 0.05 | 0.010 / 0.22 | 0.019 / 0.72 |
| P2 | 0.002 / 0.03 | 0.008 / 0.16 | 0.010 / 0.19 |
| P3 | 0.001 / 0.03 | 0.002 / 0.07 | 0.016 / 0.67 |
| P4 | 0.003 / 0.07 | 0.008 / 0.36 | 0.013 / 0.66 |
| P5 | 0.013 / 0.65 | 0.037 / 0.65 | 0.123 / 1.63 |

**Amplification of per-step differences** (collective difference in SD units of the reference spread):
- **A3 vs live:** 3–20 SD. At P5 the A3 fit predicts disorder; both empirical kernels predict ordering.
- **K_fixed vs K_live:** at most 3.4 SD (P3 persistence), from a per-step |ΔP| of **0.002**.
- **K_avg vs K_live:** every primary-measure mean at P1–P4 is within 0.9 SD of the other's. K_avg predicts the
  population as well as the live kernel.

**Reading** (governed by the map, so no headline was drawn at the time):
1. Most of the Stage 3 miss is the parametric trait summary. The single-agent data held the information; the fitted
   functional form did not carry it.
2. An order-averaged one-shot battery recovers the live per-step kernel.
3. Per-step-invisible differences can still matter collectively.

**Free follow-up checks found no non-Markov mechanism in the logs** (post-hoc):
- provider caching;
- wall-clock drift;
- history dependence;
- lag-1 correlation;
- wider neighbourhood;
- concurrency. One small effect, at one provider (Io Net), cannot explain the pattern.

---

## 6. Residual persistence (footnote-level result)

The live runs sat slightly on the high-persistence side of every Markov kernel: Stouffer z ≈ +3.1 over 7 points
against K_live. The excess was 0.01–0.04 in persistence units. We pre-registered two tests (plan 9c759f7; amendment
b27d83a) [`results/llm/RESIDUAL_TEST_RESULTS.md`].

**Test 1: heterogeneity (mover-stayer).**
- No agent-specific content is sent: all 140,005 prompts are deterministic renders of the configuration.
- The pre-stated permutation test of between-agent variance was significant (Fisher p 0.008), **contrary to the
  prediction**.
- A post-hoc within-cell shuffle traces most of that to agents clustering in configurations, leaving only P4-nc.
- A heterogeneous kernel does **not** explain the excess: Stouffer 2.84 vs 2.72 homogeneous; the pre-stated criterion
  is not met.

**Identical-request check (free, before any paid call).** The pre-stated agreement expectations were biased by
selection on the earlier choice. Selection-corrected (post-hoc, labelled), there is no dependence beyond draw luck at
two logprob points.

**Test 2: fresh live replicate**, one call in flight per endpoint, with drift sentinels.
- Cost: API $5.29, pods ≈ $0.9. Six interruptions, all logged and resumed from cache.
- **Deviation disclosed:** the remote pods built random-regular graphs with networkx 3.3 instead of 3.7, so 69 of 70
  replicate chains ran on *different* graphs from the same ensemble.
  - The pre-registered analysis (Z = −0.48) used the wrong graphs.
  - A mechanical reanalysis on the actual graphs, committed before it ran, is the reported result. Its reading does
    not depend on the deviation.
  - Prevention: library pinning and a graph-fingerprint assertion, now a project rule in `CLAUDE.md`.
- **Pre-registered result: Stouffer Z = −0.42 (p = 0.66); the excess does not repeat.** Both original point misses
  moved, in different directions.
- Sentinels show hours-scale drift at some providers, but it does not track the collective measures.
- A replay audit of all 70 replicate chains passed every check (plan f06db97).

**Interpretation (pre-registered map plus asymmetry rule):** "realisation noise **or** our own concurrency; not
separable".

Combined effect, raw persistence units [`results/llm/stage3_rep1/persistence_effect_combined.json`]:

| run | effect [95% CI] |
|---|---|
| original | +0.0116 [+0.0033, +0.0200] |
| replicate | −0.0051 [−0.0135, +0.0033] |
| pooled | +0.0033 [−0.0026, +0.0093] |

The runs are heterogeneous (Cochran Q = 7.6, p = 0.006). **We treat the persistence excess as unestablished.**

The narrowed claim stands: trait summaries fail, while order-averaged conditional tables work per step. Collective
misses at about 2 of 4 points remain, in both directions. That is consistent with small per-step errors being
amplified, which is the subject of §7.

---

## 7. Amplification near tipping points

### 7.1 Design

Stage 3 ran at the model's natural sampling setting (T = 1), so we needed an axis along which a population can be
moved towards a transition.

**Decision-noise rescaling.** Each per-prompt letter logit of a kernel is divided by T before averaging over
presentations. Then:
- the transition is located by finite-size Binder crossings at N = 100, 400 and 1,600 on random-regular d = 4 graphs;
- controlled per-step errors are added at δ = 0.1 nats per step. The shapes are a uniform stay-bias, a majority shift,
  random per-configuration noise, and real error shapes (K_fixed − K_avg; A3 − K_avg);
- **amplification** A = |Δ⟨|m|⟩| per nat of visit-weighted per-step error, from paired runs with common random
  numbers.

### 7.2 Stage B: the pre-registered test, refuted

- **Plan:** 18f619e, before any run. Amendment 1 (eef27e6) changed how T_c is located, before any perturbation run
  [`results/llm/AMPLIFICATION_RESULTS.md`].
- **T_c for P1** (Qwen-122B, K_avg kernel) = 1.93 [1.92, 1.96]. The Stage 3 temperature T = 1 is at T/T_c = 0.52, deep
  in the ordered phase.
- **H1 (N-growth at T_c): holds for all three shapes** (G = A(1600)/A(100)):
  - stay-bias 3.0 [2.3, 4.3];
  - majority-shift 3.4 [2.8, 4.2];
  - noise 5.5 [4.7, 6.7].
- **H2 (growth exceeds the controls' growth): fails** for majority-shift against P2 (ratio 0.55 [0.05, 3.8]).
- **Pre-registered verdict: refuted.**
- **Cause.** P2's majority-shift amplification at T_c was 0.002 (N = 100) and 0.010 (N = 1,600). Its own growth ratio,
  6.2 [0.9, 68], was therefore uninformative. The comparison of growth *ratios* is ill-defined when a control's
  amplification is near zero.

Two pre-stated expectations also failed:
- (ii) "bounded away from T_c". Symmetry-breaking noise keeps growth ≈ 4–6 above T_c, a finite-size effect of |m|.
  Growth reaches 12–22 near a low-T freezing edge.
- (iii) "controls bounded", again from near-zero denominators.

**Real eval errors placed at T = 1** (Stage 3's setting):

| error shape | per-step error | Δ\|m\| (N = 100, equilibrium) |
|---|---|---|
| fixed-order battery (K_fixed) | 0.48 nats | +0.004 |
| trait summary (A3) | 1.71 nats | −0.034 |

The same K_fixed error shape compared at the **same N**, at T = 1 and at T_c [`results/llm/amplification/amplification.csv`,
equilibrium protocol, rows `fixed`, λ = 1]:

| N | Δ\|m\| at T = 1 | Δ\|m\| at T_c | per nat at T = 1 [95% CI] | per nat at T_c [95% CI] |
|---|---|---|---|---|
| 100 | +0.004 | +0.068 | 0.009 [0.006, 0.011] | 0.50 [0.43, 0.57] |
| 1,600 | −0.002 | +0.169 | 0.004 [0.000, 0.017] | 1.27 [1.20, 1.34] |

- **At N = 100:** the effect per run is about 16× larger at T_c, and about 56× per nat. The per-step error is smaller at
  T_c because the logits are divided by T.
- **At N = 1,600:** the T = 1 effect is indistinguishable from zero. The per-nat ratio is at least about 75×, using the
  upper CI bound at T = 1.

An earlier draft said "about 40×". That figure compared +0.17 at N = 1,600 with +0.004 at N = 100, so it mixed the
temperature effect with N-dependence.

Figures: `results/llm/amplification/fig1_binder.png`, `fig2_amplification_abs_m.png`, `fig3_growth_abs_m.png`,
`fig4_placement_abs_m.png`.

### 7.3 Corrected confirmatory test on fresh data

The corrected test was pre-registered with its model-selection rules and costs before any run or API call
[`llm/AMPLIFICATION_CONFIRM_PLAN.md`; 76aad78, amendments 32daeac, 0b69998, ef5674d]. Fresh graphs (N = 100 included)
and fresh seeds were used; no Stage B run feeds a confirmatory number.

**Rules:**
- **A (decisive): absolute amplification beats the worst-case control.** D = A_model − max(A_P2, A_P4) at (T_c,
  N = 1,600). Pass iff D ≥ 0.10 |m| per nat and the 95% CI of D excludes 0.
- **B: N-growth,** tested only where A ≥ 0.10 at both N = 100 and N = 1,600, which removes the near-zero-denominator
  problem.
- **C: peak within ±10% of T_c.**

**Models** (selection rules fixed before any probe; record in [`results/llm/AMPLIFICATION_CONFIRM_STOP2.md`]):

| model | role | how chosen / served |
|---|---|---|
| P1 Qwen-122B | confirmatory | re-located from scratch |
| Gemma-4 31B | confirmatory | self-hosted, batch size 1, deterministic; replicates bit-identical |
| Gemini 3.1 Flash-Lite | confirmatory | sampled at 50 per prompt |
| GPT-4o-mini | confirmatory | first of 28 frozen OpenAI/Meta/DeepSeek candidates to pass a probe for logprobs, temperature honoured, repeat SD ≤ 0.3 and leak < 5% |
| GPT-6-luna | exploratory | its endpoint refuses `temperature` |

All five pass comprehension (99–100% per question type).

**1/T check, pre-registered** [`results/llm/confirm/api_summary.json`, `temp_diag.json`]:
- **Qwen-122B, Nemo-12B (control P4) and GPT-4o-mini fail.** The misfit is at T = 0.5 and goes beyond reference noise.
  At T = 1.5 all three fit within noise, though Qwen-122B narrowly misses on slope (CI 0.68–1.08 vs 0.67).
- **Gemini passes:** κ = 1.13 [0.96, 1.31].

**Transitions** [`results/llm/confirm/sim/locate_*.json`; `results/llm/confirm/fig_binder_step3.png`]:
- **P1:** T_c = 2.01 [1.96, 2.07]. Independent of Stage B's 1.93.
- **Gemini:** T_c = 1.83 [1.81, 1.86].
- **GPT-6-luna (exploratory):** 0.49, on the unverified sharpening side.
- **GPT-4o-mini and Gemma-31B: no transition in the swept range** (§8).

**Verdicts (Step 4)** [`results/llm/AMPLIFICATION_CONFIRM_RESULTS.md`; `results/llm/confirm/results/step4.json`]:

| model | shape | A(model) | max control | D [95% CI] | A | B: G [95% CI] | C |
|---|---|---|---|---|---|---|---|
| P1 | stay-bias | 0.31 | 0.16 | 0.15 [0.09, 0.22] | **pass** | 1.36 [0.96, 2.08], fail | pass |
| P1 | majority | 1.10 | 0.31 | 0.79 [0.68, 0.90] | **pass** | 1.74 [1.44, 2.11], pass | pass |
| P1 | noise | 2.18 | 0.15 | 2.03 [1.96, 2.10] | **pass** | 11.7 [9.2, 14.4], pass | pass |
| Gemini | stay-bias | 1.35 | 0.15 | 1.20 [1.12, 1.28] | **pass** | 3.43 [3.02, 3.94], pass | pass |
| Gemini | majority | 1.30 | 0.27 | 1.03 [0.94, 1.13] | **pass** | 3.75 [3.15, 4.61], pass | pass |
| Gemini | noise | 1.33 | 0.16 | 1.17 [1.10, 1.25] | **pass** | 9.5 [7.8, 12.1], pass | pass |

**Per model:**
- **P1:** amplification at the tipping point exceeds the controls for **3 of 3** shapes. B passes 2 of 3; stay-bias
  misses, against the plan's prediction that all would pass. C passes 3 of 3.
- **Gemini:** **3 of 3** on A, B and C.
- **Close call:** P1 stay-bias passes A on its point estimate, but its CI lower bound (0.09) is below the 0.10 margin.
- **Real fixed-order battery error at P1's T_c:** A = 0.62, growing with N (G 1.73 [1.43, 2.12]).
- **Persistence (secondary):** A passes for P1 in 2 of 3 shapes and for Gemini in 3 of 3; growth is mostly not
  significant.

Figures: `results/llm/confirm/results/fig_step4_amplification.png`, `fig_ruleA.png`.

**Negative control: GPT-4o-mini**, a real model with no transition, pre-registered in amendment 3.
- Over 26 temperatures spanning both confirmatory bands, its |m| amplification never reaches the 0.10 floor: max 0.075,
  0.089 and 0.096 by shape. Its prediction is therefore reported as **"below floor everywhere"**.
- Both test models exceed it at their T_c, with every CI excluding 0.
- In persistence it does cross the floor, but never with upward N-growth. A few points shrink with N, which violates
  the literal prediction in the direction opposite to growth.

**Step 5: the low-T edge** (labelled a defined decision-noise rescaling, not a verified sampling temperature).
- At 4× run length no shape shows N-growth.
- **Verdict: finite-time effect.** The Stage B growth of 12–22 near T ≈ 0.6–0.8 was a finite-run-length artefact, not
  a second transition.

**Exploratory results** (labelled; not confirmatory):
- **GPT-6-luna:** passes A for 3 of 3 shapes, but its N = 1,600 amplification curve is flat with no clear peak, and its
  T axis is unverified.
- **Gemma-31B:** see §8.

---

## 8. Model-dependence of tipping points

Of the four confirmatory models, **two** (Qwen-122B, Gemini) have an order–disorder transition in the swept
decision-noise range. **Two** do not [`results/llm/AMPLIFICATION_CONFIRM_STOP2.md`]:

- **GPT-4o-mini never orders.** The highest Binder cumulant at N = 1,600 is 0.47, at T = 0.2 (the rule needs ≥ 0.5),
  and |m| ≤ 0.22. Hot and cold starts agree everywhere.
- **Gemma-31B is frozen and start-dependent.** Hot and cold starts disagree at every T ≤ 3.16, so the rule marks that
  region as frozen.
  - **The mechanism is not identified.**
    - Stage 1 does not settle it. In the neutral framing simulated here, Gemma-31B's rivals repel at 71% of ally
      weight (b_r 0.93 [0.79, 1.11] vs b_a 1.30). With 25% rival ties that is compatible with frustration.
    - Its inertia is small and not significant: γ = 0.08 [−0.15, 0.32] (`results/llm/stage1_cells_amendment4.csv`,
      `gemma-31b-local`, `neutral_own`).
    - **Exploratory mechanism test** [`results/llm/GEMMA_MECHANISM_RESULTS.md`; plan committed first]:
      - Removing rivals' influence makes the population order strongly (|m| 0.77). Replacing the table by its
        additive projection makes it disordered (|m| 0.09). Either way start-dependence disappears, so under the
        pre-stated reading no single mechanism is identified.
      - Replica overlaps show no frozen microscopic state.
      - The regime reads as slow, start-dependent relaxation near an ordering boundary, which depends jointly on rival
        repulsion and the non-additive response.
      - Inertia was not knocked out.
  - Cold starts decay from all-up to |m| ≈ 0.44; hot starts stay near 0.05.
  - The result is unchanged with exact full-vocabulary P(A).
  - **Exploratory run-length check:** start-dependence persists at T ≈ 2.0–2.3 even at 4× run length, and dissolves
    above T ≈ 2.7 [`results/llm/confirm/results/gemma_exploratory.json`].
  - **Exploratory outcome sensitivity:** a 0.1-nat per-step error changes which collective basin a population reaches
    in 24–51% of paired runs from identical starts. A different random seed alone changes it in 43–60%. In this regime
    the outcome is close to a coin flip set by the stochastic path.

**Why this matters.** Both "no transition" models pass comprehension, and both have well-defined single-agent response
tables. Nothing in their per-agent statistics announces whether a population of them will have a tipping point.
In this study, **whether a tipping point exists, where it lies, and how strongly it amplifies errors were all
knowable only by simulating the population** with the measured kernel.

The Stage 3 setting (T = 1) sits far from P1's tipping point (T/T_c ≈ 0.5). That is consistent with the small
collective effect of the real evaluation errors there (+0.004 in |m|).

---

## 9. Limitations and scope

- **Graphs:** random-regular degree 4 for every live and amplification result. ER, heterogeneous-degree and
  real-network topologies were explored only in Stage 2 surrogates.
- **Task:** a binary choice with explicit ally/rival labels and current positions. There is no free text,
  persuasion, deliberation or memory.
- **Agents are memoryless:** every update is one call built from the current state only. The kernel tests found no
  per-step history dependence (|ΔP(stay)| ≤ 0.002), but agents with memory were not tested.
- **Models:** 9 open-weight endpoints tested in Stage 1 (4 confirmatory, 6 with the self-hosted Gemma rescue analyses), plus frontier spot checks; 5 live endpoints in Stage 3, and 4 confirmatory plus 1
  exploratory model in the amplification test. All were pinned to specific providers, and provider choice shifts
  format biases (§3.3).
- **Decision-noise axis:** the T axis divides the measured logits by T. It is a defined manipulation of decision
  noise, not a realistic deployment setting.
  - The 1/T rule fits within noise only on the noise side (T > 1). On the sharpening side (T = 0.5) it fails beyond
    reference noise for every logprob model checked.
  - GPT-6-luna's endpoint does not accept temperature at all.
  - Invalid replies reach 15% at T = 2 for Gemma.
  - Results at T < 1 (Step 5, luna, GPT-4o-mini's low-T range) are labelled accordingly.
- **Simulation-based amplification:** the amplification results come from empirical kernels, not live populations
  pushed towards T_c. Live networks were run only at T = 1. That the kernels describe live per-step behaviour rests on
  the kernel test (|ΔP| ≤ 0.004 at P1–P4), done at T = 1 only.
- **Selection:** GPT-4o-mini's selection is borderline. Its dated alias, the same weights on the same provider, failed
  the temperature probe. It is also the pre-stated fallback.
- **Residual persistence:** the small excess over Markov kernels was not established (§6). The two runs disagree, and
  our own concurrency cannot be separated from realisation noise.
- **Statistical power:** Stage 3 used N = 100 and 5 graphs per point. Live collective misses at about 2 of 4 points
  remain unexplained beyond the amplification argument.

---

## 10. Implications for evaluation and governance

These statements stay within the claims above.

Framed in Brock and Durlauf's (2001) terms, each model supplies a social logit choice rule. The collective
equilibria, including whether multiple equilibria (a tipping point) exist at all, are a property of that rule on the
network, not of any single response.

1. **Per-agent trait summaries are not enough to predict what a population of agents will do.** In this study a fitted
   trait model of each agent mispredicted live populations by 3–20 SD. A table conditioned on the full input
   configuration, averaged over the presentation details the model is sensitive to, did much better.
2. **Small per-step differences between evaluation methods can produce large collective differences,** especially near
   a tipping point.
   - At P1's tipping point, a 0.1-nat per-step error moved |m| by 0.15–2.0 per nat more than in the controls.
   - The real fixed-order battery error grew with population size there.
   - The same errors had small effects at the operating point actually used (T = 1).
3. **Whether a population is near a tipping point is a population-level property.** It differed across models that
   all passed the same single-agent checks. Evaluating multi-agent deployments therefore needs population-level
   simulation or measurement with the measured response tables, not only per-agent scores.
4. **Presentation details matter:** contact order, option order, letter mapping, batching and provider. An evaluation
   that fixes them measures one slice of the response distribution. Averaging over them recovered live behaviour per
   step.

We make no claim about other tasks, richer interaction formats, larger or heterogeneous networks, other models, or real
deployments.

---

## Appendix A. Pre-registration and amendment log

| document | commit(s) | content |
|---|---|---|
| `llm/PREREG.md` | e9c7e47 | Stage 1 battery, H1–H5 (before any battery data) |
| Amendment 1 | 8c0ab35 | H3 test replaced (score test, wild bootstrap); estimand made explicit (pre-data, synthetic tests only) |
| Amendment 2 | 9bf6c28 | answer instruction v2 after pilot leak (before main run) |
| Amendment 3 | 483decc | comprehension gate, analytic temperature, Stage 3 option-order randomisation (before main run) |
| Amendment 4 | ee18890 | self-hosted bf16 Gemma-4 (before any GPU run) |
| Amendment 5 | 7705d12 | rescue rerun of Gemma-26B at batch size 1 (before run) |
| `llm/STAGE3_PREDICTIONS.md` | 878724e; addendum de0ff08 | Stage 3 predictions (before any live call) |
| `llm/STAGE3_AUDIT_PLAN.md` | 4c3924a | harness audit (before any audit result) |
| `llm/KERNEL_TEST_PLAN.md` | fe65958; amendment f6b1922 | three-kernel test and outcome map; K_avg options |
| `llm/RESIDUAL_TEST_PLAN.md` | 9c759f7; b27d83a; 93bd950 | heterogeneity test, replicate design, sentinels, nc predictions |
| `llm/REPLAY_AUDIT_PLAN.md` | f06db97 | replicate replay audit and combined effect size |
| `llm/AMPLIFICATION_PLAN.md` | 18f619e; amendment eef27e6 | Stage B amplification curve; T_c relocation |
| `llm/AMPLIFICATION_CONFIRM_PLAN.md` | 76aad78; 32daeac; 0b69998; ef5674d | corrected confirmatory test; model selection; guardrail re-pin; negative control and labels |

The full chronological record is `llm/EXECUTION_LOG.md` (entries 1–67).

## Appendix B. Deviations

| # | where | deviation | when decided | effect | record |
|---|---|---|---|---|---|
| 1 | Stage 1 | Answer instruction changed after the pilot (leak 24–94%) | before the main run | pilot data not analysed | Amendment 2 |
| 2 | Stage 1 | Noise gate first applied per arm (bug) | before any confirmatory output | none | log 2 |
| 3 | Stage 1 | Gemma backups failed (CoreWeave 0.354; Novita no logprobs on 77%) | as pre-listed | Gemma excluded from confirmatory set and H4 | log 3 |
| 4 | Amendment 4 | Padding-free batches instead of left padding | before the GPU run | none on the estimand | `AMENDMENT4_RESULTS.md` |
| 5 | Amendment 4 | Comprehension summary overwritten by a test, then restored byte-identical | before the analysis | none | log 33 |
| 6 | Amendment 5 | GPU caps raised (2.5 h / $6 → 3.5 h / $6.50) after measuring throughput | before the main run | cost | `AMENDMENT5_RESULTS.md` |
| 7 | Kernel test | K_avg call count 11,992, not ≈ 7,500; two analysis bugs fixed before reading | before reading | none | `KERNEL_TEST_RESULTS.md` |
| 8 | Residual | Identical-request expectation biased by selection; corrected test (iii) added | post hoc, labelled | reading changed to "no dependence" | `RESIDUAL_TEST_RESULTS.md` |
| 9 | Residual | Heterogeneity robustness (within-cell shuffle) | post hoc, labelled | signal mostly configuration clustering | same |
| 10 | Replicate | Pods used networkx 3.3 graphs (69/70 chains); mechanical reanalysis on actual graphs | after the run, before reporting | verdict unchanged (Z −0.48 → −0.42) | log 51 |
| 11 | Replicate | Six interruptions, resumed from cache; replay audit PASS | during the run | none | log 43–50, 53 |
| 12 | Stage B | T_c location rule replaced (spurious low-T crossing; under-equilibrated N = 1,600) | before any perturbation run | T_c 0.36 → 1.93 | amendment 1, log 55 |
| 13 | Stage B | Locate units also ran perturbations (bug); 4 files deleted unread | before any reading | none | log 55 |
| 14 | Confirm | Guardrail blocks first-party `deepseek` provider; pin rule re-applied over reachable providers | before any data from those providers | candidates 1 and 3 re-pinned; both then failed the gate | amendment 2 |
| 15 | Confirm | Gemma fingerprint gate coded without the /√2 of the PREREG definition; corrected after it fired (owner decision) | after the gate fired | relaunch; gate 0.234 passes | log 62–63 |

## Appendix C. Costs

Each figure is taken from the phase's results document. The total is their sum and has not been reconciled against
account billing.

| phase | item | cost | source |
|---|---|---|---|
| Stage 1 | API (battery, spot checks, bridge) | $9.56 | `STAGE1_RESULTS.md` |
| Stage 3 | live networks, 140,000 updates | $4.95 | log 26 |
| Amendments 4–5 | GPU (A100), including failed launches | ≈ $11.90 | `AMENDMENT5_RESULTS.md` |
| Kernel test | K_avg battery | $0.58 | `KERNEL_TEST_RESULTS.md` |
| Residual | replicate API | $5.29 | `RESIDUAL_TEST_RESULTS.md` |
| Residual | replicate CPU pods | ≈ $0.9 | same |
| Confirmatory | API (probes, checks, comprehension, batteries) | $3.33 | `AMPLIFICATION_CONFIRM_STOP2.md` |
| Confirmatory | GPU (two A100 pods) | ≈ $0.91 | log 64 |
| Stage 2, kernel sims, amplification | CPU in session containers | $0 | — |
| **Total** | | **≈ $37.4** | sum of the above |

## Appendix D. Data and code availability

- **Code:** this repository; every analysis script is named next to its result.
- **Results:** committed under `results/` and `results/llm/`.
- **Raw API cache:** every request body and compact reply, keyed by SHA-256 of the request plus replicate index, with
  no credentials or account-linked IDs.
  - Hosted in the **private** Hugging Face dataset `RandallSPQR/social-phase-transitions-llm-cache`.
  - Listed with per-file SHA-256, size and record count in `results/llm/cache_manifest.json`.
  - `llm/fetch_cache.py` restores and verifies it. Reruns from the cache are free and exact; the replay audits show
    this.
- **Pod outputs:** GPU and CPU pod outputs and logs are in the same dataset under `runs/<pod id>/`.
- **Graphs:** Stage 3 graph fingerprints are in `results/llm/stage3/graph_hashes.json`, the replicate's actual graphs
  in `results/llm/stage3_rep1/graphs/`, and fingerprints of the confirmatory phase's fresh graphs in
  `results/llm/confirm/sim/locate_*.json`.

## Appendix E. Sourcing notes (items to verify before submission)

1. **Claims list.** The five claims in the Abstract were confirmed by the owner (2026-10-15).
2. **External literature.** Full entries are in the References section. They were written from knowledge, not checked
   against the publications in this session, so verify volume and page numbers before submission. The five September
   2026 proofs are identified only by their PDF titles in `papers/`.
3. **Cost total.** The total is a sum of per-phase figures. GPU figures are estimated as rate × uptime.
4. **Cache manifest.** It was regenerated on 2026-10-14 to include this phase's API calls. The upload is verified by
   re-download; see the commit that accompanies this report.
5. **Stage 2 claims** are quoted from `STAGE2_RESULTS.md` without re-running anything.
6. **Probe findings** are exploratory and their revisions are quoted. No probe result supports a claim here.
7. **Figures** are reused from the committed figure files listed in each section. The Substack set in
   `figures/substack/` (simulated from the measured tables; see its `CAPTIONS.md`) was made afterwards and is not cited
   in the analysis.

## References

- Brock, W. A., & Durlauf, S. N. (2001). Discrete choice with social interactions. *Review of Economic Studies*, 68(2),
  235–260.
- Crisanti, A., & Sompolinsky, H. (1987). Dynamics of spin systems with randomly asymmetric bonds: Langevin dynamics
  and a spherical model. *Physical Review A*, 36(10), 4922–4939.
- Kirman, A. (1993). Ants, rationality, and recruitment. *Quarterly Journal of Economics*, 108(1), 137–156.
- Kline, P., & Santos, A. (2012). A score based approach to wild bootstrap inference. *Journal of Econometric Methods*,
  1(1), 23–41.
- Manski, C. F. (1993). Identification of endogenous social effects: The reflection problem. *Review of Economic
  Studies*, 60(3), 531–542.
- Viana, L., & Bray, A. J. (1985). Phase diagrams for dilute spin glasses. *Journal of Physics C: Solid State Physics*,
  18(15), 3037–3051.
- Five unrefereed proofs released in September 2026 (motivation only), PDFs in `papers/`: Rokhlin's multiple mixing
  problem for one transformation; the Mézard–Parisi formula for diluted spin glasses; no percolation at criticality on
  quasi-transitive graphs; critical bond and site percolation on the cubic lattice; nonuniqueness of percolation on
  nonamenable quasi-transitive graphs.

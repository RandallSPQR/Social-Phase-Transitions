---
title: "Do LLM agents behave like spins? Response functions, live networks and amplification near tipping points"
author: "[Author — to be completed]"
date: "Draft, 2026-10-15"
bibliography: refs.bib
link-citations: true
# Typesetting: pandoc paper.md --filter pandoc-crossref --citeproc -o paper.tex (or -o paper.pdf); the
# @fig:/@tbl: cross-references need pandoc-crossref. Citations use pandoc [@key] syntax;
# every key resolves in refs.bib. Figure paths are relative to the repository root (run pandoc from there, or copy
# the files listed in the "Figure sources" note at the end).
---

<!--
DRAFTING NOTES (remove before submission)
- Source: results/llm/REPORT.md (commit 5b046b4). Every number below is taken from that report, which cites the
  committed result file for each one; the file is repeated here in a "Source:" line under each table/figure, or in
  square brackets in the text, so it can be checked. Strip the [`path`] tags at typesetting or move them to a
  supplementary "number provenance" table.
- Placeholders: §1 (bullets only), §2, §5, §6, §8. Full drafts: §3, §4, §7, appendices.
- Abstract is a draft for the author to rewrite.
- All references in refs.bib are marked UNVERIFIED and must be checked by the author.
-->

# Abstract {-}

*[Draft — to be rewritten by the author.]*

We ask whether large language model (LLM) agents placed on a signed social network behave like the binary spins of
statistical physics, and what this implies for evaluating them one at a time. Using a pre-registered battery, we
measure each model's response function (the probability of choosing a position given the positions and tie signs of
its contacts) and then compare predictions built from those measurements with live networks of 100 LLM agents. LLM
response functions are not Ising-like: in 9 of 12 confirmatory cells rivals carry less weight than allies, agents
show strong framing-dependent inertia, and they respond to label, option-order and contact-order cues. A
surrogate built from fitted parametric response functions passed 0 of 4 pre-registered live points, while an
order-marginalised conditional response table matched the live per-step kernel to $|\Delta P| \le 0.004$. Small
per-step differences were amplified into collective differences of several standard deviations. Rescaling decision
noise to locate the order–disorder transition of the measured kernels, a corrected pre-registered test found that
amplification of controlled per-step errors at the transition exceeds that of control kernels for 3 of 3 synthetic
error shapes in both confirmatory models with a transition; the original pre-registered version of this test was
refuted. Whether a population has a transition at all is model-dependent (two models yes, one no, one unresolved) and
was not readable from single-agent statistics. All claims are limited to the models, binary task, memoryless agents
and random-regular degree-4 graphs tested.

# 1 Introduction {#sec:intro}

*[PLACEHOLDER — prose to be written by the author. The section must establish:]*

- Multi-agent deployments of LLMs are evaluated mostly one agent at a time, on fixed prompts; the question is whether
  such evaluations predict what a *population* of interacting agents does.
- Statistical physics gives a precise baseline for binary agents on a signed network (consensus, disorder, frozen
  camps; Viana–Bray), with known transitions where small per-step changes can have large collective effects.
- The agent update rule is a logit discrete-choice model with a social term [@brock2001discrete]; the reflection
  problem [@manski1993identification] is sidestepped by measuring the response function directly with contacts'
  positions under experimental control.
- Motivation came from five unrefereed proofs released in September 2026 (cite as preprints; motivation only;
  nothing depends on their correctness).
- Research questions: (i) are LLM response functions Ising-like? (ii) do single-agent measurements predict live
  networks, and which summary of them does? (iii) are per-step errors amplified near tipping points, beyond
  controls? (iv) is the existence of a tipping point a model property readable from single-agent statistics?
- Approach: pre-registered stages, each plan committed before data; failures and refutations reported in the main
  text.
- Contributions (to be phrased): the five claims of the abstract, with scope statements.
- Scope statement: these models, this binary task, memoryless agents, random-regular degree-4 graphs.
- Roadmap of the paper.

# 2 Related work {#sec:related}

*[PLACEHOLDER — to be written by the author from `paper/related_work_notes.md`. Planned coverage: discrete choice
with social interactions [@brock2001discrete; @manski1993identification]; herding and recruitment
[@kirman1993ants]; paired-comparison and choice models [@bradley1952rank; @luce1959individual]; belief models
[@smets1994transferable]; dilute and asymmetric spin glasses [@viana1985phase; @crisanti1987dynamics]; social
influence network models [@friedkin1990social]; LLMs as simulated respondents and agents [@argyle2023out;
@park2023generative]; motivating proofs [@openai2026rokhlin; @openai2026mezardparisi; @openai2026nopercolation;
@openai2026cubic; @openai2026nonuniqueness].]*

# 3 Model and methods {#sec:methods}

## 3.1 Baseline: dilute signed spin model {#sec:baseline}

The baseline is the Viana–Bray ±J Ising model [@viana1985phase] on Erdős–Rényi graphs with mean degree $c$, in which
a fraction $\rho$ of ties are negative ($J_{ij}=-1$) and the rest positive ($J_{ij}=+1$). Each agent holds a position
$s_i \in \{+1,-1\}$ and is updated by heat-bath dynamics at noise level $T$:
$$
P(s_i = +1 \mid \text{neighbours}) = \frac{1}{1 + \exp\!\big(-2 h_i / T\big)}, \qquad h_i = \sum_{j \in \partial i} J_{ij} s_j .
$$
The cavity method gives two analytic transition lines: a consensus (ferromagnetic) transition at
$c(1-2\rho)\tanh(1/T) = 1$ and a spin-glass transition at $c\tanh^2(1/T) = 1$. Non-reciprocity is introduced through
a parameter $\varepsilon$, the probability that the reverse direction of a tie is redrawn independently, so that
$J_{ij} \neq J_{ji}$ for a fraction of ties.

We use this model for two purposes only: to validate the simulator against known results (§4.1), and as the "Ising
reading" against which LLM response functions are compared. Validation used parallel tempering, finite-size
crossings and 1,000 graph bootstraps.

## 3.2 Task and response function {#sec:task}

Each LLM agent holds position A or B and is shown $k = 1$–$6$ contacts. Each contact is labelled *ally* or *rival*
and shown with that contact's current position. The agent is asked to choose A or B. Three framings were used:
*neutral*, *political* and *workplace*. Across the battery, the letter mapping (which position is called A) and the
option order (which option is listed first) were swapped.

The response function is $P(A)$ as a function of the configuration of contacts and the agent's own position. For
endpoints that return log-probabilities, $P(A)$ is read from the first-token top-5 log-probabilities, renormalised
over the two answer letters. For endpoints that do not (frontier spot checks and some later models), $P(A)$ is
estimated from sampled replies (30 or 50 per prompt).

## 3.3 Stage 1 battery, gates and hypotheses {#sec:stage1-methods}

**Validity gates.** An endpoint entered the confirmatory set only if it passed two gates:

1. *Comprehension:* a battery of 2,116 factual questions about the agent's own position and the ally and rival
   majorities, with at least 90% correct per question type.
2. *Endpoint noise:* the repeat standard deviation on the logit scale must not exceed 0.3 nats. For logits
   $y$ (clipped to $\pm 6$) from two identical requests, $\mathrm{SD}_{\text{repeat}} = \mathrm{RMS}|y_1 - y_2| / \sqrt{2}$.

**Estimand.** The order-averaged (marginal) response function, fitted by cross-entropy (Amendment 1.2 to the
pre-registration). The fitted parametric form ("A3") is a logit in the ally and rival inputs with a $k^{-\alpha}$
degree scaling, a self-coupling (inertia) $\gamma$ and a content field $h_C$ in content framings; letter and
option-order fields are estimated alongside to test H2.

**Hypotheses.** Five confirmatory hypotheses were pre-registered, each Holm-corrected across cells:

- **H1** valence asymmetry: $w_{\text{rival}} \neq w_{\text{ally}}$;
- **H2** label fields (letter, order and content fields);
- **H3** non-additive responses: pairwise and cubic interaction blocks, tested by a score test with wild bootstrap
  [@kline2012score] (Amendment 1) and a pre-registered minimum effect size;
- **H4** the coupling $\beta$ rises with model size, supported if this holds in at least 3 of 4 model families in
  neutral framing;
- **H5** summation versus averaging over contacts, via the degree exponent $\alpha$ ($\alpha \approx 0$ sums,
  $\alpha \approx 1$ averages).

**Temperature.** A pre-registered check tested whether sampling temperature $T$ rescales the logit by $1/T$, by
comparing the measured slope of the logit at $T=0.5$ and $T=1.5$ against the $T=1$ reference, with a $\chi^2$
goodness-of-fit test.

**Pre-registration.** The plan (`llm/PREREG.md`) was committed before any battery data (commit e9c7e47), and
Amendments 1–3 before any main-battery data (Appendix A). Amendments 4 and 5 added a self-hosted Gemma-4 31B and a
"rescue" rerun of Gemma-4 26B at batch size 1 with deterministic kernels; both were committed before the GPU runs.

## 3.4 Stage 2 surrogates and Stage 3 live networks {#sec:stage23-methods}

**Surrogate.** Each fitted A3 response function was used as the update rule of a simulator. Where it beat the logit
on held-out data, a first-contact voter step (copying one contact, the recruitment dynamics of @kirman1993ants) was
added. With Ising agents the simulator reproduces the Viana–Bray results (§4.1).

**Live networks.** Live networks of LLM agents were run on random-regular graphs of degree $d = 4$ with $N = 100$
agents, 5 graphs × 2 replicas per point, 10 burn-in and 10 measured sweeps, random-sequential updates. Each update is
one model call built from the current state only (memoryless agents). Five points were pre-registered:

| point | model | framing | $\rho$ |
|---|---|---|---|
| P1 | Qwen3.5 122B-A10B | neutral | 0.25 |
| P2 | Qwen3.5 9B | political | 0.5 |
| P3 | Llama-3.3 70B | political | 0.25 |
| P4 | Mistral Nemo 12B | workplace | 0.25 |
| P5 | GPT-6-luna | neutral | 0 (calibration) |

Table: Pre-registered Stage 3 points. Two no-contacts arms (P2-nc, P4-nc) were also run. Source:
`llm/STAGE3_PREDICTIONS.md` (commit 878724e; pre-data addendum de0ff08). {#tbl:points}

**Measures and rule.** The two primary measures at each point were the magnetisation ($|m|$, or signed $m$ at P4)
and persistence, the overlap $\frac{1}{N}\sum_i s_i(t)\,s_i(t_0)$ of each measured sweep's configuration with the
configuration at the end of warm-up $t_0$, averaged over the measured sweeps. The surrogate was
pre-registered as supported if both primary measures fell inside its 90% predictive interval at at least 3 of
P1–P4. Two controls were also predicted: the naive Ising reading of each model and a fields-only model.

**Harness audit.** Before interpreting the Stage 3 outcome, a pre-registered audit (plan commit 4c3924a) checked the
live harness from the call cache without rerunning anything, and pushed a mock surrogate through the unchanged harness.

## 3.5 Kernels {#sec:kernels}

A *kernel* is the full table $P(s_i = +1 \mid c, s_0)$ over contact configurations
$c = (a_P, a_Q, r_P, r_Q)$ (numbers of allies and rivals holding each position) and own position $s_0$. We compare
four kernels:

- **A3**, the committed parametric surrogate, a *trait summary* of each model;
- **K_fixed**, the Stage 1 battery table, with one fixed contact order per cell;
- **K_live**, the empirical kernel estimated from the live Stage 3 updates;
- **K_avg**, a new order-marginalised battery: for each configuration, up to 8 contact orders × 2 option orders ×
  both letter mappings (11,992 calls).

Each kernel was simulated on the live graphs from the live starts ($R = 400$ runs), with the measurement code
unchanged. The plan and an outcome map were committed before any kernel was computed (fe65958; amendment f6b1922).
Per-step agreement is the live-visit-weighted mean $|\Delta P|$ and $|\Delta\,\mathrm{logit}|$ against K_live;
collective agreement is expressed in standard-deviation units of the reference spread.

## 3.6 Residual persistence tests {#sec:residual-methods}

Two tests of a small persistence excess over Markov kernels were pre-registered (plan 9c759f7; amendments b27d83a,
93bd950): (1) a mover–stayer heterogeneity test (permutation test of between-agent variance, and a heterogeneous
kernel); (2) a fresh live replicate run with one call in flight per endpoint and drift sentinels. A replay audit of
the replicate (plan f06db97) and a combined effect size were also pre-registered.

## 3.7 Decision-noise rescaling and transition location {#sec:noise-methods}

Stage 3 ran at each model's natural sampling setting ($T = 1$). To move a population towards a transition, we define
a *decision-noise* axis: each per-prompt letter logit of a kernel is divided by $T$ before averaging over
presentations. This is a defined manipulation of decision noise, not a realistic deployment temperature; its
relation to actual sampling temperature was checked separately (§4.7).

The transition temperature $T_c$ is located from the measured kernels by finite-size scaling on random-regular
$d = 4$ graphs with $N = 100$, 400 and 1,600, random-sequential heat-bath updates and common random numbers. The
locating rule takes the crossing of the Binder cumulant $U = 1 - \langle m^4 \rangle / (3\langle m^2 \rangle^2)$ for
$N = 400$ and $N = 1{,}600$, provided the region is not "frozen" by a hot-start/cold-start check (hot and cold starts
must agree); the $U(100)$–$U(400)$ crossing and scaled $\langle m^2\rangle$ serve as cross-checks. A transition is
recorded only where the maximum Binder cumulant reaches at least 0.5.

## 3.8 Amplification measure and rules {#sec:amp-methods}

**Perturbations.** Controlled per-step errors are added to the kernel's logit at $\delta = 0.1$ nats per step, in
three synthetic shapes: a uniform stay-bias (towards the agent's own position), a majority shift (towards the
majority of contacts), and random per-configuration noise (three independent draws). Real error shapes are
$\mathrm{K\_fixed} - \mathrm{K\_avg}$ (the fixed-order battery) and $\mathrm{A3} - \mathrm{K\_avg}$ (the trait
summary), applied at $\lambda = 1$.

**Amplification.** For each perturbation, paired runs with common random numbers give
$$
A = \frac{|\Delta \langle |m| \rangle|}{\varepsilon}, \qquad \varepsilon = \sum_{c,s_0} w(c,s_0)\, |\Delta\,\mathrm{logit}(c,s_0)|,
$$
where $w$ are the visit frequencies of the unperturbed run, so that $A$ is in $|m|$ per nat of visit-weighted per-step
error. Persistence is a secondary measure.

**Controls.** The kernels of P2 (Qwen3.5 9B political, $\rho = 0.5$) and P4 (Mistral Nemo 12B workplace,
$\rho = 0.25$) serve as controls.

**Stage B (first pre-registered test).** Plan 18f619e, with Amendment 1 (eef27e6) changing the $T_c$ location rule
before any perturbation run. H1: amplification grows with $N$ at $T_c$, $G = A(1600)/A(100) > 1$. H2: the model's
growth exceeds the controls' growth.

**Corrected confirmatory test.** Pre-registered with its model-selection rules and costs before any run or API call
(`llm/AMPLIFICATION_CONFIRM_PLAN.md`, 76aad78; amendments 32daeac, 0b69998, ef5674d), on fresh graphs and seeds:

- **Rule A (decisive):** $D = A_{\text{model}} - \max(A_{P2}, A_{P4})$ at $(T_c, N = 1{,}600)$. Pass iff
  $D \ge 0.10$ $|m|$ per nat and the 95% CI of $D$ excludes 0.
- **Rule B:** growth with $N$, tested only where $A \ge 0.10$ at both $N = 100$ and $N = 1{,}600$.
- **Rule C:** the amplification peak lies within ±10% of $T_c$.

Step 5 tested a low-$T$ edge at 4× run length. GPT-4o-mini was pre-registered (amendment 3) as a real-model negative
control evaluated on both confirmatory temperature bands.

**Model selection.** Confirmatory models were P1 (Qwen3.5 122B, kernel re-located from scratch), Gemma-4 31B
(self-hosted, batch size 1, deterministic), Gemini 3.1 Flash-Lite (sampled at 50 per prompt) and one OpenAI/Meta/DeepSeek
logprob model. The last was the first of 28 frozen candidates to pass a probe for logprobs, temperature honoured,
repeat SD $\le 0.3$ and leak below 5%; this selected GPT-4o-mini. GPT-6-luna was exploratory because its endpoint
refuses the `temperature` parameter. All five pass comprehension (99–100% per question type). The full list of
models studied is in Appendix D.

## 3.9 Pre-registration process {#sec:prereg}

Every confirmatory stage followed the same procedure: the plan, its hypotheses, decision rules, outcome map and
analysis code were committed to the repository before any data for that stage were collected or computed. Any change
after that point was either committed as a dated amendment before the affected data existed, or recorded as a
deviation in the execution log (`llm/EXECUTION_LOG.md`) with when it was decided and its effect (Appendix B). Analyses
not in a plan are labelled post-hoc or exploratory throughout, and pre-registered verdicts are stated before any
post-hoc reading. Remote and paid jobs streamed heartbeats and per-unit outputs to durable storage, and libraries
that generate experimental structure were pinned with committed fingerprints of the generated graphs (introduced
after the deviation in §4.6).

# 4 Results {#sec:results}

## 4.1 The simulator reproduces known physics {#sec:res-baseline}

All six simulated transitions of the symmetric baseline fall within their 95% CIs of the cavity predictions
(@tbl:validation; @fig:crossings).

| transition | theory $T$ | simulated $T$ ($N$ = 1000/2000) | 95% CI |
|---|---|---|---|
| consensus $\rho = 0$ | 3.915 | 3.917 | 3.85–4.01 |
| consensus $\rho = 0.1$ | 3.093 | 3.099 | 3.04–3.19 |
| consensus $\rho = 0.2$ | 2.254 | 2.156 | 1.99–2.28 |
| glass $c = 3$ | 1.519 | 1.528 | 1.45–1.58 |
| glass $c = 4$ | 1.820 | 1.813 | 1.74–1.87 |
| glass $c = 6$ | 2.307 | 2.233 | 2.10–2.32 |

Table: Simulated versus analytic transitions of the Viana–Bray baseline. Source: `results/validation_crossings.csv`.
{#tbl:validation}

![Finite-size crossings for the $c = 4$ baseline. Source: `figures/fig3_crossings_c4.png`.](../figures/fig3_crossings_c4.png){#fig:crossings}

Under non-reciprocity, frozen camps melt quickly. In the glass regime ($\rho = 0.5$, $c = 4$, $T = 1.0$) persistence
falls from 0.51 to 0.28, 0.11 and ≈ 0.01 as $\varepsilon$ goes from 0 to 0.1, 0.2 and 0.35, with no size dependence
across $N$ = 1k–4k. In the consensus regime ($\rho = 0.1$, $T = 2.0$) $|m|$ falls only from 0.62 to 0.55 as
$\varepsilon$ goes from 0 to 1 [`results/asymmetry_sweep.csv`; `figures/fig4_asymmetry.png`]. This is consistent with
@crisanti1987dynamics for dense asymmetric spin glasses and is not new; we report it only to establish that the
simulator reproduces known results.

## 4.2 Stage 1: LLM response functions are not Ising-like {#sec:res-stage1}

**Confirmatory set.** Four of the nine open-weight endpoints tested passed both gates: Llama-3.3 70B, Qwen3.5 9B,
Qwen3.5 122B-A10B and Mistral Nemo 12B, giving 4 endpoints × 3 framings = 12 cells (@tbl:gates). Llama-3.1 8B failed
comprehension (ally-majority 0.78, rival-majority 0.72); Mistral Small 24B failed by leak (it reasons before answering);
the provider-served Gemma-4 26B and 31B failed the noise gate (0.354–0.703); Mistral Large 4 failed the noise gate.

| endpoint | comprehension (own / ally maj. / rival maj.) | pooled repeat SD (nats) | status |
|---|---|---|---|
| Llama-3.3 70B | 1.00 / 0.96 / 0.99 | 0.088 | confirmatory |
| Qwen3.5 9B | 1.00 / 0.96 / 0.96 | 0.037 | confirmatory |
| Qwen3.5 122B-A10B | 1.00 / 1.00 / 0.99 | 0.034 | confirmatory |
| Mistral Nemo 12B | 1.00 / 0.98 / 0.94 | 0.137 | confirmatory |
| Llama-3.1 8B | 0.99 / 0.78 / 0.72 | 0.026 | fails comprehension |
| Mistral Small 24B | 0.98 / 0.18 / 0.10 | 0.04 | fails by leak |
| Gemma-4 26B-A4B (two providers) | 1.00 / 1.00 / 1.00 | 0.703 / 0.354 | fails noise gate |
| Gemma-4 31B | 1.00 / 1.00 / 1.00 | 0.304 | fails noise gate |
| Mistral Large 4 | 1.00 / 0.98 / 0.97 | > 0.30 | fails noise gate |

Table: Stage 1 validity gates. Gate thresholds: comprehension ≥ 0.90 per type; repeat SD ≤ 0.3 nats. Source:
`results/llm/STAGE1_RESULTS.md`. {#tbl:gates}

**Pre-registered verdicts.** @tbl:h1h5 gives the verdicts on the confirmatory set.

| hypothesis | verdict | detail |
|---|---|---|
| H1 valence asymmetry | supported in 9 of 12 cells | All in one direction: rivals weigh less than allies; for both Qwen models rivals mildly *attract*. Symmetric in neutral framing for Llama-70B and Nemo. |
| H2 label fields | supported | Letter field in 10 of 12 cells; content field $h_C$ in 8 of 8 content cells. |
| H3 non-additivity | 2 of 12 (Llama-70B political and workplace, pairwise) | Pairwise significant in 11 of 12 but below the minimum effect in 9; cubic: none. |
| H4 $\beta$ rises with size | not supported (1 of 4 families testable) | Qwen 9B → 122B rises in all framings (Holm $p$ = 0.003); Llama excluded by comprehension, Gemma and Mistral by the noise gate. |
| H5 sum vs average | departs from summation in 7 of 12 | Averaging ($\alpha \approx 1$) for Llama-70B, Nemo and Qwen-9B in neutral framing; Qwen-122B sums ($\alpha \approx 0$). |

Table: Stage 1 pre-registered verdicts (12 confirmatory cells, Holm-corrected). Source:
`results/llm/STAGE1_RESULTS.md`. {#tbl:h1h5}

**Amendments 4–5.** Self-hosted in bf16, the dense Gemma-4 31B passes the noise gate (0.114), while the 26B
mixture-of-experts model fails (0.592) because its bf16 outputs move with batch composition. Adding the 31B gives 15
cells; H1 then holds in 12 of 15, H3 cubic replicates for Gemma-31B in neutral and political framing, and no original
verdict changes [`results/llm/AMENDMENT4_RESULTS.md`]. A rescue rerun of the 26B at batch size 1 with deterministic
kernels gives identical replicates; the extended set of 18 cells has H1 in 15 of 18. $\beta$ also rises within the
Gemma family (26B → 31B, neutral 0.75 → 1.11; CI of the difference [0.17, 0.56]). H4 remains not supported overall:
2 of 2 testable families, whereas the rule requires at least 3 of 4 [`results/llm/AMENDMENT5_RESULTS.md`].

**Gate-relaxed sensitivity (post-data).** Adding the three gate-failed provider endpoints (21 cells), H1 holds in 18 of
21, H3 cubic in 4 cells (all on jittery endpoints; two later replicated on the clean local 31B), and H4 would be
supported (3 of 3 testable families). The confirmatory analysis was not changed.

**Provider and batch effects (secondary).** Provider choice shifts prompt-format biases (the letter field $h_L$ and
order field $h_O$). Social coefficients agree between provider and local serving in two of three framings and
differ by 7–9% in the third (the only clean pair). Batched serving of the mixture-of-experts model moves single answers
by an RMS of 0.86 nats relative to batch size 1, and flips 2.5% of majority answers.

**Temperature.** The pre-registered $1/T$ verification on GPT-4o-mini failed: the $\chi^2$ test rejects at $T = 0.5$
and $T = 1.5$, although the slopes match $1/T$. Reference-logit jitter explains most but not all of the misfit. The
same check on Qwen3.5 9B passed at both temperatures (@tbl:temp9b).

| $T$ | slope [95% CI] | expected | $\chi^2$ $p$ |
|---|---|---|---|
| 0.5 | 2.22 [1.87, 2.61] | 2.0 | 0.13 |
| 1.5 | 0.68 [0.45, 0.93] | 0.67 | 0.57 |

Table: $1/T$ check on Qwen3.5 9B. At $T = 1.5$, 99 of 1,000 samples were invalid; the $1/T$ rule describes the choice
conditional on a valid reply. Source: `results/llm/temperature_verify_qwen-9b.json`. {#tbl:temp9b}

**Spot checks (sampled, exploratory).** GPT-6-luna is the most Ising-like endpoint ($\beta$ 0.46; rivals oppose,
$b_{\text{rival}}/b_{\text{ally}} = 0.65$; small inertia, $\gamma$ 0.47; additive; $\alpha \approx 0.95$). Gemini 3.1
Flash-Lite follows allies strongly, ignores rivals, and switches away from its own position when only rivals are
present. Claude Haiku 5.5 follows allies, rivals mildly attract in neutral framing, and it has strong inertia
[`results/llm/STAGE1_RESULTS.md`].

**Exploratory probes.** Rival-majority information is decodable from Gemma-4 31B activations (deconfounded AUC up to
0.96 at layers 30–60) and is used ($|b_r|/b_a = 0.71$). An earlier "represented but not used" reading for the 26B did
not survive a confidence interval and was revised. No claim rests on probes.

**Summary: five differences from Ising spins.** (1) Rival ties behave closer to no tie than to $J < 0$. (2) Inertia
is strong and framing-dependent ($\gamma$ = 1.1–7 in content framings), a self-coupling that the heat-bath has no term
for. (3) Several models average their neighbours ($\alpha \approx 1$), which induces non-reciprocity on
heterogeneous-degree graphs. (4) Letter, order and content fields are present. (5) Contact order matters strongly: in
neutral framing, Llama-70B's logit SD across permutations of the same cell is 2.95 nats, a voter-like primacy effect.

![Stage 1 response-function invariants across endpoints and framings. Source: `figures/llm_stage1_invariants.png`.](../figures/llm_stage1_invariants.png){#fig:invariants}

## 4.3 Stage 3: surrogate predictions fail on live networks {#sec:res-stage3}

**Surrogate predictions (Stage 2).** The naive Ising reading of each model mispredicts the surrogate in both
directions; for example, in neutral framing at $\rho = 0.25$ Qwen-122B's surrogate orders ($|m|$ 0.39) while its Ising
reading does not (0.09). Rival ties often fail to frustrate; political framing freezes agents by stubbornness rather
than frustration; and directed tie perceptions change $|m|$ by at most 0.06 at Stage 3 scale
[`results/llm/STAGE2_RESULTS.md`].

**Pre-registered verdict: the surrogate is not supported.** The live run comprised 70 chains and 140,000 live
updates, with invalid replies at most 0.02% at every point. The surrogate passed both primary measures at 0 of P1–P4
(rule: at least 3). The calibration point P5 and both no-contacts arms also missed (@tbl:stage3). At P1, P2, P4 and
P5 the observed values also fell outside both control intervals (naive Ising and fields-only).

| point | primary measure 1 | primary measure 2 |
|---|---|---|
| P1 | $\|m\|$ 0.21 [0.24, 0.46] | persistence 0.78 [0.66, 0.74] |
| P3 | $\|m\|$ 0.10 [0.20, 0.37] | persistence 0.99 [0.91, 0.95] |
| P4 | $m$ 0.94 [0.87, 0.91] | persistence 0.90 [0.75, 0.84] |
| P5 | $\|m\|$ 0.60 [0.14, 0.24] | persistence 0.66 [0.11, 0.20] |

Table: Stage 3 observed values (mean of 5 graphs) against the surrogate's pre-registered 90% interval. Source:
`results/llm/REPORT.md` §4.2 (EXECUTION_LOG entry 26). {#tbl:stage3}

**Harness audit.** Every check on the live logs passed with zero mismatches: update scheme, stale state, option order
and letter mapping, own-position feed, tie labels, prompt equivalence (all 140,005 prompts), provider pinning,
invalid replies, time alignment, and measurement code (reproduced to $1.1 \times 10^{-16}$). A mock surrogate pushed
through the unchanged harness reproduced the simulator within 0.013 on every measure. The pre-stated mock check A
nonetheless failed at P2 and P4-nc, by rule. Post-hoc, the cause is that the committed intervals were not conditional
on the five graphs actually used: on those graphs the simulator itself falls inside the intervals only 80–86% of the
time. This does not rescue the surrogate: against same-graph distributions the live misses are 3.6–20 SD at P1, P3,
P4 and P5, and only P2's miss is consistent with a correct surrogate [`results/llm/STAGE3_AUDIT_RESULTS.md`].

**Contact order (post-hoc).** On live calls whose contact order matched the battery prompt's, live answers equal the
battery's (RMS logit difference 0.06–0.35); on other orders they differ by 0.44–0.97 nats. The fitted surrogate had no
order term.

## 4.4 Kernel test: order-averaged tables succeed per step {#sec:res-kernel}

**Pre-registered verdicts.** No kernel reproduces the live collective measures at 3 or more of P1–P4 (@tbl:kernels).
The governing branch of the outcome map is "K_live fails → the live system is not Markov in configuration;
investigate before claiming anything", which triggered the residual tests (§4.5).

| kernel | P1–P4 passing (rule ≥ 3) | reproduces live? |
|---|---|---|
| K_fixed | 2 | no |
| A3 | 1 | no |
| K_live | 2 | no |
| K_avg | 2 (P2, P4) | no |

Table: Kernel test, pre-registered verdicts. Source: `results/llm/KERNEL_TEST_RESULTS.md`. {#tbl:kernels}

**Per-step agreement.** The order-averaged table K_avg matches the live kernel to $|\Delta P| \le 0.004$ at P1–P4,
substantially better than the fixed-order battery and the parametric trait summary (@tbl:perstep).

| point | K_avg $\|\Delta P\|$ / $\|\Delta\,\mathrm{logit}\|$ | K_fixed | A3 |
|---|---|---|---|
| P1 | 0.004 / 0.05 | 0.010 / 0.22 | 0.019 / 0.72 |
| P2 | 0.002 / 0.03 | 0.008 / 0.16 | 0.010 / 0.19 |
| P3 | 0.001 / 0.03 | 0.002 / 0.07 | 0.016 / 0.67 |
| P4 | 0.003 / 0.07 | 0.008 / 0.36 | 0.013 / 0.66 |
| P5 | 0.013 / 0.65 | 0.037 / 0.65 | 0.123 / 1.63 |

Table: Live-visit-weighted per-step disagreement with K_live. Source: `results/llm/KERNEL_TEST_RESULTS.md`.
{#tbl:perstep}

**Collective amplification of per-step differences.** A3 differs from the live system by 3–20 SD; at P5 the A3 fit
predicts disorder while both empirical kernels predict ordering. K_fixed and K_live differ by at most 3.4 SD (P3
persistence), from a per-step $|\Delta P|$ of 0.002. Every primary-measure mean of K_avg at P1–P4 is within 0.9 SD of
K_live's; K_avg predicts the population as well as the live kernel does.

**Reading (governed by the outcome map; no headline was drawn at the time).** (1) Most of the Stage 3 miss is due to
the parametric trait summary: the single-agent data held the information, but the fitted functional form did not
carry it. (2) An order-averaged one-shot battery recovers the live per-step kernel. (3) Differences invisible per step
can still matter collectively.

**Within-model illustration (exploratory).** The saturation knockout of the Gemma mechanism test (§4.8) is in effect a
trait summary of Gemma-4 31B: an additive logit with Gemma's own fitted ally, rival and own-position weights (+1.23,
−0.90, +0.13 at $T = 2$), fitted to its measured table. It puts the population in a different phase from the table it
summarises: disordered ($|m|$ 0.09 from both starts) instead of slowly ordering and start-dependent (cold starts
$|m| \approx$ 0.34–0.40) [`results/llm/GEMMA_MECHANISM_RESULTS.md`].

**Post-hoc checks.** Follow-up checks of the logs found no non-Markov mechanism: provider caching, wall-clock drift,
history dependence, lag-1 correlation, wider neighbourhood, and concurrency (one small effect at one provider cannot
explain the pattern).

## 4.5 Residual persistence does not replicate {#sec:res-residual}

The live runs sat slightly on the high-persistence side of every Markov kernel (Stouffer $z \approx +3.1$ over 7
points against K_live; excess 0.01–0.04 in persistence units).

**Test 1 (heterogeneity).** No agent-specific content is sent: all 140,005 prompts are deterministic renders of the
configuration. The pre-stated permutation test of between-agent variance was significant (Fisher $p$ = 0.008),
contrary to the prediction. A post-hoc within-cell shuffle traces most of this to agents clustering in
configurations, leaving only P4-nc. A heterogeneous kernel does not explain the excess (Stouffer 2.84 vs 2.72
homogeneous); the pre-stated criterion is not met. A pre-registered identical-request check was biased by selection
on the earlier choice; selection-corrected (post-hoc), there is no dependence beyond draw luck at two logprob points.

**Test 2 (fresh replicate). Pre-registered result: Stouffer $Z = -0.42$ ($p = 0.66$); the excess does not repeat.**
Both original point misses moved, in different directions. Sentinels show hours-scale drift at some providers, but it
does not track the collective measures. A replay audit of all 70 replicate chains passed every check. A deviation
affected this test: see §4.6.

| run | effect [95% CI] |
|---|---|
| original | +0.0116 [+0.0033, +0.0200] |
| replicate | −0.0051 [−0.0135, +0.0033] |
| pooled | +0.0033 [−0.0026, +0.0093] |

Table: Combined persistence effect, raw persistence units. The runs are heterogeneous (Cochran $Q$ = 7.6,
$p$ = 0.006). Source: `results/llm/stage3_rep1/persistence_effect_combined.json`. {#tbl:residual}

Under the pre-registered map the reading is "realisation noise or our own concurrency; not separable". We treat the
persistence excess as unestablished.

## 4.6 A disclosed deviation in the replicate {#sec:res-deviation}

The remote pods that ran the replicate built random-regular graphs with networkx 3.3 instead of the local 3.7, and
`random_regular_graph` builds different graphs from the same seed across versions. As a result 69 of 70 replicate
chains ran on different graphs drawn from the same ensemble. The pre-registered analysis ($Z = -0.48$) used the wrong
graphs; a mechanical reanalysis on the actual graphs, committed before it ran, is the reported result ($Z = -0.42$).
The reading does not depend on the deviation. The deviation was found through an implausible secondary measure after
the run; library pinning and a graph-fingerprint assertion are now required for every remote job (Appendix B, row 10).

## 4.7 Amplification near tipping points {#sec:res-amp}

### 4.7.1 Stage B: the first pre-registered test was refuted {#sec:res-stageb}

For P1 (Qwen-122B, K_avg kernel), $T_c = 1.93$ [1.92, 1.96]; the Stage 3 setting $T = 1$ sits at $T/T_c = 0.52$, deep
in the ordered phase [`results/llm/AMPLIFICATION_RESULTS.md`]. H1 (growth with $N$ at $T_c$) holds for all three
shapes: $G$ = 3.0 [2.3, 4.3] for stay-bias, 3.4 [2.8, 4.2] for majority shift and 5.5 [4.7, 6.7] for noise. H2 (growth
exceeds the controls' growth) fails for majority shift against P2 (ratio 0.55 [0.05, 3.8]). **Pre-registered verdict:
refuted.**

The cause is a defect of the rule rather than of the data. P2's majority-shift amplification at $T_c$ was 0.002
($N = 100$) and 0.010 ($N = 1{,}600$), so its own growth ratio, 6.2 [0.9, 68], was uninformative: a comparison of
growth *ratios* is ill-defined when a control's amplification is near zero. Two further pre-stated expectations also
failed. (ii) Growth is not "bounded away from $T_c$": symmetry-breaking noise keeps growth ≈ 4–6 above $T_c$, a
finite-size effect of $|m|$, and growth reaches 12–22 near a low-$T$ freezing edge. (iii) The controls were not
"bounded", again because of near-zero denominators. Before any perturbation run, Amendment 1 had replaced the $T_c$
location rule after a spurious low-$T$ crossing from an under-equilibrated $N = 1{,}600$ run (Appendix B, row 12).

**Real evaluation errors placed at the Stage 3 operating point.** At $T = 1$ the fixed-order battery error (0.48 nats
per step) moves $|m|$ by +0.004 and the trait-summary error (1.71 nats) by −0.034 ($N = 100$, equilibrium). The same
K_fixed error compared at the same $N$ is much larger at $T_c$ (@tbl:samen): at $N = 100$ the effect per run is about
16× larger at $T_c$ and about 56× per nat (the per-step error is smaller at $T_c$ because logits are divided by $T$);
at $N = 1{,}600$ the $T = 1$ effect is indistinguishable from zero and the per-nat ratio is at least about 75×, using
the upper CI bound at $T = 1$.

| $N$ | $\Delta\|m\|$ at $T = 1$ | $\Delta\|m\|$ at $T_c$ | per nat at $T = 1$ [95% CI] | per nat at $T_c$ [95% CI] |
|---|---|---|---|---|
| 100 | +0.004 | +0.068 | 0.009 [0.006, 0.011] | 0.50 [0.43, 0.57] |
| 1,600 | −0.002 | +0.169 | 0.004 [0.000, 0.017] | 1.27 [1.20, 1.34] |

Table: The fixed-order battery error at the Stage 3 operating point and at P1's $T_c$, same $N$. Source:
`results/llm/amplification/amplification.csv` (equilibrium protocol, rows `fixed`, $\lambda = 1$). {#tbl:samen}

![Stage B amplification of $|m|$ across decision noise. Source: `results/llm/amplification/fig2_amplification_abs_m.png`.](../results/llm/amplification/fig2_amplification_abs_m.png){#fig:stageb}

### 4.7.2 Corrected confirmatory test {#sec:res-confirm}

**Temperature check (pre-registered).** Qwen-122B, Nemo-12B (control P4) and GPT-4o-mini fail the $1/T$ check, with
misfit at $T = 0.5$ beyond reference noise; at $T = 1.5$ all three fit within noise, although Qwen-122B narrowly
misses on slope (CI 0.68–1.08 vs 0.67). Gemini passes ($\kappa = 1.13$ [0.96, 1.31]). For Gemma-31B served locally the
rule holds by construction (softmax of logits divided by $T$, no top-$p$/top-$k$); GPT-6-luna's endpoint refuses
`temperature` [`results/llm/confirm/api_summary.json`; `results/llm/confirm/temp_diag.json`].

**Transitions.** P1: $T_c = 2.01$ [1.96, 2.07], located independently of Stage B's 1.93. Gemini: $T_c = 1.83$ [1.81,
1.86]. GPT-6-luna (exploratory): 0.49, on the unverified sharpening side. GPT-4o-mini and Gemma-31B: "no transition in
the swept range", as pre-registered (§4.8) [`results/llm/confirm/sim/locate_*.json`; @fig:binder].

![Binder cumulants for the confirmatory models. Source: `results/llm/confirm/fig_binder_step3.png`.](../results/llm/confirm/fig_binder_step3.png){#fig:binder}

**Pre-registered verdicts.** Rule A passes for 3 of 3 shapes in both models with a transition (@tbl:rulea; @fig:rulea).

| model | shape | $A$(model) | max control | $D$ [95% CI] | Rule A | Rule B: $G$ [95% CI] | Rule C |
|---|---|---|---|---|---|---|---|
| P1 | stay-bias | 0.31 | 0.16 | 0.15 [0.09, 0.22] | pass | 1.36 [0.96, 2.08], fail | pass |
| P1 | majority | 1.10 | 0.31 | 0.79 [0.68, 0.90] | pass | 1.74 [1.44, 2.11], pass | pass |
| P1 | noise | 2.18 | 0.15 | 2.03 [1.96, 2.10] | pass | 11.7 [9.2, 14.4], pass | pass |
| Gemini | stay-bias | 1.35 | 0.15 | 1.20 [1.12, 1.28] | pass | 3.43 [3.02, 3.94], pass | pass |
| Gemini | majority | 1.30 | 0.27 | 1.03 [0.94, 1.13] | pass | 3.75 [3.15, 4.61], pass | pass |
| Gemini | noise | 1.33 | 0.16 | 1.17 [1.10, 1.25] | pass | 9.5 [7.8, 12.1], pass | pass |

Table: Corrected confirmatory amplification test at $(T_c, N = 1{,}600)$; $A$ and $D$ in $|m|$ per nat. Source:
`results/llm/AMPLIFICATION_CONFIRM_RESULTS.md`; `results/llm/confirm/results/step4.json`. {#tbl:rulea}

![Rule A: amplification at $T_c$ minus the worst-case control, with 95% CIs. Source: `results/llm/confirm/results/fig_ruleA.png`.](../results/llm/confirm/results/fig_ruleA.png){#fig:rulea}

For P1, amplification at the tipping point exceeds the controls for 3 of 3 shapes; Rule B passes for 2 of 3, with
stay-bias missing against the plan's prediction that all would pass; Rule C passes for 3 of 3. P1's stay-bias result
is a close call: it passes Rule A on the point estimate, but the CI lower bound (0.09) is below the 0.10 margin.
Gemini passes Rules A, B and C for 3 of 3 shapes. The real fixed-order battery error at P1's $T_c$ has $A = 0.62$ and
grows with $N$ ($G$ = 1.73 [1.43, 2.12]). In persistence (secondary), Rule A passes for P1 in 2 of 3 shapes and for
Gemini in 3 of 3; growth is mostly not significant.

**Negative control (pre-registered, amendment 3).** Over 26 temperatures spanning both confirmatory bands,
GPT-4o-mini's $|m|$ amplification never reaches the 0.10 floor (maxima 0.075, 0.089 and 0.096 by shape), so its
prediction is reported as "below floor everywhere"; both test models exceed it at their $T_c$ with every CI excluding
0. In persistence it does cross the floor, but never with upward $N$-growth; a few points shrink with $N$, which
violates the literal prediction in the direction opposite to growth.

**Step 5 (low-$T$ edge; a defined decision-noise rescaling, not a verified sampling temperature).** At 4× run length
no shape shows $N$-growth. Verdict: the Stage B growth of 12–22 near $T \approx$ 0.6–0.8 was a finite-run-length
artefact, not a second transition.

**Exploratory.** GPT-6-luna passes Rule A for 3 of 3 shapes, but its $N = 1{,}600$ amplification curve is flat with no
clear peak, and its $T$ axis is unverified.

## 4.8 Whether a tipping point exists is model-dependent {#sec:res-modeldep}

Of the four confirmatory models, two (Qwen-122B and Gemini) have a clear order–disorder transition in the swept
decision-noise range, GPT-4o-mini has none, and Gemma-4 31B is unresolved
[`results/llm/AMPLIFICATION_CONFIRM_STOP2.md`; `results/llm/GEMMA_MECHANISM_RESULTS.md`].

**GPT-4o-mini never orders.** The highest Binder cumulant at $N = 1{,}600$ is 0.47, at $T = 0.2$ (the rule needs at
least 0.5), and $|m| \le 0.22$. Hot and cold starts agree everywhere.

**Gemma-4 31B: pre-registered verdict "no transition in range"; post-hoc reading "unresolved".** Hot and cold starts
disagree at every $T \le 3.16$, so the locating rule marked that region "frozen" (the rule's term) and found no
transition. Cold starts decay from all-up to $|m| \approx 0.44$ while hot starts stay near 0.05, and the result is
unchanged with exact full-vocabulary $P(A)$. Our post-hoc reading is that a hot/cold rule cannot locate a transition
when relaxation near it is this slow; we describe the regime as slow, start-dependent relaxation near an ordering
boundary. This label is inferred from the knockouts below; the mechanism is not identified, and the start-dependence is
borderline and seed-sensitive at $T = 2.0$.

Stage 1 does not settle the mechanism. In the neutral framing simulated here, Gemma-31B's rivals repel at 71% of ally
weight ($b_r$ 0.93 [0.79, 1.11] vs $b_a$ 1.30), which with 25% rival ties is compatible with frustration; its inertia
is small and not significant ($\gamma$ = 0.08 [−0.15, 0.32]) [`results/llm/stage1_cells_amendment4.csv`].

**Exploratory mechanism test (plan committed first).** Removing rivals' influence makes the population order strongly
($|m|$ 0.77); replacing the table by its additive projection makes it disordered ($|m|$ 0.09). Either way
start-dependence disappears, so under the pre-stated reading no single mechanism is identified. Replica overlaps show
no frozen microscopic state. The regime depends jointly on rival repulsion and the non-additive response; inertia was
not knocked out.

**Exploratory run-length and outcome checks.** Start-dependence persists at $T \approx$ 2.0–2.3 even at 4× run
length and dissolves above $T \approx 2.7$ [`results/llm/confirm/results/gemma_exploratory.json`]. This is
seed-sensitive: in the mechanism test's rerun with other seeds it persisted at $T = 2.0$ only (gap 0.27 → 0.18), and
at 2.16 and 2.34 it decayed by more than half (to 0.13 and 0.07). A 0.1-nat per-step error changes which collective
basin a population reaches in 24–51% of paired runs from identical starts; a different random seed alone changes it in
43–60%. In this regime the outcome is close to a coin flip set by the stochastic path.

**Summary.** GPT-4o-mini and Gemma both pass comprehension and have well-defined single-agent response tables, yet
nothing in their per-agent statistics announces whether a population of them will have a tipping point. In this
study, whether a tipping point exists, where it lies and how strongly it amplifies errors were knowable only by
simulating the population with the measured kernel. The Stage 3 setting ($T = 1$) sits far from P1's tipping point
($T/T_c \approx 0.5$), consistent with the small collective effect of the real evaluation errors there (+0.004 in
$|m|$).

# 5 Discussion {#sec:discussion}

*[PLACEHOLDER — to be written by the author.]*

# 6 Implications {#sec:implications}

*[PLACEHOLDER — to be written by the author. Source material: `results/llm/REPORT.md` §10, which keeps every
statement within the claims above.]*

# 7 Limitations {#sec:limitations}

**Graphs.** Every live and amplification result uses random-regular graphs of degree 4. Erdős–Rényi,
heterogeneous-degree and real-network topologies were explored only with Stage 2 surrogates. Because several models
average over contacts ($\alpha \approx 1$), heterogeneous degree would induce effective non-reciprocity that our
graphs do not exercise.

**Task.** The task is a binary choice with explicit ally/rival labels and visible current positions. There is no free
text, persuasion, deliberation or memory, and the framings (neutral, political, workplace) are short vignettes. We
make no claim about richer interaction formats.

**Memoryless agents.** Every update is one call built from the current state only. The kernel tests found no per-step
history dependence ($|\Delta P(\text{stay})| \le 0.002$), but agents with memory or persistent context were not
tested, and memory is a plausible source of exactly the non-Markov effects we looked for.

**Models and providers.** Nine open-weight endpoints were tested in Stage 1 (4 confirmatory; 6 with the self-hosted
Gemma rescue analyses), plus sampled frontier spot checks; five endpoints were run live in Stage 3; and four
confirmatory models plus one exploratory model entered the amplification test. All were pinned to specific providers,
and provider choice and batched serving shift format biases and single answers (§4.2). Results may not transfer to
other versions of the same models, other providers or other serving configurations.

**The decision-noise axis.** The $T$ axis divides measured logits by $T$. It is a defined manipulation of decision
noise, not a realistic deployment setting. The $1/T$ rule fits within noise only on the noise side ($T > 1$); on the
sharpening side ($T = 0.5$) it fails beyond reference noise for every logprob model checked. GPT-6-luna's endpoint does
not accept a temperature parameter, and Gemma's invalid-reply rate reaches 15% at $T = 2$. Results at $T < 1$ (Step 5,
GPT-6-luna, GPT-4o-mini's low-$T$ range) are labelled accordingly.

**Amplification is simulation-based.** The amplification results come from empirical kernels, not from live
populations pushed towards $T_c$; live networks were run only at $T = 1$. That the kernels describe live per-step
behaviour rests on the kernel test ($|\Delta P| \le 0.004$ at P1–P4), which was done at $T = 1$ only. Even at
$T = 1$, K_avg and K_live each miss 2 of 4 live points collectively.

**Model selection.** The selection of GPT-4o-mini as the confirmatory logprob model is borderline: its dated alias,
the same weights on the same provider, failed the temperature probe ($\kappa$ 0.12 [−0.22, 0.43]), and the probe's
5 × 40-sample test is noisy. GPT-4o-mini was also the pre-stated fallback.

**Unresolved cases.** Gemma-4 31B's collective behaviour is unresolved: the locating rule could not classify it, the
mechanism test did not identify a single mechanism, and the start-dependence is seed-sensitive. The persistence excess
over Markov kernels was not established; the two runs disagree, and our own concurrency cannot be separated from
realisation noise.

**Statistical power.** Stage 3 used $N = 100$ and 5 graphs per point. Live collective misses at about 2 of 4 points
remain unexplained beyond the amplification argument.

**Pre-registration and deviations.** Two first-stage pre-registered tests failed for reasons in their own design (the
Stage 3 intervals were not conditional on the graphs used; the Stage B growth-ratio comparison was ill-defined near
zero), and the corrected versions were designed after seeing those failures, although they were run on fresh graphs
and seeds. One replicate ran on different graphs than pre-registered (§4.6). All deviations are listed in Appendix B.

**Motivating results.** The five September 2026 proofs that motivated the study are unrefereed preprints. Nothing in
this paper depends on their correctness.

# 8 Conclusion {#sec:conclusion}

*[PLACEHOLDER — to be written by the author.]*

# Data and code availability {-}

All code, pre-registrations, amendments, the execution log and result files are in the repository
`randallspqr/social-phase-transitions` [URL to be added], archived with DOI [10.5281/zenodo.XXXXXXX, to be added].
Code is released under Apache-2.0 and data under CC BY 4.0 (`LICENSES.md`).

**Data.** The Hugging Face dataset `RandallSPQR/social-phase-transitions-llm-cache` [to be made public] holds:
- the raw API cache: every request body and compact reply, keyed by the SHA-256 of the request plus replicate index,
  with no credentials or account-linked identifiers; 31 files, 709,422 records, each listed with its SHA-256 in
  `results/llm/cache_manifest.json`;
- the GPU and CPU pod outputs and logs;
- the per-unit outputs of the long simulations.

**Replication.** `make replicate-fast` (about RT_FAST on 4 cores) and `make replicate` (about RT_FULL; recomputes every
simulation unit) reproduce every analysis and figure in this paper from that dataset, with no API keys and no GPU:
- every downloaded file is checked against its committed SHA-256;
- every regenerated result is compared with the committed one within stated tolerances;
- each claim of §4 gets a pass/fail result, including the headline numbers printed here.

The environment is pinned (`replication/requirements.lock`, `Dockerfile`). `replication/README.md` maps each claim to
its scripts, result files and figures, and separates this exact reproduction from re-collection of the data, which
needs API keys and a GPU (about $37) and is not expected to be identical because providers drift.

**Graphs.** Graph fingerprints are asserted before use. They are stored in:
- `results/llm/stage3/graph_hashes.json` (Stage 3);
- `results/llm/stage3_rep1/graphs/` (the replicate's actual graphs);
- `results/llm/confirm/sim/locate_*.json` (the confirmatory phase).

# AI use {-}

Claude (Anthropic; Opus 5.5, used via Claude Cowork and Claude Code) assisted with experimental design review,
simulation and analysis code, statistical checks, figure generation, and drafting of the Methods and Results
sections [@anthropic2026claude; @anthropic2026claudecode]. All pre-registrations, amendments and analysis decisions
were reviewed and approved by the author. All numbers were traced to committed result files, and all references were
verified by the author. The language models studied as experimental subjects are listed separately in Section 3 and
Appendix D; one of them (Claude Haiku 5.5, a Stage 1 spot-check subject) is from the same developer as the assistant
but played no role in the analysis.

# References {-}

::: {#refs}
:::

<!-- Reference list is generated from refs.bib. All entries are marked UNVERIFIED in their `note` field. -->

\appendix

# Appendix A. Pre-registration and amendment log {#app:prereg}

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
| `llm/RESIDUAL_TEST_PLAN.md` | 9c759f7; b27d83a; 93bd950 | heterogeneity test, replicate design, sentinels, no-contacts predictions |
| `llm/REPLAY_AUDIT_PLAN.md` | f06db97 | replicate replay audit and combined effect size |
| `llm/AMPLIFICATION_PLAN.md` | 18f619e; amendment eef27e6 | Stage B amplification curve; $T_c$ relocation |
| `llm/AMPLIFICATION_CONFIRM_PLAN.md` | 76aad78; 32daeac; 0b69998; ef5674d | corrected confirmatory test; model selection; guardrail re-pin; negative control and labels |
| `llm/GEMMA_MECHANISM_PLAN.md` | f9d5e0c | exploratory Gemma mechanism test with pre-stated reading (before any run) |

Table: Pre-registration documents and amendments, with the commits that fixed them. The full chronological record is
`llm/EXECUTION_LOG.md` (entries 1–71). {#tbl:prereg}

# Appendix B. Deviations {#app:deviations}

| # | where | deviation | when decided | effect | record |
|---|---|---|---|---|---|
| 1 | Stage 1 | Answer instruction changed after the pilot (leak 24–94%) | before the main run | pilot data not analysed | Amendment 2 |
| 2 | Stage 1 | Noise gate first applied per arm (bug) | before any confirmatory output | none | log 2 |
| 3 | Stage 1 | Gemma backups failed (one provider 0.354; another returned no logprobs on 77% of calls) | as pre-listed | Gemma excluded from confirmatory set and H4 | log 3 |
| 4 | Amendment 4 | Padding-free batches instead of left padding | before the GPU run | none on the estimand | `AMENDMENT4_RESULTS.md` |
| 5 | Amendment 4 | Comprehension summary overwritten by a test, then restored byte-identical | before the analysis | none | log 33 |
| 6 | Amendment 5 | GPU caps raised (2.5 h / \$6 → 3.5 h / \$6.50) after measuring throughput | before the main run | cost | `AMENDMENT5_RESULTS.md` |
| 7 | Kernel test | K_avg call count 11,992, not ≈ 7,500; two analysis bugs fixed before reading | before reading | none | `KERNEL_TEST_RESULTS.md` |
| 8 | Residual | Identical-request expectation biased by selection; corrected test added | post hoc, labelled | reading changed to "no dependence" | `RESIDUAL_TEST_RESULTS.md` |
| 9 | Residual | Heterogeneity robustness (within-cell shuffle) | post hoc, labelled | signal mostly configuration clustering | same |
| 10 | Replicate | Pods used networkx 3.3 graphs (69/70 chains); mechanical reanalysis on actual graphs | after the run, before reporting | verdict unchanged ($Z$ −0.48 → −0.42) | log 51 |
| 11 | Replicate | Six interruptions, resumed from cache; replay audit passed | during the run | none | log 43–50, 53 |
| 12 | Stage B | $T_c$ location rule replaced (spurious low-$T$ crossing; under-equilibrated $N = 1{,}600$) | before any perturbation run | $T_c$ 0.36 → 1.93 | amendment 1, log 55 |
| 13 | Stage B | Locate units also ran perturbations (bug); 4 files deleted unread | before any reading | none | log 55 |
| 14 | Confirm | Account guardrail blocked one first-party provider; pin rule re-applied over reachable providers | before any data from those providers | candidates 1 and 3 re-pinned; both then failed the gate | amendment 2 |
| 15 | Confirm | Gemma fingerprint gate coded without the $/\sqrt{2}$ of the pre-registered definition; corrected after it fired (author decision) | after the gate fired | relaunch; gate 0.234 passes | log 62–63 |

Table: Deviations from pre-registered procedures. "log $n$" refers to `llm/EXECUTION_LOG.md` entry $n$.
{#tbl:deviations}

# Appendix C. Costs {#app:costs}

| phase | item | cost (USD) | source |
|---|---|---|---|
| Stage 1 | API (battery, spot checks, bridge) | 9.56 | `STAGE1_RESULTS.md` |
| Stage 3 | live networks, 140,000 updates | 4.95 | log 26 |
| Amendments 4–5 | GPU (A100), including failed launches | ≈ 11.90 | `AMENDMENT5_RESULTS.md` |
| Kernel test | K_avg battery | 0.58 | `KERNEL_TEST_RESULTS.md` |
| Residual | replicate API | 5.29 | `RESIDUAL_TEST_RESULTS.md` |
| Residual | replicate CPU pods | ≈ 0.9 | same |
| Confirmatory | API (probes, checks, comprehension, batteries) | 3.33 | `AMPLIFICATION_CONFIRM_STOP2.md` |
| Confirmatory | GPU (two A100 pods) | ≈ 0.91 | log 64 |
| Stage 2, kernel sims, amplification, mechanism test | CPU in session containers | 0 | — |
| **Total** | | **≈ 37.4** | sum of the above |

Table: Compute and API costs by phase. The total is the sum of per-phase figures and has not been reconciled against
account billing; GPU figures are estimated as rate × uptime. {#tbl:costs}

# Appendix D. Models studied {#app:models}

The models below are the *experimental subjects* of this study. They are distinct from the AI tools used to conduct
and write it (see "AI use"). Model identifiers and provider pins are as in `llm/endpoints.py` and
`results/llm/AMPLIFICATION_CONFIRM_STOP2.md`; all API access was via OpenRouter with the provider pinned and fallbacks
disabled, except the self-hosted Gemma runs.

| model | identifier | serving (provider pin) | readout | role(s) |
|---|---|---|---|---|
| Llama-3.3 70B Instruct | `meta-llama/llama-3.3-70b-instruct` | CoreWeave fp16 (Parasail fp8 for provider robustness) | logprobs | Stage 1 confirmatory; Stage 3 P3 |
| Qwen3.5 9B | `qwen/qwen3.5-9b` | Parasail bf16 | logprobs | Stage 1 confirmatory; Stage 3 P2; amplification control |
| Qwen3.5 122B-A10B | `qwen/qwen3.5-122b-a10b` | Alibaba | logprobs | Stage 1 confirmatory; Stage 3 P1; amplification confirmatory |
| Mistral Nemo 12B | `mistralai/mistral-nemo` | Io Net fp16 | logprobs | Stage 1 confirmatory; Stage 3 P4; amplification control |
| Llama-3.1 8B Instruct | `meta-llama/llama-3.1-8b-instruct` | CoreWeave bf16 | logprobs | Stage 1, failed comprehension |
| Mistral Small 3.2 24B | `mistralai/mistral-small-3.2-24b-instruct` | Parasail bf16 | logprobs | Stage 1, failed by leak |
| Mistral Large 4 | `mistralai/mistral-large-4-0` | Mistral | logprobs | Stage 1, failed noise gate |
| Gemma-4 26B-A4B | `google/gemma-4-26b-a4b-it` | Parasail / CoreWeave bf16; self-hosted bf16 (Amendments 4–5) | logprobs | Stage 1, failed noise gate; rescue (Amendment 5) |
| Gemma-4 31B | `google/gemma-4-31b-it` | Io Net; Novita (backup); self-hosted bf16, batch size 1 | logprobs / full-vocabulary $P(A)$ | Stage 1 extended set (Amendment 4); amplification confirmatory (unresolved) |
| GPT-4o-mini | `openai/gpt-4o-mini` | OpenAI | logprobs; sampled (bridge) | Stage 1 bridge and temperature check; amplification confirmatory (no transition) and negative control |
| GPT-6-luna | `openai/gpt-6-luna` | OpenAI | sampled | Stage 1 spot check; Stage 3 P5; amplification exploratory |
| Gemini 3.1 Flash-Lite | `google/gemini-3.1-flash-lite` | Google AI Studio | sampled | Stage 1 spot check; amplification confirmatory |
| Claude Haiku 5.5 | `anthropic/claude-haiku-5.5` | Anthropic | sampled | Stage 1 spot check only (experimental subject; not an analysis tool) |

Table: Language models studied as experimental subjects. Candidates screened but not selected in the confirmatory
model-selection probe (28 frozen OpenAI/Meta/DeepSeek candidates) are listed in
`results/llm/AMPLIFICATION_CONFIRM_STOP2.md`. {#tbl:models}

<!--
Figure sources (copy alongside paper.tex at typesetting):
  figures/fig3_crossings_c4.png
  figures/llm_stage1_invariants.png
  results/llm/amplification/fig2_amplification_abs_m.png
  results/llm/confirm/fig_binder_step3.png
  results/llm/confirm/results/fig_ruleA.png
Candidates not yet placed: figures/fig4_asymmetry.png, figures/llm_stage1_H4.png, figures/llm_stage2_predictions.png,
  results/llm/amplification/fig3_growth_abs_m.png, fig4_placement_abs_m.png,
  results/llm/confirm/results/fig_step4_amplification.png.
The Substack figures in figures/substack/ are illustrations simulated from the measured tables and are not cited.
-->

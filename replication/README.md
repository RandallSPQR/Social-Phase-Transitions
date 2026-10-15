# Replication package

One command reruns every analysis of the paper from the recorded model responses and checks the result:

```bash
make replicate-fast        # about 55 min on 4 cores
make replicate             # about 37 h on 4 cores (recomputes everything; about 16 h with --skip s1.null)
```

Both modes run the same pipeline:

1. **Fetch** the public Hugging Face dataset `RandallSPQR/social-phase-transitions-llm-cache`: the response cache
   (31 files, 58.5 MB) and, in fast mode, the simulation unit bundles (140 MB). Full mode instead fetches the Gemma
   activations used by the exploratory probes (820 MB).
2. **Verify** every downloaded file against its committed SHA-256: `results/llm/cache_manifest.json`,
   `replication/sim_units_manifest.json`, `replication/activations_manifest.json`.
3. **Rerun** every analysis from the cache. `llm/client.py` runs with `SPT_OFFLINE=1`, so a request missing from the
   cache is an error, never an API call. No API key and no GPU are used.
4. **Regenerate** every figure.
5. **Compare** every regenerated number with the committed results, within the stated tolerances below.
6. **Print a pass/fail table per claim.** It is also written to `replication/out/REPORT.md` (with `report.json`,
   per-step logs in `replication/out/logs/`, and a record of every compared file).

The exit status is 0 only if every claim that ran passes. Every regenerated file that was compared is also kept in
`replication/out/regenerated/` next to the committed copies in `replication/out/reference/`, so any comparison can
be inspected or redone (`python replication/replicate.py --recompare`) after a Docker run has exited.

## Two tiers of replication

| tier | what it reproduces | needs | expected agreement |
|---|---|---|---|
| **1. Exact reproduction from the cache** (`make replicate`, `make replicate-fast`) | every analysis, table, verdict and figure, from the recorded responses | Python 3.13 with `replication/requirements.lock`, or Docker; about 3 GB of free disk (fast) or 4 GB (full); internet access to huggingface.co | bit-identical numbers on the pinned environment (tolerances below) |
| **2. Re-collection** (new API calls and GPU runs) | the data themselves | an OpenRouter API key with the providers pinned as in `llm/endpoints.py`; an A100-class GPU for the self-hosted Gemma-4 runs; about $37 at 2026-10 prices (paper Appendix C) | **not identical.** Providers change serving stacks, quantisation and batching over time (paper §4.2 measured batch effects of 0.86 nats RMS), sampled replies are random, and some model versions may be retired. Expect agreement in distribution, not in digits |

## Quick start

**Docker** (recommended; the image pins Python, every library and ffmpeg):

```bash
make docker-replicate-fast          # or: make docker-replicate
# results in replication/out/REPORT.md
```

**Local** (Python 3.13):

```bash
make env                            # .venv with replication/requirements.lock (hash-checked)
make replicate-fast PY=.venv/bin/python
```

Run on a clean checkout. The steps overwrite result files in place, and the comparison reference is a snapshot of
the committed files taken at the start of the first run (`replication/out/reference/`). Afterwards,
`git status` shows any file that changed. `git checkout -- results figures data` restores the committed versions.
Delete `replication/out/` to start over.

Until the dataset is public, set `HF_TOKEN` to a token with read access.

## Modes, runtime and hardware

Reference machine: 4 vCPU (x86-64), 15 GB RAM, Linux, running the Docker image. The fast-mode figures are measured
in the clean run recorded in `replication/logs/`. The full-mode figures are estimated from the original runs' logs
(`llm/EXECUTION_LOG.md`, `logs/val.log`, `logs/asym.log`); a measured full run is still to be recorded.

| mode | what runs | wall time | peak memory (largest process) | download | free disk |
|---|---|---|---|---|---|
| fast | every analysis; spot checks recompute 1–6 sampled simulation units per family and require them to be bit-identical to the fetched ones | 55 min (measured: 54.6 min from container start to exit, of which 25 s download and verification) | 4.6 GB | 200 MB | 3 GB |
| full | everything in fast mode, plus every simulation unit recomputed from scratch, the baseline physics sweeps, the 14 Amendment 4–5 bootstrap caches, the 100-replicate null distribution of Amendment 4, and the exploratory probes | about 37 h (partly measured, see below): baseline sweeps 2.6 h (measured; the validation sweep uses 2 processes), bootstrap caches 1.3 h (measured), Amendment 4 null distribution about 21 h (measured at 12.5 min per replicate × 100), amplification Stage B 5.5 h, confirmatory Steps 4–5 2.8 h, the rest as in fast mode. `PY=… make replicate ARGS="--skip s1.null"` leaves out the null distribution (the step that dominates) | about 5 GB (estimate) | 1 GB | 4 GB |

Per-step times are in the generated report and in the table below.

## Claims map

Paper claim → script(s) → committed result file(s) → figure(s). Step names refer to `replication/steps.py`. The
headline numbers that the report checks against the paper's printed values are listed in `replication/claims.py`.

| claim (paper section) | steps | scripts | result files | figures |
|---|---|---|---|---|
| **C0** The simulator reproduces the Viana–Bray transitions (6 of 6 within 95% CIs) and the melting of frozen camps under non-reciprocity (§4.1) | base.sim (full), base.tables, base.figures | `sim/run_validation.py`, `sim/run_asym.py`, `sim/analyze.py`, `replication/baseline_tables.py`, `sim/plots.py`, `sim/plot_asym.py` | `data/*.npz`, `data/asym/asym.npy`, `results/validation_crossings.csv`, `results/asymmetry_sweep.csv` | `figures/fig1`–`fig4` |
| **C1** LLM response functions are not Ising-like (H1–H5) (§4.2) | s1.cache, s1.fits, s1.summary, s1.temperature, s1.amendments, s1.null (full), s1.figures | `llm/run_stage1.py` (via `replication/stage1_rematerialise.py`), `llm/analysis.py`, `llm/comprehension.py`, `llm/summarize.py`, `llm/secondary.py`, `llm/bridge.py`, `llm/temp_check.py`, `llm/temp_diag.py`, `llm/compare_local*.py`, `llm/cubic_31b.py`, `llm/batch_effect.py` | `results/llm/stage1_cells*.csv`, `stage1_verdicts*.json`, `stage1_H4*.csv`, `comprehension_*.csv`, `fits/main/*.json`, `temperature_verify*.json`, `amendment4/`, `amendment5/` | `figures/llm_stage1_*.png`, `figures/llm_pilot_llama8b.png` |
| **C2** Trait summaries fail live (surrogate 0 of 4); order-averaged tables succeed per step (§4.3–4.4) | s2.surrogates, s2.sims, s3.replay, s3.audit, k.test, k.avg, k.followups | `llm/voter.py`, `llm/stage2.py`, `llm/stage3.py`, `llm/stage3_evaluate.py`, `llm/stage3_audit*.py`, `llm/kernel_test.py`, `llm/k_avg.py`, `llm/kernel_*.py` | `results/llm/stage2/`, `stage3/evaluation.json`, `stage3/summary.csv`, `stage3_audit/`, `kernel_test/verdict.json`, `kernel_test/k_avg_*.csv`, `kernel_test/k_avg_verdict.json` | `figures/llm_stage2_*.png` |
| **C3** Small per-step differences are amplified collectively (§4.4, §4.7.1) | k.test, k.avg, amp.stageB | `llm/kernel_test.py`, `llm/k_avg.py`, `llm/amplification_curve.py`, `llm/amplification_analyse.py` | `kernel_test/amplification.csv`, `kernel_test/k_avg_amplification.csv`, `amplification/amplification.csv` | `results/llm/amplification/fig2`, `fig4` |
| **C4** Near a tipping point, amplification exceeds controls (Stage B refuted; corrected Rule A 3 of 3 in both models with a transition; GPT-4o-mini below the floor) (§4.7) | s1.temperature, amp.stageB, conf.api, conf.locate, conf.step45 | `llm/amplification_curve.py`, `llm/amplification_analyse.py`, `llm/confirm_api.py`, `llm/confirm_temp_diag.py`, `llm/confirm_sim.py`, `llm/confirm_step45.py`, `llm/confirm_analyse.py` | `amplification/verdict.json`, `confirm/api_summary.json`, `confirm/temp_diag.json`, `confirm/results/step4.json`, `step5.json`, `control_gpt4omini.json` | `results/llm/amplification/fig1`–`fig4`, `results/llm/confirm/results/fig_step4_amplification.png`, `fig_ruleA.png` |
| **C5** Whether a tipping point exists is model-dependent (two clear, GPT-4o-mini none, Gemma unresolved) (§4.8) | conf.api, conf.locate, conf.step45, conf.gemma | `llm/confirm_sim.py`, `llm/confirm_report_step3.py`, `llm/confirm_step45.py`, `llm/gemma_mechanism.py` | `confirm/sim/locate_*.json`, `confirm/results/gemma_exploratory.json`, `confirm/gemma_mechanism/summary.json` | `results/llm/confirm/fig_binder_step3.png` |
| **S1** Residual persistence excess does not replicate (Z = −0.42) (§4.5–4.6) | r.tests, r.replicate | `llm/residual_*.py`, `llm/replicate_predictions.py`, `llm/stage3_replicate_analyse.py`, `llm/stage3_replicate_reanalyse.py`, `llm/replicate_replay_audit.py`, `llm/persistence_effect_combined.py` | `residual/*.json`, `stage3_rep1/reanalysis.json`, `stage3_rep1/replay_audit/`, `stage3_rep1/persistence_effect_combined.json` | — |
| **E1** Exploratory probes on Gemma-4 activations (§4.2; no claim rests on them) | s1.probes (full) | `llm/probes.py`, `llm/probes_followup.py` | `amendment4/probes_*`, `amendment5/probes_followup_*` | — |
| **X** Illustrations (not cited in the analysis) | substack | `llm/substack_visuals.py`, `llm/substack_mdist.py` | `figures/substack/*.json` | `figures/substack/` |

All result paths without a prefix are under `results/llm/`.

## Tolerances

`replication/compare.py` compares every regenerated file with its committed copy. Numbers must satisfy
|new − committed| ≤ atol + rtol·|committed|:

| profile | rtol | atol | used for |
|---|---|---|---|
| exact | 1e-12 | 1e-12 | simulations, tables, kernels, verdicts (deterministic code, seeded per unit by hashing) |
| fit | 1e-6 | 1e-9 | iterative optimiser outputs (Stage 1 fits, local-vs-provider comparisons, probes) |
| boot | 0 | 0.05 | bootstrap CI bounds of the six baseline crossings only (point estimates: 6e-5, i.e. the 4 printed decimals) |
| figure | — | — | the PNG is valid and each pixel dimension is within 2% of the committed one. Figures are saved with a tight bounding box, so installed fonts move the size by a few pixels (up to 18 px, under 1%, between the authoring machine and the clean container); bytes are not compared |
| media | — | — | the MP4/GIF exists and is non-empty |

Strings must match exactly. Order is ignored (profile `rows`) only for the two logs written in completion order: the
K_avg call log and the Stage 3 audit's mock runs.

### What is not compared, and why

A short list of fields cannot be reproduced from the recorded data, by construction. Each is named in
`replication/compare.py` with its reason, and nothing else is skipped:

| field | where | why it cannot be reproduced |
|---|---|---|
| `source`, `path`, `time`, … and absolute-path values | fit JSONs, logs | where or when a file was written |
| `gif_bytes` | `figures/substack/twins_stats.json` | encoded size depends on the ffmpeg build |
| `spend` | `results/llm/confirm/api_summary.json` | the original run's live API spend; a replay from the cache spends $0 |
| `error`, `first_error` (string values only) | `results/llm/confirm/api_summary.json` | live error text of probes that failed in the original run (for example "http 404"). Failed calls were never cached, so the replay reports "not in cache". The probe outcomes are compared |
| `stage1_records_with_same_body_signature` | `results/llm/stage3_audit/logs_checks.json` | counts the records in the cache at the time of the audit; later phases added records of the same format. The check itself (`body_differs_from_battery_body` = 0) is compared |
| `predictions_commit`, `*_changed_since_predictions_commit` | `results/llm/stage3_audit/B11_surrogate.json` | read from the git history, which archives and the Docker image do not carry; recompute in a git clone with `python llm/stage3_audit.py surrogate` |

JSON fields present only in a regenerated file (added by a later version of the producing script) are listed in the
record but not counted as differences; every committed field must be reproduced. Separately, 53 headline numbers are read from the regenerated files and
must equal the values printed in the paper at the paper's rounding.

## What is regenerated, and what is an input

Regenerated from the cache (bit-identical on the pinned environment):
- every Stage 1 CSV from an API endpoint (checked row by row against the committed CSVs; see below);
- the Stage 3 live chains (replayed call by call);
- the K_avg battery;
- the confirmatory batteries and probes;
- every fit, simulation, verdict, table and figure.

Taken as inputs (they cannot be regenerated without paid calls or a GPU; they are committed and covered by the
SHA-256 manifests):
- **Self-hosted Gemma-4 outputs (GPU):** `results/llm/stage1/*/gemma-*-local*.csv` and
  `results/llm/confirm/gemma/*.csv`.
- **The live replicate's chain logs:** `results/llm/stage3_rep1/chains/`. Their replies are in the cache, and
  `llm/replicate_replay_audit.py` replays and checks every call.
- **Two items from before Amendment 2:** the Amendment-2 pilot (`stage1/pilot/llama-8b_instruction-v1.csv`) and the
  original instruction it used. The current battery code builds only the revised instruction.

### One known, explained difference: duplicate requests in the Stage 1 batteries

Some contact permutations render the identical prompt, so the request has the same body and replicate index.

- **What happened:** the original run made both live calls and recorded both answers. The cache, keyed by (body,
  replicate), kept one. Re-materialising therefore reproduces one of the two recorded answers for those rows: 614 of
  12,428 rows per logprob endpoint belong to such groups, and 14–50 of them differ.
- **How the replication handles it:** `replication/stage1_rematerialise.py` checks two things. Every row outside a
  duplicate group must be identical. Every row inside a group must equal one of that group's recorded answers.
- **Result:** 0 rows are unexplained across all 28 files.
- **Fits:** these run on the committed CSVs (the record of every original answer), which this check ties to the
  cache.

## Reproducibility safeguards kept from the study

- **Library pins:** networkx is pinned exactly, because `random_regular_graph` builds different graphs from the same
  seed across versions (EXECUTION_LOG entry 51). The driver refuses to run if networkx, numpy, numba, llvmlite,
  scipy or pandas differ from the lock.
- **Graph fingerprints:** `llm/stage3.py` asserts the committed fingerprints (`results/llm/stage3/graph_hashes.json`)
  before using any Stage 3 graph. The amplification and confirmatory graph fingerprints
  (`amplification/graph_fingerprints.json`, `confirm/sim/locate_*.json`) are regenerated and compared exactly.
- **Seeds:** every simulation unit derives its seeds by hashing its parameters, so results do not depend on process
  scheduling.

## Tier 2: re-collecting the data

Use a separate clone so the tier-1 cache cannot answer the requests:

```bash
export OPENROUTER_API_KEY=...         # never commit it
mv results/llm/cache results/llm/cache.tier1   # requests now go to the API (and are cached anew)
unset SPT_OFFLINE
```

Then run the collection scripts phase by phase, in the order of `replication/steps.py`. Each command in a step that
reads the cache makes the calls on a fresh cache:

| phase | commands | approximate cost (2026-10) |
|---|---|---|
| Stage 1 | `llm/run_stage1.py` per endpoint and phase (arguments: `replication/stage1_rematerialise.py` reads them off the committed CSVs); `llm/temp_check.py` | $9.56 |
| Stage 3 live networks | `llm/stage3.py run` (140,000 calls) | $4.95 |
| K_avg battery | `llm/k_avg.py run` | $0.58 |
| Residual replicate | `llm/stage3_replicate.py run` on a CPU pod (`llm/pod_replicate.sh`; about 40 h) | $5.29 API + $0.9 pod |
| Confirmatory batteries | `llm/confirm_api.py all` | $3.33 |
| Self-hosted Gemma-4 (GPU) | `llm/local_gemma.py`, `llm/local_gemma_confirm.py` on an A100 via `llm/runpod_launch.py` (heartbeat, per-unit uploads and teardown rules in `CLAUDE.md`) | ≈ $12.8 |

Comparisons against the committed results then measure provider drift, not replication error. The pre-registered
analyses can be rerun on the new data with the same step commands.

## Files

| file | purpose |
|---|---|
| `replicate.py` | driver: environment check, fetch, verify, reference snapshot, steps, comparison, report |
| `steps.py` | the steps in dependency order, with the files each one is compared on |
| `claims.py` | claims, headline numbers and the pass/fail table |
| `compare.py` | file comparison and tolerance profiles |
| `stage1_rematerialise.py` | rebuilds every Stage 1 request and checks every committed row against the cache |
| `baseline_tables.py` | regenerates the baseline summary tables from the raw simulation output |
| `fetch_units.py`, `sim_units_manifest.json` | simulation unit bundles (fast mode) |
| `fetch_activations.py`, `activations_manifest.json` | Gemma activations (full mode, probes) |
| `requirements.in`, `requirements.lock` | pinned Python environment (hash-checked) |
| `audit_secrets.py` | credential scan of the git history and of the dataset, run before release |
| `hf_dataset_card/README.md` | dataset card for the Hugging Face dataset |
| `RELEASE_CHECKLIST.md` | the steps for the owner to make the release public |
| `logs/` | the clean-run logs from a fresh container |

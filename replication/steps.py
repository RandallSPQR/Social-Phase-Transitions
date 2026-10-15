"""The replication steps, in dependency order.

Each step is a list of shell commands run from the repository root with SPT_OFFLINE=1 (llm/client.py then raises on a
cache miss instead of calling the API), MPLBACKEND=Agg and PYTHONHASHSEED=0.

mode
  "both"  runs in fast and full mode
  "full"  runs only in full mode (long recomputations)
clean_full
  paths deleted before the step in full mode. The simulation scripts skip every unit whose output file exists, so fast
  mode (unit outputs fetched and SHA-256-verified) reruns only the analysis, while full mode recomputes every unit.
clean
  paths deleted before the step in both modes (resumable caches that would otherwise short-circuit the analysis)
compare
  (glob, profile[, per-column tolerances]) of files to compare with the committed copies after the step
claims
  which paper claims (replication/claims.py) the step feeds
minutes
  wall time measured on the reference machine (4 vCPU, 16 GB; see replication/README.md), for the mode(s) it runs in
"""

LLM = "python llm/"
API_MAIN = ["gemini-lite", "gemma-26b-cw", "gemma-26b", "gemma-31b", "gpt4o-mini", "gpt4o-mini_sampled", "gpt6-luna",
            "haiku", "llama-70b-pa", "llama-70b", "llama-8b", "mistral-large", "mistral-small", "nemo-12b",
            "qwen-122b", "qwen-9b"]
LOCAL_MAIN = ["gemma-26b-local", "gemma-26b-local-bs1", "gemma-31b-local"]
API_COMPREHENSION = ["gemma-26b-cw", "gemma-26b", "gemma-31b", "gpt4o-mini", "llama-70b", "llama-8b", "mistral-large",
                     "mistral-small", "nemo-12b", "qwen-122b", "qwen-9b"]
S1 = "results/llm/stage1"

# bootstrap CI bounds of the baseline crossings: the generator state depends on the unrecorded order in which tags
# were analysed (replication/baseline_tables.py); point estimates are exact
BOOT_CI = {c: (0.0, 0.05) for c in ("scaledchi_lo95", "scaledchi_hi95", "binder_lo95", "binder_hi95")}
BOOT_CI.update({c: (0.0, 6e-5) for c in ("scaledchi_T", "binder_T", "theory_T")})


def step(id, title, cmds, compare=(), mode="both", claims=(), clean=(), clean_full=(), minutes=None, note=""):
    return dict(id=id, title=title, cmds=list(cmds), compare=list(compare), mode=mode, claims=list(claims),
                clean=list(clean), clean_full=list(clean_full), minutes=minutes, note=note)


STEPS = [
    # ------------------------------------------------------------------ baseline physics (paper §3.1, §4.1)
    step("base.sim", "Viana-Bray validation sweep and asymmetry sweep (raw simulation output)",
         ["cd sim && python run_validation.py", "cd sim && python run_asym.py"],
         compare=[("data/*.npz", "exact"), ("data/asym/asym.npy", "exact")],
         mode="full", claims=["C0"], clean_full=["data/*.npz", "data/asym/asym.npy"],
         note="run_validation uses 2 processes (hard-coded); about 3.4 h, asymmetry about 17 min"),
    step("base.tables", "Crossing table and asymmetry table from the raw output",
         ["python replication/baseline_tables.py"],
         compare=[("results/validation_crossings.csv", "exact", BOOT_CI), ("results/asymmetry_sweep.csv", "exact")],
         claims=["C0"]),
    step("base.figures", "Baseline figures 1-4",
         ["cd sim && python plots.py 1 2 3", "cd sim && python plot_asym.py"],
         compare=[("figures/fig[1-4]_*.png", "figure")], claims=["C0"]),

    # ------------------------------------------------------------------ Stage 1 (paper §3.3, §4.2)
    step("s1.cache", "Stage 1: rebuild every API request and check every committed CSV row against the cache",
         ["python replication/stage1_rematerialise.py"],
         compare=[], claims=["C1"],
         note="writes replication/out/stage1/cache_check.json; checked in claims.py"),
    step("s1.fits", "Stage 1: response-function fits (bootstrap B = 1000) for every endpoint",
         [f"{LLM}analysis.py {S1}/main/{k}.csv --B 1000 --out results/llm/fits/main" for k in API_MAIN + LOCAL_MAIN]
         + [f"{LLM}analysis.py {S1}/pilot/llama-8b.csv --B 1000 --out results/llm/fits/pilot"],
         compare=[("results/llm/fits/main/*.json", "fit"), ("results/llm/fits/pilot/llama-8b.json", "fit")],
         claims=["C1"]),
    step("s1.summary", "Stage 1: comprehension gate, hypothesis verdicts H1-H5, secondary contrasts, bridge",
         [f"{LLM}comprehension.py " + " ".join(f"{S1}/comprehension/{k}.csv" for k in API_COMPREHENSION),
          f"{LLM}comprehension.py {S1}/comprehension/gemma-26b-local.csv {S1}/comprehension/gemma-31b-local.csv"
          " --out results/llm/comprehension_local.csv",
          f"{LLM}comprehension.py {S1}/comprehension/gemma-26b-local-bs1.csv --out results/llm/comprehension_local_bs1.csv",
          f"{LLM}summarize.py", f"{LLM}summarize.py --relax-gate", f"{LLM}summarize.py --amendment4",
          f"{LLM}summarize.py --amendment5", f"{LLM}secondary.py " + " ".join(API_MAIN), f"{LLM}bridge.py"],
         compare=[("results/llm/comprehension_summary.csv", "exact"), ("results/llm/comprehension_local*.csv", "exact"),
                  ("results/llm/stage1_cells*.csv", "fit"), ("results/llm/stage1_H4*.csv", "fit"),
                  ("results/llm/stage1_verdicts*.json", "fit"), ("results/llm/stage1_secondary.csv", "fit"),
                  ("results/llm/bridge_comparison.csv", "fit")],
         claims=["C1"]),
    step("s1.temperature", "Stage 1: 1/T temperature checks (re-materialised from the cache) and diagnostic",
         [f"{LLM}temp_check.py semantics llama-8b llama-70b llama-70b-pa qwen-9b qwen-122b gemma-26b gemma-31b nemo-12b "
          "mistral-small mistral-large gpt4o-mini", f"{LLM}temp_check.py verify", f"{LLM}temp_check.py verify qwen-9b",
          f"{LLM}temp_check.py verify qwen-122b", f"{LLM}temp_check.py verify nemo-12b", f"{LLM}temp_diag.py"],
         compare=[("results/llm/temperature_*.csv", "exact"), ("results/llm/temperature_verify*.json", "fit")],
         claims=["C1", "C4"]),
    step("s1.amendments", "Amendments 4-5: local-vs-provider comparison, cubic replication, batch effect",
         [f"{LLM}compare_local.py --B 1000",
          f"{LLM}compare_local.py --B 1000 --out results/llm/amendment5 --pairs gemma-26b-local-bs1:gemma-26b:Parasail "
          "gemma-26b-local-bs1:gemma-26b-cw:CoreWeave",   # as in llm/run_amendment5.sh
          f"{LLM}cubic_31b.py", f"{LLM}batch_effect.py"],
         compare=[("results/llm/amendment4/local_vs_provider*", "fit"), ("results/llm/amendment5/local_vs_provider*", "fit"),
                  ("results/llm/amendment5/cubic_gemma-31b-local.json", "fit"),
                  ("results/llm/amendment5/batch_effect*", "exact")],
         clean_full=["results/llm/amendment4/cache/*.npz", "results/llm/amendment5/cache/*.npz"], claims=["C1"],
         note="the 14 bootstrap caches (B = 1000) take about 1.5 h; fast mode keeps the committed ones and recomputes one"),
    step("s1.null", "Amendment 4: null distribution of the local-vs-provider rule (100 replicates)",
         [f"{LLM}compare_local_null.py --R 100 --B 1000"],
         compare=[("results/llm/amendment4/compare_rule_null_B1000.json", "fit")],
         clean=["results/llm/amendment4/compare_rule_null_runs.jsonl"], mode="full", claims=["C1"]),
    step("s1.probes", "Amendments 4-5 (exploratory): linear probes on Gemma-4 activations",
         ["python replication/fetch_activations.py",
          f"{LLM}probes.py --acts replication/out/activations --key gemma-31b-local",
          f"{LLM}probes.py --acts replication/out/activations --key gemma-26b-local",
          f"{LLM}probes_followup.py --acts replication/out/activations --key gemma-31b-local",
          f"{LLM}probes_followup.py --acts replication/out/activations --key gemma-26b-local"],
         compare=[("results/llm/amendment4/probes_*_summary.json", "fit"), ("results/llm/amendment4/probes_gemma-*-local.csv", "fit"),
                  ("results/llm/amendment5/probes_followup_*", "fit")],
         clean=["results/llm/amendment4/probes_*_partial.csv"], mode="full", claims=["E1"],
         note="downloads about 820 MB of activations; exploratory, no claim rests on it"),
    step("s1.figures", "Stage 1 figures",
         [f"{LLM}plot_stage1.py {S1}/pilot/llama-8b.csv results/llm/fits/pilot/llama-8b.json figures/llm_pilot_llama8b.png --pilot",
          f"{LLM}plot_stage1_summary.py"],
         compare=[("figures/llm_pilot_llama8b.png", "figure"), ("figures/llm_stage1_*.png", "figure")], claims=["C1"]),

    # ------------------------------------------------------------------ Stage 2 (paper §3.4, §4.3)
    step("s2.surrogates", "Stage 2: voter-vs-logit selection and surrogate parameters",
         [f"{LLM}voter.py " + " ".join(f"{S1}/main/{k}.csv" for k in API_MAIN)],
         compare=[("results/llm/fits/surrogate/*.json", "fit"), ("results/llm/voter/*.csv", "fit"),
                  ("results/llm/voter_vs_logit.csv", "fit")], claims=["C2"]),
    step("s2.sims", "Stage 2: surrogate simulations (floor, Stage-3 protocol, finite-size scaling, jitter)",
         [f"{LLM}stage2.py floor", f"{LLM}stage2.py stage3", f"{LLM}stage2.py fss", f"{LLM}stage2.py jitter",
          f"{LLM}stage2_report.py", f"{LLM}plot_stage2.py"],
         compare=[("results/llm/stage2/*.csv", "exact"), ("figures/llm_stage2_*.png", "figure")], claims=["C2"]),

    # ------------------------------------------------------------------ Stage 3 (paper §4.3)
    step("s3.replay", "Stage 3: replay all 70 live chains from the cache, measures, pre-registered evaluation",
         [f"{LLM}stage3.py check", f"{LLM}stage3.py run", f"{LLM}stage3.py measures", f"{LLM}stage3_evaluate.py"],
         compare=[("results/llm/stage3/chains/*/*.csv.gz", "exact"), ("results/llm/stage3/chains/*/*.npy", "exact"),
                  ("results/llm/stage3/measures_by_graph.csv", "exact"), ("results/llm/stage3/summary.csv", "exact"),
                  ("results/llm/stage3/evaluation.*", "exact")],
         clean=["results/llm/stage3/chains/*/*_states.npy"], claims=["C2"]),
    step("s3.audit", "Stage 3 harness audit (logs, mock surrogate through the harness, same-graph diagnostic, order)",
         [f"{LLM}stage3_audit.py logs", f"{LLM}stage3_audit.py mock --reps 40 --workers 4",
          f"{LLM}stage3_audit.py mock --reps 160 --points P1 P3 --workers 4",   # P1, P3 at 160 (STAGE3_AUDIT_RESULTS.md)
          f"{LLM}stage3_audit.py mock-summary", f"{LLM}stage3_audit.py sim-diag", f"{LLM}stage3_audit.py surrogate",
          f"{LLM}stage3_audit.py report", f"{LLM}stage3_audit_order.py"],
         compare=[("results/llm/stage3_audit/*.json", "exact"), ("results/llm/stage3_audit/*.csv", "exact"),
                  ("results/llm/stage3_audit/A_mock_runs.jsonl", "exact"), ("results/llm/stage3_audit/*.md", "text")],
         clean=["results/llm/stage3_audit/A_mock_runs.jsonl"], claims=["C2"]),

    # ------------------------------------------------------------------ kernel test (paper §3.5, §4.4)
    step("k.test", "Kernel test: K_fixed, A3, K_live simulated on the live graphs (R = 400)",
         [f"{LLM}kernel_test.py run --R 400"],
         compare=[("results/llm/kernel_test/sims_*_K_fixed*.csv", "exact"), ("results/llm/kernel_test/sims_*_K_live*.csv", "exact"),
                  ("results/llm/kernel_test/evaluation.csv", "exact"), ("results/llm/kernel_test/kernel_tables.csv", "exact"),
                  ("results/llm/kernel_test/amplification.csv", "exact"), ("results/llm/kernel_test/verdict.json", "exact")],
         claims=["C2", "C3"]),
    step("k.avg", "Kernel test: K_avg battery re-materialised from the cache, simulated (R = 400), verdict",
         [f"{LLM}k_avg.py build", f"{LLM}k_avg.py run", f"{LLM}k_avg.py analyse --R 400"],
         compare=[("results/llm/kernel_test/k_avg_prompts.csv", "exact"), ("results/llm/kernel_test/k_avg_calls.csv", "rows"),
                  ("results/llm/kernel_test/sims_*_K_avg.csv", "exact"), ("results/llm/kernel_test/k_avg_evaluation.csv", "exact"),
                  ("results/llm/kernel_test/k_avg_amplification.csv", "exact"),
                  ("results/llm/kernel_test/k_avg_verdict.json", "exact")],
         claims=["C2", "C3"]),
    step("k.followups", "Kernel test follow-ups: estimator check, hidden state, neighbourhood",
         [f"{LLM}kernel_estimator_check.py", f"{LLM}kernel_hidden_state.py", f"{LLM}kernel_neighbourhood.py"],
         compare=[("results/llm/kernel_test/estimator_check.json", "exact"), ("results/llm/kernel_test/hidden_state.json", "exact"),
                  ("results/llm/kernel_test/neighbourhood.json", "exact")], claims=["C2"]),

    # ------------------------------------------------------------------ residual persistence (paper §4.5-4.6)
    step("r.tests", "Residual persistence: heterogeneity tests, identical requests, repeat rule",
         [f"{LLM}residual_heterogeneity.py", f"{LLM}residual_heterogeneity_c.py", f"{LLM}residual_identical_requests.py",
          f"{LLM}residual_repeat_rule.py"],
         compare=[("results/llm/residual/*.json", "exact"), ("results/llm/residual/*.csv", "exact")], claims=["S1"]),
    step("r.replicate", "Residual persistence: replicate predictions, analysis, reanalysis on actual graphs, replay "
         "audit, combined effect",
         [f"{LLM}replicate_predictions.py", f"{LLM}stage3_replicate.py ncbattery",   # k = 0 battery (32 cached calls) + nc predictions
          f"{LLM}stage3_replicate_analyse.py", f"{LLM}stage3_replicate_reanalyse.py",
          f"{LLM}replicate_replay_audit.py", f"{LLM}persistence_effect_combined.py"],
         compare=[("results/llm/replicate/predictions.json", "exact"), ("results/llm/replicate/k_avg_nc_calls.csv", "exact"),
                  ("results/llm/stage3_rep1/analysis.json", "exact"),
                  ("results/llm/stage3_rep1/reanalysis.json", "exact"), ("results/llm/stage3_rep1/replay_audit/*.json", "exact"),
                  ("results/llm/stage3_rep1/replay_audit/*.csv", "exact"),
                  ("results/llm/stage3_rep1/persistence_effect_combined.json", "exact")], claims=["S1"]),

    # ------------------------------------------------------------------ amplification Stage B (paper §4.7.1)
    step("amp.stageB", "Amplification Stage B: locate T_c, perturbation runs, analysis and figures",
         [f"{LLM}amplification_curve.py locate --R 200", f"{LLM}amplification_curve.py locate2",
          f"{LLM}amplification_curve.py run", f"{LLM}amplification_curve.py analyse"],
         compare=[("results/llm/amplification/*.json", "exact"), ("results/llm/amplification/*.csv", "exact"),
                  ("results/llm/amplification/fig*.png", "figure")],
         clean_full=["results/llm/amplification/locate", "results/llm/amplification/locate2",
                     "results/llm/amplification/perturb"], claims=["C3", "C4"],
         note="full mode about 5.5 h on 4 processes"),

    # ------------------------------------------------------------------ confirmatory test (paper §4.7.2, §4.8)
    step("conf.api", "Confirmatory phase: model-selection probes, comprehension, 1/T checks, K_avg batteries "
         "re-materialised from the cache",
         [f"{LLM}confirm_api.py prompts", f"{LLM}confirm_api.py all", f"{LLM}confirm_temp_diag.py"],
         compare=[("results/llm/confirm/*.csv", "exact"), ("results/llm/confirm/*.json", "fit")],
         claims=["C4", "C5"]),
    step("conf.locate", "Confirmatory phase: locate T_c for every model (hot/cold Binder rule) and Binder figure",
         [f"{LLM}confirm_sim.py locate P1 gemini gemma gemma_exact selected luna", f"{LLM}confirm_sim.py luna_draws",
          f"{LLM}confirm_report_step3.py"],
         compare=[("results/llm/confirm/sim/locate_*.json", "exact"), ("results/llm/confirm/fig_binder_step3.png", "figure")],
         clean_full=["results/llm/confirm/sim/coarse", "results/llm/confirm/sim/fine", "results/llm/confirm/sim/wrule"],
         claims=["C4", "C5"]),
    step("conf.step45", "Confirmatory phase: Steps 4-5 perturbation runs, negative control, Gemma checks, verdicts",
         [f"{LLM}confirm_step45.py run", f"{LLM}confirm_step45.py gemma", f"{LLM}confirm_analyse.py"],
         compare=[("results/llm/confirm/results/*.json", "exact"), ("results/llm/confirm/results/*.csv", "exact"),
                  ("results/llm/confirm/results/*.png", "figure")],
         clean_full=["results/llm/confirm/step45", "results/llm/confirm/sim/gemma_len"], claims=["C4", "C5"],
         note="full mode about 2.8 h on 4 processes"),
    step("conf.gemma", "Gemma mechanism test (exploratory): knockouts, run length, replica overlaps",
         [f"{LLM}gemma_mechanism.py"],
         compare=[("results/llm/confirm/gemma_mechanism/summary.json", "exact")],
         clean_full=["results/llm/confirm/gemma_mechanism/len", "results/llm/confirm/gemma_mechanism/glass"],
         claims=["C5"]),

    # ------------------------------------------------------------------ illustrations (not cited in the analysis)
    step("substack", "Illustrations simulated from the measured tables (figures/substack/)",
         [f"{LLM}substack_visuals.py", f"{LLM}substack_mdist.py"],
         compare=[("figures/substack/*.json", "exact"), ("figures/substack/*.png", "figure"),
                  ("figures/substack/*.mp4", "media"), ("figures/substack/*.gif", "media")], claims=["X"]),
]

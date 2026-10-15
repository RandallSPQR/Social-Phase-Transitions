"""Claims of the paper and how each one is checked; builds the pass/fail table.

A claim passes when (a) every step that feeds it ran without error, (b) every regenerated file it depends on matches
the committed copy within the stated tolerance (replication/compare.py), (c) every recomputed spot-check unit is
bit-identical, and (d) every headline number below, read from the regenerated files, equals the value printed in
the paper at the paper's rounding.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "replication", "out")

CLAIMS = {
    "QC": "The comparator detects the smallest corruption: one flipped byte in a base.sim array and a one-ulp change "
          "in a CSV and a JSON result are each rejected, unmodified copies pass (replication/canary.py)",
    "C0": "Baseline: the simulator reproduces the Viana-Bray transitions (6 of 6 within 95% CIs) and the melting of "
          "frozen camps under non-reciprocity (paper §4.1)",
    "C1": "LLM response functions are not Ising-like: H1 9 of 12 (12 of 15, 15 of 18), H2 supported, H3 2 of 12, "
          "H4 not supported, H5 7 of 12 (paper §4.2)",
    "C2": "Trait summaries fail live (surrogate 0 of 4); order-averaged tables succeed per step (K_avg |dP| <= 0.004) "
          "(paper §4.3-4.4)",
    "C3": "Small per-step differences are amplified collectively (K_fixed vs K_live 3.4 SD from |dP| 0.002) "
          "(paper §4.4, §4.7.1)",
    "C4": "Near a tipping point amplification exceeds controls: Stage B refuted; corrected Rule A 3 of 3 shapes for "
          "both models with a transition; GPT-4o-mini below floor (paper §4.7)",
    "C5": "Whether a tipping point exists is model-dependent: two clear transitions, GPT-4o-mini none, Gemma "
          "unresolved (paper §4.8)",
    "S1": "Residual persistence excess does not replicate (Stouffer Z = -0.42) (paper §4.5-4.6)",
    "E1": "Exploratory probes on Gemma-4 activations (paper §4.2; no claim rests on them; full mode only)",
    "X": "Illustrations in figures/substack/ (not part of the analysis)",
}


# ---------------------------------------------------------------- headline numbers (value printed in the paper)
def _j(p):
    return json.load(open(os.path.join(ROOT, p)))


def headline_checks():
    """(claim, description, value printed in the paper, function returning the regenerated value, decimals)."""
    return HEADLINES


def _csv(p):
    import pandas as pd
    return pd.read_csv(os.path.join(ROOT, p))


def _vc(tag, col):
    d = _csv("results/validation_crossings.csv")
    return float(d[(d.tag == tag) & (d.N1 == 1000)][col].iloc[0])


def _vc_within():
    d = _csv("results/validation_crossings.csv"); d = d[d.N1 == 1000]
    return int(((d.theory_T >= d.scaledchi_lo95) & (d.theory_T <= d.scaledchi_hi95)).sum())


def _asym(rho, T, eps, col, N=4000):   # the paper quotes the N = 4,000 rows
    d = _csv("results/asymmetry_sweep.csv")
    return float(d[(d.rho == rho) & (d["T"] == T) & (d.eps == eps) & (d.N == N)][col].iloc[0])


def _cells(f, col, cond=None):
    d = _csv(f"results/llm/{f}"); d = d[d.confirmatory == True]  # noqa: E712
    return int((d[col] < 0.05).sum()) if cond == "holm" else int(d[col].sum())


def _kavg(point, col):
    d = _csv("results/llm/kernel_test/k_avg_amplification.csv").set_index("point")
    return float(d.loc[point, col])


def _step4(model, shape, key):
    return _j("results/llm/confirm/results/step4.json")[model]["abs_m"][shape][key]


def _rule_a_count(model):
    return sum(bool(_step4(model, s, "rule_A")) for s in ("stay", "majority", "noise"))


def _ctrl_max(shape):
    return max(r["A1600"] for r in _j("results/llm/confirm/results/control_gpt4omini.json")["abs_m"][shape]["rows"])


def _gemma_ko(kernel):
    import pandas as pd
    g = pd.DataFrame(_j("results/llm/confirm/gemma_mechanism/summary.json")["runlength"])
    g = g[(g.kernel == kernel) & (g.N == 1600)]
    return float((g.hot.mean() + g.cold.mean()) / 2)


def _same_n(N, T, col):
    d = _csv("results/llm/amplification/amplification.csv")
    d = d[(d.proto == "eq") & (d.point == "P1") & (d["shape"] == "fixed") & (d["size"] == 1.0) & (d.measure == "abs_m")
          & (d.N == N) & ((d["T"] - T).abs() < 1e-3)]
    return float(d[col].iloc[0])


HEADLINES = [
    # C0 baseline (paper Table 4.1 and text)
    ("C0", "transitions within their 95% CI of theory (of 6)", 6, _vc_within, 0),
    ("C0", "consensus rho=0: simulated T", 3.917, lambda: _vc("f_r00", "scaledchi_T"), 3),
    ("C0", "consensus rho=0.2: simulated T", 2.156, lambda: _vc("f_r20", "scaledchi_T"), 3),
    ("C0", "glass c=4: simulated T", 1.813, lambda: _vc("sg_c4", "scaledchi_T"), 3),
    ("C0", "glass c=6: simulated T", 2.233, lambda: _vc("sg_c6", "scaledchi_T"), 3),
    ("C0", "glass persistence at eps=0 (N = 4000)", 0.51, lambda: _asym(0.5, 1.0, 0.0, "persistence"), 2),
    ("C0", "glass persistence at eps=0.1", 0.28, lambda: _asym(0.5, 1.0, 0.1, "persistence"), 2),
    ("C0", "glass persistence at eps=0.2", 0.11, lambda: _asym(0.5, 1.0, 0.2, "persistence"), 2),
    ("C0", "glass persistence at eps=0.35", 0.01, lambda: _asym(0.5, 1.0, 0.35, "persistence"), 2),
    ("C0", "consensus |m| at eps=0", 0.62, lambda: _asym(0.1, 2.0, 0.0, "abs_m"), 2),
    ("C0", "consensus |m| at eps=1", 0.55, lambda: _asym(0.1, 2.0, 1.0, "abs_m"), 2),
    # C1 Stage 1 verdicts
    ("C1", "H1 supported (of 12 confirmatory cells)", 9, lambda: _cells("stage1_cells.csv", "H1_supported"), 0),
    ("C1", "H1 supported with Amendment 4 (of 15)", 12, lambda: _cells("stage1_cells_amendment4.csv", "H1_supported"), 0),
    ("C1", "H1 supported with Amendment 5 (of 18)", 15, lambda: _cells("stage1_cells_amendment5_rescue.csv", "H1_supported"), 0),
    ("C1", "H2 letter field (of 12)", 10, lambda: _cells("stage1_cells.csv", "H2_letter_holm", "holm"), 0),
    ("C1", "H2 content field (of 8 content cells)", 8, lambda: _cells("stage1_cells.csv", "H2_content_holm", "holm"), 0),
    ("C1", "H3 pairwise supported (of 12)", 2, lambda: _cells("stage1_cells.csv", "H3_pairwise_supported"), 0),
    ("C1", "H3 cubic supported (of 12)", 0, lambda: _cells("stage1_cells.csv", "H3_cubic_supported"), 0),
    ("C1", "H5 departs from summation (of 12)", 7, lambda: _cells("stage1_cells.csv", "H5_departs_from_sum"), 0),
    ("C1", "H1 gate-relaxed (of 21)", 18, lambda: _cells("stage1_cells_sensitivity_gate_relaxed.csv", "H1_supported"), 0),
    ("C1", "H4 verdict", "not supported",
     lambda: _j("results/llm/stage1_verdicts.json")["H4_verdict"].split(" (")[0], None),
    # C2 Stage 3 and kernel test
    ("C2", "surrogate: P1-P4 points passing both primary measures", 0,
     lambda: _j("results/llm/stage3/evaluation.json")["P1_P4_points_passing_both_primary"], 0),
    ("C2", "kernel test: K_avg points passing (of P1-P4)", 2,
     lambda: _j("results/llm/kernel_test/k_avg_verdict.json")["K_avg"]["P1_P4_passing"], 0),
    ("C2", "kernel test: A3 points passing (of P1-P4)", 1,
     lambda: _j("results/llm/kernel_test/verdict.json")["verdict"]["K_fixed-A3"]["P1_P4_passing"], 0),
    ("C2", "K_avg vs K_live |dP| at P1", 0.004, lambda: _kavg("P1", "K_avg_vs_K_live_dP"), 3),
    ("C2", "K_avg vs K_live |dP| at P4", 0.003, lambda: _kavg("P4", "K_avg_vs_K_live_dP"), 3),
    ("C2", "K_avg vs K_live |dlogit| at P4 (nats)", 0.07, lambda: _kavg("P4", "K_avg_vs_K_live_dlogit_nats"), 2),
    # C3 amplification of per-step differences
    ("C3", "K_fixed vs K_live |dP| at P3", 0.002, lambda: _kavg("P3", "K_fixed_vs_K_live_dP"), 3),
    ("C3", "K_fixed vs K_live persistence difference at P3 (SD)", -3.4,
     lambda: float(_csv("results/llm/kernel_test/amplification.csv").set_index("point").loc["P3", "K_fixed_z_kernel_persistence"]), 1),
    ("C3", "same-N table: per nat at T_c, N = 1600", 1.27, lambda: _same_n(1600, 1.9346, "A"), 2),
    ("C3", "same-N table: per nat at T = 1, N = 100", 0.009, lambda: _same_n(100, 1.0, "A"), 3),
    # C4 amplification near tipping points
    ("C4", "Stage B: T_c of P1", 1.93, lambda: _j("results/llm/amplification/verdict.json")["Tc_P1"], 2),
    ("C4", "Stage B: H1 growth G, noise shape", 5.5,
     lambda: _j("results/llm/amplification/verdict.json")["abs_m"]["per_shape"]["noise"]["G"], 1),
    ("C4", "Stage B: pre-registered verdict (H2 for majority shift)", False,
     lambda: _j("results/llm/amplification/verdict.json")["abs_m"]["per_shape"]["majority"]["H2"], None),
    ("C4", "confirmatory: Rule A passes for P1 (of 3 shapes)", 3, lambda: _rule_a_count("P1"), 0),
    ("C4", "confirmatory: Rule A passes for Gemini (of 3 shapes)", 3, lambda: _rule_a_count("gemini"), 0),
    ("C4", "confirmatory: D, P1 stay-bias", 0.15, lambda: _step4("P1", "stay", "D"), 2),
    ("C4", "confirmatory: D, P1 noise", 2.03, lambda: _step4("P1", "noise", "D"), 2),
    ("C4", "confirmatory: D, Gemini majority", 1.03, lambda: _step4("gemini", "majority", "D"), 2),
    ("C4", "confirmatory: Rule B growth G, P1 noise", 11.7, lambda: _step4("P1", "noise", "G"), 1),
    ("C4", "GPT-4o-mini control: max |m| amplification, stay", 0.075, lambda: _ctrl_max("stay"), 3),
    ("C4", "GPT-4o-mini control: max |m| amplification, noise", 0.096, lambda: _ctrl_max("noise"), 3),
    # C5 model-dependence
    ("C5", "T_c of P1 (confirmatory)", 2.01, lambda: _j("results/llm/confirm/sim/locate_P1.json")["Tc"], 2),
    ("C5", "T_c of Gemini", 1.83, lambda: _j("results/llm/confirm/sim/locate_gemini.json")["Tc"], 2),
    ("C5", "GPT-4o-mini: transition located", None, lambda: _j("results/llm/confirm/sim/locate_selected.json").get("Tc"), None),
    ("C5", "GPT-4o-mini: highest Binder cumulant at N = 1600", 0.47,
     lambda: max(_j("results/llm/confirm/sim/locate_selected.json")["binder_coarse"]["1600"]), 2),
    ("C5", "Gemma-31B: transition located (pre-registered verdict)", None,
     lambda: _j("results/llm/confirm/sim/locate_gemma.json").get("Tc"), None),
    ("C5", "Gemma rival knockout |m| (N = 1600)", 0.77, lambda: _gemma_ko("rival_ko"), 2),
    ("C5", "Gemma additive projection |m| (N = 1600)", 0.09, lambda: _gemma_ko("additive"), 2),
    # S1 residual persistence
    ("S1", "replicate: Stouffer Z (actual graphs)", -0.42,
     lambda: _j("results/llm/stage3_rep1/reanalysis.json")["primary"]["stouffer_Z"], 2),
    ("S1", "combined effect, original run", 0.0116,
     lambda: _j("results/llm/stage3_rep1/persistence_effect_combined.json")["original"]["raw"]["estimate"], 4),
    ("S1", "combined effect, pooled", 0.0033,
     lambda: _j("results/llm/stage3_rep1/persistence_effect_combined.json")["pooled"]["raw"]["estimate"], 4),
    ("S1", "heterogeneity between runs, Cochran Q", 7.6,
     lambda: _j("results/llm/stage3_rep1/persistence_effect_combined.json")["heterogeneity_between_runs"]["Q"], 1),
]


def _check(h):
    claim, desc, paper, fn, nd = h
    try:
        got = fn()
    except Exception as e:  # missing file, key error: the check fails and says why
        return dict(claim=claim, what=desc, paper=paper, got=f"error: {type(e).__name__}: {e}"[:120], ok=False)
    if isinstance(paper, (int, float)) and not isinstance(paper, bool) and nd is not None:
        ok = got is not None and round(float(got), nd) == round(float(paper), nd)
        shown = None if got is None else round(float(got), nd + 2)
    else:
        ok = got == paper; shown = got
    return dict(claim=claim, what=desc, paper=paper, got=shown, ok=bool(ok))


# ---------------------------------------------------------------- report
def report(records, mode, env):
    import steps as S
    heads = [_check(h) for h in HEADLINES]
    s1 = os.path.join(OUT, "stage1", "cache_check.json")
    cache_ok = None
    if "s1.cache" in records:
        try:
            cc = json.load(open(s1)); cache_ok = all(r["ok"] for r in cc.values())
        except Exception:
            cache_ok = False
        records["s1.cache"]["ok"] = records["s1.cache"]["ok"] and bool(cache_ok)
    rows = []
    for c, text in CLAIMS.items():
        st = [r for r in records.values() if c in r["claims"]]
        hc = [h for h in heads if h["claim"] == c]
        ran = bool(st)
        ok = ran and all(r["ok"] for r in st) and all(h["ok"] for h in hc)
        nfiles = sum(len(r["compared"]) for r in st)
        nbad = sum(sum(not x["ok"] for x in r["compared"]) for r in st)
        nspot = sum(len(r["spot"]) for r in st)
        rows.append(dict(claim=c, text=text, ok=ok, ran=ran, steps=[r["id"] for r in st], files=nfiles, files_differ=nbad,
                         spot=nspot, headlines=f"{sum(h['ok'] for h in hc)}/{len(hc)}",
                         minutes=round(sum(r["seconds"] for r in st) / 60, 1)))
    out = dict(mode=mode, env=env, claims=rows, headlines=heads, steps=list(records.values()),
               all_ok=all(r["ok"] for r in rows if r["ran"]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    L = [f"# Replication report ({mode} mode)", "",
         f"Python {env.get('python')} on {env.get('platform')}, {env.get('cpus')} CPUs.", "",
         "| claim | result | steps | files compared | files differing | units recomputed | headline numbers | minutes |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        res = "PASS" if r["ok"] else ("not run" if not r["ran"] else "FAIL")
        L.append(f"| {r['claim']} | **{res}** | {', '.join(r['steps'])} | {r['files']} | {r['files_differ']} | {r['spot']} "
                 f"| {r['headlines']} | {r['minutes']} |")
    L += ["", "Claims:", ""] + [f"- **{c}** {t}" for c, t in CLAIMS.items()]
    L += ["", "## Headline numbers (regenerated value vs the value printed in the paper)", "",
          "| claim | quantity | paper | regenerated | ok |", "|---|---|---|---|---|"]
    for h in heads:
        L.append(f"| {h['claim']} | {h['what']} | {h['paper']} | {h['got']} | {'yes' if h['ok'] else '**NO**'} |")
    L += ["", "## Steps", "", "| step | result | minutes | files | differing | spot units | peak RSS so far (MB) | exit |",
          "|---|---|---|---|---|---|---|---|"]
    for r in records.values():
        L.append(f"| {r['id']} | {'PASS' if r['ok'] else '**FAIL**'} | {r['seconds'] / 60:.1f} | {len(r['compared'])} | "
                 f"{sum(not x['ok'] for x in r['compared'])} | {len(r['spot'])} | {r.get('peak_child_rss_mb', '')} | {r['exit']} |")
    L += ["", f"Total step time: {sum(r['seconds'] for r in records.values()) / 60:.1f} min."]
    bad = [(r["id"], x) for r in records.values() for x in r["compared"] + r["spot"] if not x["ok"]]
    if bad:
        L += ["", "## Differences", ""]
        for sid, x in bad[:200]:
            L.append(f"- `{x['file']}` ({sid}, profile {x['profile']}): {'; '.join(x['probs'][:3])}")
    if cache_ok is not None:
        L += ["", f"Stage 1 cache check (replication/out/stage1/cache_check.json): {'all rows explained' if cache_ok else 'FAILED'}."]
    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(L) + "\n")
    w = max(len(r["claim"]) for r in rows)
    print("\n" + "=" * 100)
    print(f"{'claim':<{w}}  {'result':<8} {'files':>6} {'differ':>6} {'units':>6} {'headline':>9}  description")
    for r in rows:
        res = "PASS" if r["ok"] else ("not run" if not r["ran"] else "FAIL")
        print(f"{r['claim']:<{w}}  {res:<8} {r['files']:>6} {r['files_differ']:>6} {r['spot']:>6} {r['headlines']:>9}  {r['text'][:60]}")
    print("=" * 100)
    print("report:", os.path.relpath(os.path.join(OUT, "REPORT.md"), ROOT))
    return 0 if out["all_ok"] else 1

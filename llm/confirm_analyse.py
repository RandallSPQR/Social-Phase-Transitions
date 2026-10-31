"""Confirmatory amplification phase: Step 4-5 + amendment 3 analysis (plan §5-6, A3). Written before the results exist.

  python llm/confirm_analyse.py

Rule A (decisive, per model x shape): D = A_model(T_c, 1600) - max(A_P2, A_P4)(T_c, 1600); pass iff D >= 0.10 and the
  95% bootstrap CI of D excludes 0 (max taken inside each resample).
Rule B: G = A(T_c, 1600) / A(T_c, 100), tested only if A >= 0.10 at both N; pass iff the CI lower bound > 1.
Rule C: argmax_T A(T, 1600) over the Step 4 band within [T_c/1.1, 1.1 T_c] (+ bootstrap share).
Persistence: same rules, secondary. GPT-4o-mini: negative control (A3.2). Step 5 rule (§6). Gemma exploratory (A3.3).
Outputs: results/llm/confirm/results/ (json, csv, figures).
"""
import glob, json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amplification_analyse as AA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results/llm/confirm/step45"); SIM = os.path.join(ROOT, "results/llm/confirm/sim")
OUT = os.path.join(ROOT, "results/llm/confirm/results")
FLOOR, MARGIN = 0.10, 0.10
SHAPES = ["stay", "majority", "noise", "fixed"]
CONF = ["P1", "gemini"]; EXPL = ["luna"]
LABEL = {"P1": "P1 Qwen-122B (fresh)", "gemini": "Gemini 3.1 Flash-Lite", "luna": "GPT-6-luna (exploratory)", "selected": "GPT-4o-mini (neg. control)",
         "P2": "P2 control", "P4": "P4 control"}
UNVERIFIED = "defined decision-noise rescaling, not verified sampling temperature"


def ci(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [float("nan")] * 2


def load(step):
    rows, draws = [], {}
    for f in sorted(glob.glob(os.path.join(IN, step, "*.csv.gz"))):
        df = pd.read_csv(f); m, N, T, W = df.model.iloc[0], int(df.N.iloc[0]), float(df["T"].iloc[0]), int(df.W.iloc[0])
        r, d = AA.unit_stats(df, AA.A.seed("confirm-boot", step, m, N, T, W))
        rows += [dict(model=m, N=N, T=T, W=W, **x) for x in r]
        for k, v in d.items():
            draws[(m, N, T, W) + k] = v
    return pd.DataFrame(rows), draws


def size_of(shape):
    return 1.0 if shape == "fixed" else 0.1


def step4(D, dr):
    loc = {m: json.load(open(os.path.join(SIM, f"locate_{m}.json"))) for m in CONF + EXPL}
    res = {}
    for m in CONF + EXPL:
        band = loc[m]["step4_band"]; tc = band[6]; W = loc[m]["step4"]["W"]; r = {"T_c": loc[m]["Tc"], "T_c_ci": loc[m]["Tc_ci"], "band": band, "W": W,
                                                                               "role": "confirmatory" if m in CONF else "exploratory (" + UNVERIFIED + ")"}
        for X in ("abs_m", "persistence"):
            per = {}
            for sh in SHAPES:
                sz = size_of(sh); key = lambda mm, N, T=tc: (mm, N, T, W, sh, sz, X)
                if key(m, 1600) not in dr:
                    continue
                a_m = dr[key(m, 1600)]; a_c = {c: dr.get(key(c, 1600)) for c in ("P2", "P4")}
                q = D.set_index(["model", "N", "T", "W", "shape", "size", "measure"]).A
                pt = lambda mm, N, T=tc: float(q.get((mm, N, T, W, sh, sz, X), np.nan))
                s = dict(A_model=pt(m, 1600), A_model_ci=ci(a_m))
                if sh != "fixed" and all(v is not None for v in a_c.values()):
                    mx = np.maximum(a_c["P2"], a_c["P4"]); Dd = a_m - mx
                    s.update(A_P2=pt("P2", 1600), A_P4=pt("P4", 1600), D=s["A_model"] - max(pt("P2", 1600), pt("P4", 1600)), D_ci=ci(Dd))
                    s["rule_A"] = bool(s["D"] >= MARGIN and s["D_ci"][0] > 0)
                elif sh == "fixed":
                    s["rule_A"] = None; s["note"] = "real error shape: controls have no comparable shape; reported, not in Rule A"
                a100 = pt(m, 100)
                s["A_N100"] = a100
                if s["A_model"] >= FLOOR and a100 >= FLOOR:
                    g = a_m / dr[key(m, 100)]; s.update(G=s["A_model"] / a100, G_ci=ci(g)); s["rule_B"] = bool(s["G_ci"][0] > 1)
                else:
                    s["rule_B"] = "not tested (amplification below floor)"
                curve = {T: pt(m, 1600, T) for T in band}; pk = max(curve, key=curve.get)
                boots = np.array([dr[(m, 1600, T, W, sh, sz, X)] for T in band]); arg = np.array(band)[boots.argmax(0)]
                s.update(peak_T=pk, rule_C=bool(tc / 1.1 <= pk <= tc * 1.1), rule_C_boot_share=float(np.mean((arg >= tc / 1.1) & (arg <= tc * 1.1))),
                         curve_1600={str(T): v for T, v in curve.items()})
                if ("selected", 1600, tc, 500, sh, sz, X) in dr and sh != "fixed":
                    s["vs_gpt4omini_D"] = s["A_model"] - float(q.get(("selected", 1600, tc, 500, sh, sz, X), np.nan))
                    s["vs_gpt4omini_D_ci"] = ci(a_m - dr[("selected", 1600, tc, 500, sh, sz, X)])
                per[sh] = s
            r[X] = per
            npass = sum(1 for sh in ("stay", "majority", "noise") if per.get(sh, {}).get("rule_A"))
            r[X + "_summary"] = f"amplification at the tipping point exceeds the controls for {npass} of 3 synthetic shapes"
        res[m] = r
    return res


def control_selected(D, dr):
    loc = {m: json.load(open(os.path.join(SIM, f"locate_{m}.json"))) for m in CONF}
    Ts = sorted(set(loc["P1"]["step4_band"] + loc["gemini"]["step4_band"])); q = D.set_index(["model", "N", "T", "W", "shape", "size", "measure"]).A; out = {}
    for X in ("abs_m", "persistence"):
        per = {}
        for sh in ("stay", "majority", "noise"):
            tested, rows = [], []
            for T in Ts:
                a16, a1 = float(q.get(("selected", 1600, T, 500, sh, 0.1, X), np.nan)), float(q.get(("selected", 100, T, 500, sh, 0.1, X), np.nan))
                row = dict(T=T, A1600=a16, A100=a1)
                if a16 >= FLOOR and a1 >= FLOOR:
                    g = dr[("selected", 1600, T, 500, sh, 0.1, X)] / dr[("selected", 100, T, 500, sh, 0.1, X)]; row.update(G=a16 / a1, G_ci=ci(g))
                    tested.append(row["G_ci"][0] <= 1 <= row["G_ci"][1])
                rows.append(row)
            a = np.array([r["A1600"] for r in rows])
            per[sh] = dict(rows=rows, n_tested=len(tested), prediction_holds=(all(tested) if tested else "below floor everywhere"),
                           max_over_median=float(np.nanmax(a) / np.nanmedian(a)), any_above_floor=bool(np.nanmax(a) >= FLOOR), max_A1600=float(np.nanmax(a)))
        out[X] = per
    return out


def step5(D, dr):
    q = D.set_index(["model", "N", "T", "W", "shape", "size", "measure"]).A
    Ts = sorted(D["T"].unique()); out = {"label": UNVERIFIED}; Gmax = {}
    for W in (500, 1000, 2000):
        for sh in ("stay", "majority", "noise"):
            best, rows = None, []
            for T in Ts:
                a16, a1 = float(q.get(("P1", 1600, T, W, sh, 0.1, "abs_m"), np.nan)), float(q.get(("P1", 100, T, W, sh, 0.1, "abs_m"), np.nan))
                row = dict(T=T, A1600=a16, A100=a1)
                if a1 >= FLOOR:
                    g = dr[("P1", 1600, T, W, sh, 0.1, "abs_m")] / dr[("P1", 100, T, W, sh, 0.1, "abs_m")]; row.update(G=a16 / a1, G_ci=ci(g))
                    if best is None or row["G"] > best["G"]:
                        best = row
                rows.append(row)
            out[f"W{W}:{sh}"] = dict(rows=rows, max_G=best)
            Gmax[(W, sh)] = best
    ok = []
    for sh in ("stay", "majority", "noise"):
        rows4 = out[f"W2000:{sh}"]["rows"]; growth4 = any(r.get("G_ci", [0])[0] > 1 for r in rows4)
        g1, g4 = Gmax[(500, sh)], Gmax[(2000, sh)]
        keep = bool(g1 and g4 and g4["G"] >= 0.5 * g1["G"])
        out[f"persist:{sh}"] = dict(growth_at_4x=growth4, maxG_1x=g1 and g1["G"], maxG_4x=g4 and g4["G"], not_halved=keep)
        ok.append(growth4 and keep)
    out["verdict"] = "dynamical transition" if sum(ok) >= 2 else "finite-time effect"
    return out


def gemma():
    out = {"label": "exploratory (Gemma-31B has no transition in the swept range)"}
    L = []
    for f in glob.glob(os.path.join(SIM, "gemma_len", "*.csv.gz")):
        b = os.path.basename(f)[:-7].split("_"); N, T, s, W = int(b[1][1:]), float(b[2][1:]), b[3], int(b[4][1:])
        L.append(dict(N=N, T=T, start=s, W=W, abs_m=pd.read_csv(f).abs_m.mean()))
    L = pd.DataFrame(L)
    if len(L):
        g = L.pivot_table(index=["N", "T", "W"], columns="start", values="abs_m").reset_index(); g["gap"] = (g.c - g.h).abs()
        out["runlength"] = g.round(4).to_dict(orient="records")
        g16 = g[g.N == 1600].pivot_table(index="T", columns="W", values="gap")
        out["start_dependence_persists"] = bool(((g16[4000] > 0.02) & (g16[4000] >= 0.5 * g16[1000])).any())
    B = [pd.read_csv(f) for f in glob.glob(os.path.join(IN, "gemma_basin", "*.csv.gz"))]
    if B:
        B = pd.concat(B); basin = lambda x: np.where(np.abs(x) < 0.05, 0, np.sign(x)); rows = []
        for (N, T), g in B.groupby(["N", "T"]):
            b0 = basin(g.base.values)
            for k in ["stay:0.1", "majority:0.1", "noise0:0.1", "ref_otherseed"]:
                n = len(g); x = int((basin(g[k].values) != b0).sum()); p = x / n; z = 1.96
                c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
                rows.append(dict(N=int(N), T=float(T), what=k, changed=x, n=n, rate=p, ci=[c - h, c + h], undecided_base=float((b0 == 0).mean())))
        out["outcome_sensitivity"] = rows
    return out


# ---------------------------------------------------------------- figures
def figs(D, res, ctl):
    plt = AA.style(); q = D.set_index(["model", "N", "T", "W", "shape", "size", "measure"]).A
    models = CONF + EXPL
    fig, axs = plt.subplots(4, len(models), figsize=(11, 10), sharex=True, sharey=True)
    for j, m in enumerate(models):
        r = res[m]; tc, band, W = r["band"][6], r["band"], r["W"]
        for i, sh in enumerate(SHAPES):
            ax = axs[i, j]; sz = size_of(sh); x = np.array(band) / tc
            y = [q.get((m, 1600, T, W, sh, sz, "abs_m"), np.nan) for T in band]
            ax.plot(x, y, color=AA.ACC, lw=1.6)
            ax.plot([1], [q.get((m, 100, tc, W, sh, sz, "abs_m"), np.nan)], "o", mfc="none", color=AA.ACC, ms=5)
            if sh != "fixed":
                for c, mk in (("P2", "s"), ("P4", "^")):
                    ax.plot([1], [q.get((c, 1600, tc, W, sh, sz, "abs_m"), np.nan)], mk, color="#999", ms=4, alpha=0.8)
                if m in CONF:
                    ys = [q.get(("selected", 1600, T, 500, sh, sz, "abs_m"), np.nan) for T in band]
                    ax.plot(x, ys, color="#999", lw=1.0, alpha=0.5)
            ax.axvline(1, color=AA.ACC, lw=0.5, alpha=0.4); ax.axhline(FLOOR, color="#ccc", lw=0.6, ls="--")
            ax.set_yscale("log"); ax.set_ylim(3e-3, 10)
            if i == 0:
                ax.set_title(LABEL[m], loc="left", fontsize=9.5, color="#333")
            if j == 0:
                ax.set_ylabel({"stay": "stay-bias", "majority": "majority-shift", "noise": "random noise", "fixed": "real: K_fixed − K_avg"}[sh] + "\n|Δ|m|| per nat", fontsize=8.5)
            if i == 3:
                ax.set_xlabel("T / T_c")
            if i == 0 and j == 0:
                ax.annotate("N = 1,600", (x[3], y[3]), xytext=(-48, 12), textcoords="offset points", fontsize=7.5, color=AA.ACC)
                ax.annotate("GPT-4o-mini", (x[-1], ys[-1]), xytext=(-40, -12), textcoords="offset points", fontsize=7.5, color="#999")
    AA.head(fig, "Amplification of a 0.1-nat per-step error into |m|, near each model's tipping point",
            "Red line: test model at N = 1,600 (open circle: N = 100 at T_c); grey square/triangle: controls P2/P4 at T_c; grey line: GPT-4o-mini; dashed = 0.10 floor",
            "Fresh random-regular d = 4 graphs, common random numbers, R = 400 (luna 200); luna's axis is a defined decision-noise rescaling. Source: results/llm/confirm/results/",
            y=(0.965, 0.945))
    fig.subplots_adjust(top=0.92); fig.savefig(os.path.join(OUT, "fig_step4_amplification.png")); plt.close(fig)
    # Rule A forest
    fig, ax = plt.subplots(figsize=(8, 3.6)); rows = []
    for m in models:
        for sh in ("stay", "majority", "noise"):
            s = res[m]["abs_m"].get(sh)
            if s and "D" in s:
                rows.append((f"{LABEL[m].split(' (')[0]} · {sh}", s["D"], s["D_ci"], m in CONF))
    for k, (lab, d, c, conf) in enumerate(rows[::-1]):
        col = AA.ACC if conf else "#999"
        ax.plot(c, [k, k], color=col, lw=1.4); ax.plot([d], [k], "o", color=col, ms=4)
    ax.axvline(0, color="#ccc", lw=0.6); ax.axvline(MARGIN, color="#ccc", lw=0.6, ls="--")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8); ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("D = A(model) − max(A(P2), A(P4)) at T_c, N = 1,600 (|m| per nat)")
    AA.head(fig, "Rule A: amplification beyond the worst-case control", "Pass if D ≥ 0.10 (dashed) and the 95% CI excludes 0; red = confirmatory, grey = exploratory",
            "Bootstrap over runs (1,000); max over controls taken inside each resample. Source: results/llm/confirm/results/step4.json")
    fig.savefig(os.path.join(OUT, "fig_ruleA.png")); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    D4, d4 = load("step4"); D4.to_csv(os.path.join(OUT, "step4_amplification.csv"), index=False)
    res = step4(D4, d4); ctl = control_selected(D4, d4)
    D5, d5 = load("step5"); D5.to_csv(os.path.join(OUT, "step5_amplification.csv"), index=False); s5 = step5(D5, d5)
    gm = gemma()
    json.dump(res, open(os.path.join(OUT, "step4.json"), "w"), indent=1, default=float)
    json.dump(ctl, open(os.path.join(OUT, "control_gpt4omini.json"), "w"), indent=1, default=float)
    json.dump(s5, open(os.path.join(OUT, "step5.json"), "w"), indent=1, default=float)
    json.dump(gm, open(os.path.join(OUT, "gemma_exploratory.json"), "w"), indent=1, default=float)
    figs(D4, res, ctl)
    for m in CONF + EXPL:
        print(m, res[m]["role"], res[m]["abs_m_summary"])
        for sh, s in res[m]["abs_m"].items():
            print("  ", sh, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in ("A_model", "A_P2", "A_P4", "D", "D_ci", "rule_A", "G", "G_ci", "rule_B", "peak_T", "rule_C")})
    print("GPT-4o-mini control:", {sh: (v["prediction_holds"], v["n_tested"], round(v["max_A1600"], 3)) for sh, v in ctl["abs_m"].items()})
    print("Step 5:", s5["verdict"], {k: v for k, v in s5.items() if k.startswith("persist")})
    print("Gemma:", gm.get("start_dependence_persists"))


if __name__ == "__main__":
    main()

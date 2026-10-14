"""Substack visuals: kymographs, network animations and twin runs, simulated from each model's MEASURED order-averaged
response table (K_avg). No API calls. Output: figures/substack/.

  python llm/substack_visuals.py            # everything (simulate, statistics, render)

FIXED IN ADVANCE (committed before any rendering):
- Graph: ONE random-regular graph, d = 4, N = 400, rival ties with probability rho = 0.25, seed `substack-graph`,
  networkx 3.7 (fingerprint written to figures/substack/graph.json). The same graph for every model.
- Kernels: confirm_sim.table(model, T) = the measured K_avg with each per-prompt letter logit divided by T (decision
  noise, not sampling temperature). Models: P1 Qwen3.5-122B, Gemini 3.1 Flash-Lite, GPT-4o-mini, Gemma-4 31B.
- Temperatures: P1 and Gemini at 0.8 T_c, T_c, 1.25 T_c (T_c from results/llm/confirm/sim/locate_*.json);
  GPT-4o-mini and Gemma at T = 1, 2, 3. Gemma from a hot start (random 50/50) and a cold start (all A), side by side.
- Run length: 1,000 sweeps from the start for every run (>= every model's confirmed equilibration length: 500 for
  P1, Gemini, GPT-4o-mini; 1,000 for Gemma). Random-sequential updates; one sweep = N updates.
- Seeds: pair j = 0..49 uses start seed `substack-start|j` and dynamics seed `substack-dyn|j`, identical across models
  and temperatures. Single-run kymographs and the single-network animations use pair j = 0.
- Twins (P1, Gemini): run A = the kernel; run B = the kernel + 0.1 nats majority-shift (logit + 0.1 * sign of the
  socially implied direction h = (allies P - allies Q) - (rivals P - rivals Q)), same start and dynamics seed
  (common random numbers).
  * Final state of a run = sign of mean m over the last 100 sweeps; |mean m| < 0.05 = "undecided".
  * Twins "end in different states" iff their final-state labels differ. Rate over the 50 pairs, Wilson 95% CI.
  * Divergence d_j = mean over the last 100 sweeps of the fraction of agents whose positions differ between twins.
  * DISPLAYED PAIR: at T_c, sort the 50 pairs by d_j and take the 25th (lower median, rank 25 of 50, ties broken by
    j). That pair is used in every panel and animation of that model.
- Agent ordering in kymographs: the Fiedler vector (2nd-smallest eigenvector) of the unsigned graph Laplacian.
  Network layout: spectral embedding (Laplacian eigenvectors 2 and 3), fixed for the graph.
- Colours: position A = #4e79a7, position B = #f28e2b; background #fffff8; serif type.
"""
import json, os, subprocess, sys, hashlib
import numpy as np
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S, kernel_test as K, amplification_curve as A, confirm_sim as C

ROOT = K.ROOT
OUT = os.path.join(ROOT, "figures", "substack")
CA, CB, BG = "#4e79a7", "#f28e2b", "#fffff8"
N, SWEEPS, LAST, NPAIRS, DELTA = 400, 1000, 100, 50, 0.1
NAME = {"P1": "Qwen3.5-122B", "gemini": "Gemini 3.1 Flash-Lite", "selected": "GPT-4o-mini", "gemma": "Gemma-4 31B"}
CAPTION = ("Simulated from each model's measured response table (order-averaged one-shot battery), not live LLM calls. "
           "T is decision noise (every logit divided by T), not literal sampling temperature. "
           "Random-regular network, 400 agents, 4 contacts each, 25% rival ties; the same network for every model.")
seed = A.seed


def tc(m):
    return json.load(open(os.path.join(ROOT, f"results/llm/confirm/sim/locate_{m}.json")))["Tc"]


def temps(m):
    return [0.8 * tc(m), tc(m), 1.25 * tc(m)] if m in ("P1", "gemini") else [1.0, 2.0, 3.0]


def graph():
    ip, nb, tie = S.make_graph(N, 0.25, 0.0, np.random.default_rng(seed("substack-graph")), "rrg", d=4)
    return ip, nb, tie


def logit_table(m, T, shift=0.0):
    L = A.lg(np.nan_to_num(C.table(m, T), nan=0.5))
    if shift:
        L = L + shift * A.social_sign()
    return np.clip(L, -A.CLIP, A.CLIP)


@njit(cache=True)
def run_record(ip, nb, tie, L, start, dyn_seed, sweeps):
    """States after every sweep (sweeps + 1, N) as int8, from the given start, random-sequential updates."""
    n = ip.shape[0] - 1
    s = start.copy(); out = np.empty((sweeps + 1, n), np.int8); out[0] = s
    np.random.seed(dyn_seed); P = 1.0 / (1.0 + np.exp(-L))
    for t in range(sweeps):
        for _u in range(n):
            i = np.random.randint(n); u = np.random.random(); c = 0
            for q in range(ip[i], ip[i + 1]):
                if tie[q] > 0:
                    c += 125 if s[nb[q]] > 0 else 25
                else:
                    c += 5 if s[nb[q]] > 0 else 1
            c = c * 2 + (1 if s[i] > 0 else 0)
            s[i] = 1 if u < P[c] else -1
        out[t + 1] = s
    return out


def hot_start(j):
    return np.where(np.random.default_rng(seed("substack-start", j)).random(N) < 0.5, 1, -1).astype(np.int64)


def final_label(states):
    mm = states[-LAST:].mean()
    return 0 if abs(mm) < 0.05 else int(np.sign(mm))


def wilson(x, n, z=1.96):
    p = x / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


# ---------------------------------------------------------------- simulation
def simulate(G):
    ip, nb, tie = G; runs = {}
    for m in ("P1", "gemini", "selected", "gemma"):
        for T in temps(m):
            L = logit_table(m, T)
            runs[(m, round(T, 4), "hot")] = run_record(ip, nb, tie, L, hot_start(0), seed("substack-dyn", 0), SWEEPS)
            if m == "gemma":
                runs[(m, round(T, 4), "cold")] = run_record(ip, nb, tie, L, np.ones(N, np.int64), seed("substack-dyn", 0), SWEEPS)
    twins, stats = {}, {}
    for m in ("P1", "gemini"):
        for T in temps(m):
            La, Lb = logit_table(m, T), logit_table(m, T, DELTA); rows = []
            for j in range(NPAIRS):
                a = run_record(ip, nb, tie, La, hot_start(j), seed("substack-dyn", j), SWEEPS)
                b = run_record(ip, nb, tie, Lb, hot_start(j), seed("substack-dyn", j), SWEEPS)
                rows.append(dict(j=j, la=final_label(a), lb=final_label(b), d=float((a[-LAST:] != b[-LAST:]).mean())))
                twins[(m, round(T, 4), j)] = (a, b)
            x = sum(r["la"] != r["lb"] for r in rows)
            stats[(m, round(T, 4))] = dict(model=m, T=T, T_over_Tc=T / tc(m), different=x, n=NPAIRS, rate=x / NPAIRS, ci=wilson(x, NPAIRS),
                                         mean_divergence=float(np.mean([r["d"] for r in rows])), rows=rows)
        r = stats[(m, round(tc(m), 4))]["rows"]
        order = sorted(r, key=lambda z: (z["d"], z["j"])); stats[("display", m)] = order[NPAIRS // 2 - 1]["j"]   # rank 25 of 50
    return runs, twins, stats


# ---------------------------------------------------------------- layout helpers
def spectral(G):
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    ip, nb, tie = G; rows = np.repeat(np.arange(N), np.diff(ip))
    Adj = sp.csr_matrix((np.ones(len(nb)), (rows, nb)), shape=(N, N)); Lap = sp.diags(np.asarray(Adj.sum(1)).ravel()) - Adj
    w, v = np.linalg.eigh(Lap.toarray())
    return np.argsort(v[:, 1]), v[:, 1:3]


def style():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino", "Georgia", "DejaVu Serif"], "font.size": 10,
                         "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": "#cccccc",
                         "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": "#999",
                         "ytick.color": "#999", "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelcolor": "#666",
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.15})
    return plt


def kymo(ax, axm, st, order, plt, title=None, tmax=SWEEPS):
    from matplotlib.colors import ListedColormap
    img = st[:tmax + 1, order].T
    ax.imshow(img, aspect="auto", cmap=ListedColormap([CB, CA]), vmin=-1, vmax=1, interpolation="nearest", extent=[0, tmax, N, 0])
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    m = np.abs(st.mean(1))[:tmax + 1]
    axm.plot(np.arange(len(m)), m, color="#333", lw=0.8); axm.set_ylim(0, 1); axm.set_xlim(0, tmax)
    axm.set_xticks([]); axm.set_yticks([0, 1]); axm.spines["bottom"].set_visible(False)
    if title:
        axm.set_title(title, loc="left", fontsize=8.5, color="#333", pad=2)


def foot(fig, text, y=-0.01):
    fig.text(0.01, y, text, fontsize=7.5, color="#888", ha="left", va="top", family="sans-serif", wrap=True)


def head(fig, t, s, y=(1.0, 0.975)):
    fig.text(0.01, y[0], t, fontsize=15, color="#111", ha="left", va="bottom"); fig.text(0.01, y[1], s, fontsize=10, color="#666", ha="left", va="bottom")


# ---------------------------------------------------------------- figures
def fig_kymographs(runs, order, plt):
    rows = [("P1", "hot"), ("gemini", "hot"), ("selected", "hot"), ("gemma", "hot"), ("gemma", "cold")]
    fig = plt.figure(figsize=(11, 12.5)); gs = fig.add_gridspec(len(rows) * 2, 3, height_ratios=[1, 4] * len(rows), hspace=0.35, wspace=0.06)
    for i, (m, start) in enumerate(rows):
        for k, T in enumerate(temps(m)):
            axm = fig.add_subplot(gs[2 * i, k]); ax = fig.add_subplot(gs[2 * i + 1, k])
            lab = (({0.8: "T = 0.8·T_c", 1.0: "T = T_c", 1.25: "T = 1.25·T_c"}[round(T / tc(m), 2)] + f" ({T:.2f})") if m in ("P1", "gemini") else f"T = {T:.0f}")
            kymo(ax, axm, runs[(m, round(T, 4), start)], order, plt, title=lab)
            if k == 0:
                ax.set_ylabel(f"{NAME[m]}" + (f"\n{start} start" if m == "gemma" else ""), fontsize=9, color="#333")
            if i < len(rows) - 1:
                ax.set_xticks([])
            elif k == 1:
                ax.set_xlabel("time (sweeps; one sweep = every agent updates once on average)")
    head(fig, "Who holds which position, over time, in four simulated populations",
         "Each row of pixels is one agent (blue = A, orange = B); thin line above each panel = |m(t)|, the size of the majority", y=(0.915, 0.903))
    foot(fig, CAPTION + " Agents are ordered on the vertical axis by the graph's Fiedler vector (spectral ordering), so neighbours tend to sit close together. "
         "Qwen and Gemini at 0.8, 1 and 1.25 × their tipping point T_c; GPT-4o-mini and Gemma at T = 1, 2, 3 (neither has a tipping point in range). "
         "Gemma is shown from a random (hot) and an all-A (cold) start. One run per panel, seed fixed in advance (pair 0).", y=0.075)
    fig.savefig(os.path.join(OUT, "kymographs.png"), dpi=150); plt.close(fig)


def fig_twins(twins, stats, order, plt):
    fig = plt.figure(figsize=(14, 7)); gs = fig.add_gridspec(4, 6, height_ratios=[1, 4, 1, 4], hspace=0.4, wspace=0.08)
    for i, m in enumerate(("P1", "gemini")):
        j = stats[("display", m)]
        for k, T in enumerate(temps(m)):
            a, b = twins[(m, round(T, 4), j)]
            for h, (st, lab) in enumerate(((a, "A: table"), (b, "B: table + 0.1-nat error"))):
                axm = fig.add_subplot(gs[2 * i, 2 * k + h]); ax = fig.add_subplot(gs[2 * i + 1, 2 * k + h])
                s = stats[(m, round(T, 4))]; tl = {0.8: "0.8·T_c", 1.0: "T_c", 1.25: "1.25·T_c"}[round(T / tc(m), 2)]
                kymo(ax, axm, st, order, plt, title=(f"{tl}   A: table" if h == 0 else f"B: + 0.1-nat error   ({s['different']}/{s['n']} pairs end differently)"))
                if 2 * k + h == 0:
                    ax.set_ylabel(NAME[m], fontsize=9, color="#333")
                ax.set_xticks([] if i == 0 else [0, 500, 1000])
    head(fig, "Identical twins, one with a tiny error",
         "Same network, same start, same random draws; run B's agents lean 0.1 nats more toward their contacts' side. Counts over 50 seed pairs per T", y=(0.95, 0.925))
    foot(fig, CAPTION + " Agents ordered by the Fiedler vector. The pair shown is the median-divergence pair at T_c (rank 25 of 50, rule fixed before rendering), "
         "used in every panel of that model; the counts are over all 50 seed pairs at each T.", y=0.05)
    fig.savefig(os.path.join(OUT, "twins_kymographs.png"), dpi=150); plt.close(fig)
    # statistics
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for m, col, mk in (("P1", "#333", "o"), ("gemini", "#888", "s")):
        xs = [stats[(m, round(T, 4))]["T_over_Tc"] for T in temps(m)]; ys = [stats[(m, round(T, 4))]["rate"] for T in temps(m)]
        lo = [stats[(m, round(T, 4))]["ci"][0] for T in temps(m)]; hi = [stats[(m, round(T, 4))]["ci"][1] for T in temps(m)]
        ax.vlines(xs, lo, hi, color=col, lw=1); ax.plot(xs, ys, mk + "-", color=col, ms=4, lw=1)
        ax.annotate(NAME[m], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=col, va="center")
    ax.set_xticks([0.8, 1.0, 1.25]); ax.set_xticklabels(["0.8·T_c", "T_c", "1.25·T_c"]); ax.set_ylim(0, 1); ax.set_ylabel("share of 50 twin pairs\nending in different states")
    head(fig, "How often a 0.1-nat error changes the outcome", "Final state = sign of the mean majority over the last 100 sweeps (|m| < 0.05 = undecided); bars = 95% Wilson CI",
         y=(1.04, 0.99))
    foot(fig, CAPTION, y=-0.08)
    fig.savefig(os.path.join(OUT, "twins_outcome_rate.png"), dpi=150); plt.close(fig)


def animate(frames_states, pos, G, path, titles, plt, fps=15, seconds=12, width_px=600):
    """frames_states: list of (sweeps+1, N) arrays shown side by side."""
    from matplotlib.collections import LineCollection
    from matplotlib.animation import FFMpegWriter
    ip, nb, tie = G; rows = np.repeat(np.arange(N), np.diff(ip)); keep = rows < nb
    segs = [[pos[a], pos[b]] for a, b in zip(rows[keep], nb[keep])]
    k = len(frames_states); w_in = width_px / 100
    fig, axs = plt.subplots(1, k, figsize=(w_in, w_in / k + 0.6), dpi=100); axs = np.atleast_1d(axs); scs, txs = [], []
    for ax, ttl in zip(axs, titles):
        ax.add_collection(LineCollection(segs, colors="#bbbbbb", linewidths=0.25, alpha=0.5))
        sc = ax.scatter(pos[:, 0], pos[:, 1], s=9 if k == 1 else 6, c=[CA] * N, linewidths=0); scs.append(sc)
        txs.append(ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9, color="#333"))
        ax.set_title(ttl, fontsize=9, color="#333", loc="left"); ax.set_axis_off(); ax.set_aspect("equal")
        pad = 0.05 * (pos.max(0) - pos.min(0)); ax.set_xlim(pos[:, 0].min() - pad[0], pos[:, 0].max() + pad[0]); ax.set_ylim(pos[:, 1].min() - pad[1], pos[:, 1].max() + pad[1])
    fig.text(0.01, 0.005, "Simulated from measured response tables, not live LLM calls; T = decision noise.", fontsize=6.5, color="#888", family="sans-serif")
    idx = np.linspace(0, SWEEPS, fps * seconds).astype(int)
    wr = FFMpegWriter(fps=fps, bitrate=1800, extra_args=["-pix_fmt", "yuv420p"])
    with wr.saving(fig, path + ".mp4", dpi=100):
        for t in idx:
            for st, sc, tx in zip(frames_states, scs, txs):
                s = st[t]; sc.set_color(np.where(s[:, None] > 0, np.array([[0x4e, 0x79, 0xa7]]) / 255, np.array([[0xf2, 0x8e, 0x2b]]) / 255))
                tx.set_text(f"sweep {t:4d}   |m| = {abs(s.mean()):.2f}")
            wr.grab_frame()
    plt.close(fig)
    for colors in (48, 32, 16):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path + ".mp4", "-vf",
                        f"fps={fps},split[a][b];[a]palettegen=max_colors={colors}[p];[b][p]paletteuse=dither=none", path + ".gif"], check=True)
        if os.path.getsize(path + ".gif") <= 8e6:
            break
    return os.path.getsize(path + ".gif")


def main():
    os.makedirs(OUT, exist_ok=True); G = graph(); order, pos = spectral(G)
    fp = hashlib.sha256(np.concatenate(G).astype(np.int64).tobytes()).hexdigest()[:16]
    runs, twins, stats = simulate(G)
    out = {"graph_fingerprint": fp, "N": N, "sweeps": SWEEPS, "T_c": {m: tc(m) for m in ("P1", "gemini")},
           "display_pair": {m: stats[("display", m)] for m in ("P1", "gemini")},
           "twins": [{k: v for k, v in s.items() if k != "rows"} for key, s in stats.items() if key[0] != "display"],
           "twin_rows": {f"{s['model']}@{s['T']:.4f}": s["rows"] for key, s in stats.items() if key[0] != "display"}}
    json.dump(out, open(os.path.join(OUT, "twins_stats.json"), "w"), indent=1)
    plt = style()
    fig_kymographs(runs, order, plt); fig_twins(twins, stats, order, plt)
    sizes = {}
    for m, T in (("P1", tc("P1")), ("gemini", tc("gemini")), ("selected", 2.0), ("gemma", 2.0)):
        lab = f"{NAME[m]}, T = " + (f"T_c ({T:.2f})" if m in ("P1", "gemini") else "2.0 (no tipping point)")
        sizes[m] = animate([runs[(m, round(T, 4), "hot")]], pos, G, os.path.join(OUT, f"network_{m}"), [lab], plt)
    for m in ("P1", "gemini"):
        T = tc(m); j = stats[("display", m)]; a, b = twins[(m, round(T, 4), j)]
        sizes["twins_" + m] = animate([a, b], pos, G, os.path.join(OUT, f"twins_{m}"),
                                      [f"{NAME[m]} at T_c: A (table)", "B (table + 0.1-nat error)"], plt, width_px=800)
    out["gif_bytes"] = sizes; json.dump(out, open(os.path.join(OUT, "twins_stats.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("graph_fingerprint", "display_pair", "gif_bytes")}, indent=1))
    for s in out["twins"]:
        print(s["model"], f"{s['T_over_Tc']:.2f} Tc", f"{s['different']}/{s['n']}", [round(x, 2) for x in s["ci"]], f"mean div {s['mean_divergence']:.3f}")


if __name__ == "__main__":
    main()

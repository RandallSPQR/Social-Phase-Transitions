"""Stage 3 audit, post-hoc diagnostic for C (not in the plan): live P(A) vs Stage 1 battery P(A) split by whether the
live call listed its contacts in the same order as the battery prompt for that cell.   python llm/stage3_audit_order.py
Output: results/llm/stage3_audit/C_order_matched.json."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'llm')
import stage3, stage2, stage3_audit as A
from endpoints import E
cl = lambda p: np.clip(np.log(np.clip(p, 1e-12, 1) / np.clip(1 - p, 1e-12, 1)), -6, 6)
out = {}
for pid in ["P1", "P2", "P3", "P4", "P5"]:
    key, framing, rho, contacts = stage3.POINTS[pid]; model, tag, mode, roff, _ = E[key]
    recs, _ = A.load_cache(model)
    d1 = pd.read_csv(f"results/llm/stage1/main/{key}.csv", dtype={"counts": str, "seq": str})
    d1 = d1[(d1.arm == stage2.FRAME_ARM[framing]) & (d1.perm == 0)]; d1["counts"] = d1.counts.str.zfill(4)
    bat = d1.groupby(["counts", "s0", "mapping", "order"]).agg(pA=("pA", "mean"), seq=("seq", "first"))
    rows = []
    for g in range(5):
        for r in (0, 1):
            df = pd.read_csv(f"results/llm/stage3/chains/{pid}/g{g}_r{r}.csv.gz", dtype={"counts": str}); df["counts"] = df.counts.str.zfill(4)
            for row in df.itertuples(index=False):
                rec = recs[f"stage3:{pid}:{g}:{r}:{row.sweep}:{row.step}:{row.attempts}"]
                P_ = A.parse_prompt(rec["body"]["messages"][0]["content"], framing)
                lP = "A" if row.mapping == 0 else "B"
                listed = "".join(("a" if al else "r") + ("P" if le == lP else "Q") for al, le in P_["contacts"])
                counts = tuple(int(x) for x in row.counts); s0 = row.s0; mp = row.mapping
                if framing == "neutral" and mp == 1:          # relabel P<->Q, as check_prompt
                    counts = (counts[1], counts[0], counts[3], counts[2]); s0 = -s0; mp = 0
                    listed = listed.replace("P", "x").replace("Q", "P").replace("x", "Q")
                b = bat.loc[("".join(map(str, counts)), s0, mp, row.order)]
                rows.append(dict(match=listed == b.seq, live_pA=row.pA, live_A=int(row.letter == "A"), battery_pA=b.pA))
    D = pd.DataFrame(rows); res = {}
    for k, G in D.groupby("match"):
        x = dict(n=len(G), obs_A=G.live_A.mean(), battery=G.battery_pA.mean(),
                 mean_abs_diff_prob=float((G.live_A - G.battery_pA).abs().mean()) if mode != "lp" else float((G.live_pA - G.battery_pA).abs().mean()))
        if mode == "lp":
            x["rms_logit"] = float(np.sqrt(np.mean((cl(G.live_pA.values) - cl(G.battery_pA.values)) ** 2)))
        res["same_order_as_battery" if k else "different_order"] = x
    out[pid] = res; print(pid, json.dumps(res, default=float), flush=True)
json.dump(out, open("results/llm/stage3_audit/C_order_matched.json", "w"), indent=1, default=float)

"""Replay audit of the option B replicate (plan: llm/REPLAY_AUDIT_PLAN.md). Free; never sends a request.

  python llm/replicate_replay_audit.py

Each of the 70 chains is replayed by the unchanged harness (stage3.chain, run_tag "rep1") on its recorded graph, with a
cache-only call keyed by the exact request (body + tag). Checks R1-R7 of the plan; output
results/llm/stage3_rep1/replay_audit/{audit.json, boundaries.csv, audit_table.md}.
"""
import collections, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client, stage3
from endpoints import E

ROOT = stage3.ROOT
REP = os.path.join(ROOT, "results/llm/stage3_rep1")
OUT = os.path.join(REP, "replay_audit")
STREAM_MODELS = sorted({E[v[0]][0] for v in stage3.POINTS.values()})
LOGCOLS = ["sweep", "step", "agent", "counts", "s0", "new", "attempts", "invalid_first", "kept_after_invalid", "mapping", "order", "letter"]


def load_cache():
    """All replicate records: key -> list of records (in file order), tag -> set of keys."""
    by_key, by_tag = collections.defaultdict(list), collections.defaultdict(set)
    for m in STREAM_MODELS:
        with open(client._path(m)) as f:
            for l in f:
                if '"rep":"stage3-rep1:' in l:
                    r = json.loads(l); by_key[r["key"]].append(r); by_tag[r["rep"]].add(r["key"])
    return by_key, by_tag


def main():
    os.makedirs(OUT, exist_ok=True)
    cg = json.load(open(os.path.join(REP, "graphs/chain_graph.json")))
    starts = sorted((json.loads(l)["t"], json.loads(l).get("pod") or json.loads(l).get("host")) for l in open(os.path.join(REP, "interruptions.jsonl")))
    st_t = np.array([s[0] for s in starts]); st_host = [s[1] for s in starts]
    by_key, by_tag = load_cache()
    print("replicate records:", sum(len(v) for v in by_key.values()), "distinct keys:", len(by_key), "tags:", len(by_tag), flush=True)
    orig_graph = stage3.graph
    used_keys = set(); rows, bnd = [], []
    c = collections.Counter()
    for pid, (key, framing, rho, contacts) in stage3.POINTS.items():
        model, tag, mode, roff, _ = E[key]
        prov_expected = {"alibaba": "Alibaba", "parasail/bf16": "Parasail", "coreweave/fp16": "CoreWeave", "io-net/fp16": "Io Net", "openai": "OpenAI"}[tag]
        for g in range(stage3.N_GRAPHS):
            for r in (0, 1):
                z = np.load(os.path.join(REP, "graphs", f"{pid.split('-')[0]}_g{g}_{cg[f'{pid}:{g}:{r}']}.npz"))
                gr = (z["ip"], z["nb"], z["tie"])
                stage3.graph = lambda p, gg, check=True, gr=gr: gr
                seq = []; miss = []; redrawn = 0; prov_bad = 0; pin_bad = 0

                def call(body, rep):
                    nonlocal redrawn, prov_bad, pin_bad
                    k = client.request_key(body, rep)
                    recs = by_key.get(k)
                    if not recs:
                        miss.append(rep); raise KeyError("cache miss " + rep)
                    if len({json.dumps(x["resp"].get("first") or x["resp"].get("content"), sort_keys=True) for x in recs}) > 1:
                        redrawn += 1
                    rec = recs[-1]                     # the client index keeps the last occurrence
                    used_keys.add(k); seq.append((rec["t"], rep))
                    if rec["resp"].get("provider") != prov_expected: prov_bad += 1
                    pv = rec["body"].get("provider", {})
                    if pv.get("order") != [tag] or pv.get("allow_fallbacks") is not False: pin_bad += 1
                    return rec
                ok_replay = True
                try:
                    df, st = stage3.chain(pid, g, r, call=call, run_tag="rep1")
                except KeyError:
                    ok_replay = False
                d0 = pd.read_csv(os.path.join(REP, "chains", pid, f"g{g}_r{r}.csv.gz"), dtype={"counts": str})
                s0 = np.load(os.path.join(REP, "chains", pid, f"g{g}_r{r}_states.npy"))
                # R3
                keys = list(zip(d0.sweep, d0.step))
                r3 = len(d0) == 2000 and len(set(keys)) == 2000 and keys == [(t, u) for t in range(20) for u in range(100)] and s0.shape == (21, 100)
                # R2
                if ok_replay:
                    df["counts"] = df["counts"].astype(str).str.zfill(4); d0["counts"] = d0["counts"].str.zfill(4)
                    log_eq = all((df[k_].astype(str).values == d0[k_].astype(str).values).all() for k_ in LOGCOLS)
                    pa_eq = np.allclose(df.pA.astype(float).values, d0.pA.astype(float).values, equal_nan=True, rtol=0, atol=1e-12)
                    st_eq = bool((st.astype(np.int8) == s0).all())
                    r2 = bool(log_eq and pa_eq and st_eq)
                else:
                    r2 = False
                # R4: genuine resume points = where the chain's running-max segment increases (it moves on to a later
                # host for good); calls whose segment is below the running max are replies REUSED from an earlier host's
                # identical request (same prompt text and tag => same request key), counted separately.
                ts = np.array([t for t, _ in seq]); segs = np.searchsorted(st_t, ts, side="right") - 1
                runmax = np.maximum.accumulate(segs) if len(segs) else segs
                reused_n = int((segs < runmax).sum()) if len(segs) else 0
                for j in np.flatnonzero(np.diff(runmax) > 0) if len(segs) else []:
                    a, b = seq[j][1].split(":"), seq[j + 1][1].split(":")
                    bnd.append(dict(chain=f"{pid}:{g}:{r}", before=f"sweep {a[4]} step {a[5]} att {a[6]}", after=f"sweep {b[4]} step {b[5]} att {b[6]}",
                                    host_before=st_host[segs[j]], host_after=st_host[segs[j + 1]],
                                    consecutive=(int(b[4]) * 100 + int(b[5]) - (int(a[4]) * 100 + int(a[5]))) in (0, 1),
                                    time_gap_min=round((ts[j + 1] - ts[j]) / 60, 1)))
                # R6
                inv_log = int(d0.invalid_first.sum()); res_log = int((d0.attempts - 1).sum()); kept = int(d0.kept_after_invalid.sum())
                calls_used = len(seq)
                rows.append(dict(chain=f"{pid}:{g}:{r}", point=pid, graph=cg[f"{pid}:{g}:{r}"], replay_ok=ok_replay, misses=len(miss),
                                 R2=r2, R3=bool(r3), redrawn_keys=redrawn, provider_bad=prov_bad, pin_bad=pin_bad,
                                 calls_used=calls_used, calls_expected=2000 + res_log, invalid_first=inv_log, resampled=res_log, kept=kept,
                                 segments_spanned=int(len(set(segs.tolist()))) if len(segs) else 0, reused_earlier_replies=reused_n))
                print(rows[-1], flush=True)
    stage3.graph = orig_graph
    R = pd.DataFrame(rows); B = pd.DataFrame(bnd)
    R.to_csv(os.path.join(OUT, "chains.csv"), index=False); B.to_csv(os.path.join(OUT, "boundaries.csv"), index=False)
    all_recs = sum(len(v) for v in by_key.values()); orphan = [k for k in by_key if k not in used_keys]
    orphan_cost = sum(((x["resp"].get("usage") or {}).get("cost") or 0.0) for k in orphan for x in by_key[k])
    dup_records = sum(len(v) - 1 for v in by_key.values())
    res = dict(
        R1=dict(passed=bool(R.misses.sum() == 0 and R.replay_ok.all()), misses=int(R.misses.sum()), chains_failed=R[~R.replay_ok].chain.tolist()),
        R2=dict(passed=bool(R.R2.all()), chains_failed=R[~R.R2].chain.tolist()),
        R3=dict(passed=bool(R.R3.all()), chains_failed=R[~R.R3].chain.tolist()),
        R4=dict(passed=bool((B.consecutive.all() if len(B) else True) and R.redrawn_keys.sum() == 0 and R.R2.all()),
                resume_points=int(len(B)), chains_with_resume=int(B.chain.nunique()) if len(B) else 0,
                reused_earlier_replies=int(R.reused_earlier_replies.sum()), chains_with_reuse=int((R.reused_earlier_replies > 0).sum()),
                nonconsecutive=int((~B.consecutive).sum()) if len(B) else 0, redrawn_keys=int(R.redrawn_keys.sum())),
        R5=dict(passed=bool(R.provider_bad.sum() == 0 and R.pin_bad.sum() == 0), provider_bad=int(R.provider_bad.sum()), pin_bad=int(R.pin_bad.sum())),
        R6=dict(passed=bool((R.calls_used == R.calls_expected).all()),
                per_point=R.groupby("point")[["invalid_first", "resampled", "kept"]].sum().to_dict(orient="index")),
        R7=dict(passed=bool(R.R2.all()), note="R2 includes the logged configuration (counts) of every update, recomputed on the recorded graph"),
        orphaned=dict(records=int(sum(len(by_key[k]) for k in orphan)), distinct_keys=len(orphan), cost_usd=round(orphan_cost, 4),
                      duplicate_records_same_key=int(dup_records), total_records=int(all_recs)))
    json.dump(res, open(os.path.join(OUT, "audit.json"), "w"), indent=1, default=str)
    md = ["| check | result | evidence |", "|---|---|---|"]
    ev = {"R1": f"{res['R1']['misses']} cache misses over {int(R.calls_used.sum())} replayed calls",
          "R2": f"{int(R.R2.sum())}/70 chains reproduce log, pA and states exactly",
          "R3": f"{int(R.R3.sum())}/70 chains with exactly 2,000 updates, complete (sweep, step), 21 states",
          "R4": f"{res['R4']['resume_points']} resume points in {res['R4']['chains_with_resume']} chains, all consecutive (non-consecutive {res['R4']['nonconsecutive']}); re-drawn keys {res['R4']['redrawn_keys']}; {res['R4']['reused_earlier_replies']} replies in {res['R4']['chains_with_reuse']} chains reused from an earlier host's identical request",
          "R5": f"provider mismatches {res['R5']['provider_bad']}, pin violations {res['R5']['pin_bad']}",
          "R6": f"calls used = 2,000 + resamples in every chain: {bool((R.calls_used == R.calls_expected).all())}; " + "; ".join(f"{p} inv {v['invalid_first']} res {v['resampled']} kept {v['kept']}" for p, v in res['R6']['per_point'].items()),
          "R7": "configurations recomputed on the recorded graph match the log in every update (part of R2)"}
    for k_ in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        md.append(f"| {k_} | {'PASS' if res[k_]['passed'] else 'FAIL'} | {ev[k_]} |")
    open(os.path.join(OUT, "audit_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print(json.dumps(res["orphaned"]))


if __name__ == "__main__":
    main()

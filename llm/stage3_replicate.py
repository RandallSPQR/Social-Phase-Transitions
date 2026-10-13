"""Fresh live replicate of Stage 3, option B (llm/RESIDUAL_TEST_PLAN.md + amendment 1).

  python llm/stage3_replicate.py ncbattery          # k = 0 K_avg battery (32 calls) + nc predictions (commit before chains)
  python llm/stage3_replicate.py run [--mock]       # the replicate: one stream per endpoint, sentinels, heartbeats
  python llm/stage3_replicate.py status             # print the last local status

Streams (one call in flight per endpoint): Alibaba [P1], Parasail [P2, P2-nc], CoreWeave [P3], Io Net [P4, P4-nc],
OpenAI [P5]. Chains: stage3.chain(..., run_tag="rep1") on the original graphs, fresh starts and seeds; cache tags
"stage3-rep1:...". Sentinels: 3 fixed prompts per endpoint every 5 min in that endpoint's stream (P5: 5 samples each),
for the whole run. Every 15 min: status.json + gzip copies of the cache files + finished chains -> private HF dataset
runs/stage3-rep1/ (verified by remote listing). Resumable: per-call cache, per-chain files. Budget stop $7 (spend
counted from the cache records of this run, so it survives restarts).
--mock: no API calls (a surrogate-mock agent answers), short intervals, output and HF path suffixed "-mock"; used to
prove the whole chain (streams, sentinels, heartbeat, upload + verify, resume) before paying.
"""
import argparse, gzip, json, math, os, random, shutil, sys, threading, time, traceback
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery, client, stage3, kernel_test as K
from endpoints import E
from run_stage1 import parse_lp, parse_sample

TAG = "rep1"
STREAMS = [("alibaba", ["P1"]), ("parasail", ["P2", "P2-nc"]), ("coreweave", ["P3"]), ("ionet", ["P4", "P4-nc"]), ("openai", ["P5"])]
BUDGET = 7.0


class Cfg:
    def __init__(self, mock):
        self.mock = mock
        sfx = "_mock" if mock else ""
        self.out = os.path.join(stage3.ROOT, "results", "llm", f"stage3_rep1{sfx}")
        self.hf_path = f"runs/stage3-rep1{'-mock' if mock else ''}"
        self.sent_every = 5 if mock else 300
        self.hb_every = 20 if mock else 900


# ---------------------------------------------------------------- prompts
def nc_prompt(framing, s0, m, o):
    p = stage3.render_live(framing, (1, 0, 0, 0), s0, m, o, ["aP"], contacts=False)
    stage3.check_prompt(p, framing, (1, 0, 0, 0), s0, m, o, contacts=False)
    return p


def sentinel_prompts(pid):
    """3 fixed prompts: the K_avg prompt with P(A) closest to 0.5, and the 2 most frequent live configurations with a
    fixed presentation (mapping 0, option order 0, contacts in battery.TYPES order)."""
    key, framing, rho, contacts = stage3.POINTS[pid]
    kc = pd.read_csv(os.path.join(K.OUT, "k_avg_calls.csv"), dtype={"counts": str, "seq": str})
    kc = kc[(kc.point == pid) & kc.valid]; kc["counts"] = kc.counts.str.zfill(4)
    g = kc.groupby(["counts", "s0", "mapping", "order", "seq"]).pA.mean().reset_index()
    h = g.iloc[(g.pA - 0.5).abs().argmin()]
    seq = [h.seq[i:i + 2] for i in range(0, len(h.seq), 2)]
    c = tuple(int(x) for x in h["counts"])
    out = [dict(name="near_half", counts=h["counts"], s0=int(h.s0), mapping=int(h.mapping), order=int(h.order), seq=h.seq,
                pA_kavg=float(h.pA), prompt=stage3.render_live(framing, c, int(h.s0), int(h.mapping), int(h.order), seq))]
    d = K.live_rows(pid)
    top = d.groupby(["counts", "s0"]).size().sort_values(ascending=False).index[:2]
    for j, (cs, s0) in enumerate(top):
        c = tuple(int(x) for x in cs); seq = [t for t, n in zip(battery.TYPES, c) for _ in range(n)]
        out.append(dict(name=f"typical_{j + 1}", counts=cs, s0=int(s0), mapping=0, order=0, seq="".join(seq), pA_kavg=None,
                        prompt=stage3.render_live(framing, c, int(s0), 0, 0, seq)))
    for s in out:
        stage3.check_prompt(s["prompt"], framing, tuple(int(x) for x in s["counts"]), s["s0"], s["mapping"], s["order"])
    return out


# ---------------------------------------------------------------- mock agent (no API)
def mock_call(body, rep):
    rng = random.Random(stage3._seed("rep-mock", rep))
    if body.get("max_tokens") == 1:
        pa = min(max(rng.random(), 1e-6), 1 - 1e-6)
        return {"resp": {"first": {"top": [["A", math.log(pa)], ["B", math.log(1 - pa)]]}, "usage": {"cost": 0.0}, "provider": "mock"}}
    return {"resp": {"content": "A" if rng.random() < 0.5 else "B", "usage": {"cost": 0.0}, "provider": "mock"}}


# ---------------------------------------------------------------- run
class Run:
    def __init__(self, cfg):
        self.cfg = cfg; os.makedirs(cfg.out, exist_ok=True)
        t0f = os.path.join(cfg.out, "run_t0.txt")
        if not os.path.exists(t0f):
            open(t0f, "w").write(str(time.time()))
        self.t0 = float(open(t0f).read())
        self.lock = threading.Lock()
        self.state = {name: dict(points=pts, current=None, chains_done=0, chains_total=len(pts) * 10, calls=0, errors=0,
                                 sentinel_rounds=0, last_sentinel=None, finished=False) for name, pts in STREAMS}
        self.prior_spend = self.spend_from_cache()
        self.stop = threading.Event()
        sp = os.path.join(cfg.out, "sentinel_prompts.json")
        if os.path.exists(sp):                     # a resumed run (any host) reuses the exact sentinel prompts
            self.sent = json.load(open(sp))
        else:
            self.sent = {pts[0]: sentinel_prompts(pts[0]) for _, pts in STREAMS}
            json.dump(self.sent, open(sp, "w"), indent=1)
        self.log_start()

    def log_start(self):
        """Interruption record: every (re)start appends host, pod id, chains saved and chains with cached calls but no
        saved file (mid-run when the previous segment ended). Segment ends are read from cache timestamps."""
        saved, partial = [], []
        for _, pts in STREAMS:
            for pid in pts:
                for g in range(stage3.N_GRAPHS):
                    for r in (0, 1):
                        (saved if os.path.exists(os.path.join(self.cfg.out, "chains", pid, f"g{g}_r{r}_states.npy")) else partial).append(f"{pid}:{g}:{r}")
        started = set()
        if not self.cfg.mock:
            for _, pts in STREAMS:
                p = client._path(E[stage3.POINTS[pts[0]][0]][0])
                if os.path.exists(p):
                    for l in open(p):
                        if '"rep":"stage3-rep1:' in l:
                            started.add(":".join(l.split('"rep":"stage3-rep1:')[1].split(":")[:3]))
        rec = dict(event="start", t=time.time(), utc=time.strftime("%FT%TZ", time.gmtime()), host=os.uname().nodename,
                   pod=os.environ.get("RUNPOD_POD_ID"), chains_saved=len(saved), chains_midrun=sorted(set(partial) & started))
        with open(os.path.join(self.cfg.out, "interruptions.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("RUN_START " + json.dumps(rec), flush=True)

    def spend_from_cache(self):
        if self.cfg.mock:
            return 0.0
        tot = 0.0
        for _, pts in STREAMS:
            model = E[stage3.POINTS[pts[0]][0]][0]
            p = client._path(model)
            if os.path.exists(p):
                for l in open(p):
                    if '"rep":"stage3-rep1:' in l or '"rep":"sentinel-rep1:' in l or '"rep":"kavgnc-rep1:' in l:
                        r = json.loads(l); tot += ((r["resp"].get("usage") or {}).get("cost") or 0.0)
        return tot

    def spend(self):
        return self.prior_spend + client.spend()["usd"]

    def call(self, body, rep):
        if self.cfg.mock:
            return mock_call(body, rep)
        return client.chat(body, rep=rep, budget_usd=max(0.0, BUDGET - self.prior_spend))

    def sentinels(self, stream, pid, rnd):
        key, framing, rho, contacts = stage3.POINTS[pid]
        model, tag, mode, roff, _ = E[key]
        rows = []
        for j, s in enumerate(self.sent[pid]):
            for k in range(5 if mode == "sample" else 1):
                rec = self.call(battery.body(model, tag, s["prompt"], mode, roff, 1.0), f"sentinel-{TAG}:{key}:{j}:{rnd}:{k}")
                resp = rec["resp"]
                if mode == "lp":
                    q = parse_lp(resp.get("first")); v = q["pA"]
                else:
                    a = parse_sample(resp.get("content")); v = None if a is None else float(a == "A")
                rows.append(dict(endpoint=key, sentinel=j, name=s["name"], round=rnd, sample=k, t=time.time(), pA=v,
                                 provider=resp.get("provider"), cost=(resp.get("usage") or {}).get("cost") or 0.0))
        df = pd.DataFrame(rows)
        p = os.path.join(self.cfg.out, f"sentinels_{key}.csv")
        with self.lock:
            df.to_csv(p, mode="a", header=not os.path.exists(p), index=False)
            st = self.state[stream]; st["sentinel_rounds"] += 1
            st["last_sentinel"] = {n: round(float(g.pA.mean()), 4) for n, g in df.groupby("name") if g.pA.notna().any()}

    def stream(self, name, pts):
        st = self.state[name]
        nxt = [0.0]

        def maybe_sentinel():
            now = time.time()
            if now >= nxt[0]:
                rnd = int((now - self.t0) // self.cfg.sent_every)
                self.sentinels(name, pts[0], rnd)
                nxt[0] = self.t0 + (rnd + 1) * self.cfg.sent_every

        def wrapped(body, rep):
            if self.stop.is_set():
                raise client.BudgetExceeded("stop requested")
            maybe_sentinel()
            rec = self.call(body, rep)
            with self.lock:
                st["calls"] += 1
            return rec
        try:
            for pid in pts:
                for g in range(stage3.N_GRAPHS):
                    for r in (0, 1):
                        d = os.path.join(self.cfg.out, "chains", pid)
                        if os.path.exists(os.path.join(d, f"g{g}_r{r}_states.npy")):
                            with self.lock:
                                st["chains_done"] += 1
                            continue
                        with self.lock:
                            st["current"] = f"{pid} g{g} r{r}"
                        df, states = stage3.chain(pid, g, r, call=wrapped, run_tag=TAG)
                        os.makedirs(d, exist_ok=True)
                        df.to_csv(os.path.join(d, f"g{g}_r{r}.csv.gz"), index=False)
                        np.save(os.path.join(d, f"g{g}_r{r}_states.npy"), states.astype(np.int8))
                        with self.lock:
                            st["chains_done"] += 1
                        print(f"[{name}] chain {pid} g{g} r{r} saved; invalid_first {df.invalid_first.mean():.3%}; "
                              f"spend ${self.spend():.3f}", flush=True)
        except client.BudgetExceeded as e:
            print(f"[{name}] BUDGET STOP: {e}", flush=True); self.stop.set()
        except Exception:
            with self.lock:
                st["errors"] += 1
            print(f"[{name}] STREAM ERROR\n{traceback.format_exc()}", flush=True); self.stop.set()
        with self.lock:
            st["finished"] = True; st["current"] = None
        # keep sentinels going until every stream has finished its chains
        while not self.stop.is_set() and not all(s["finished"] for s in self.state.values()):
            try:
                maybe_sentinel()
            except Exception:
                print(f"[{name}] sentinel error\n{traceback.format_exc()}", flush=True)
            time.sleep(1)

    def status(self):
        with self.lock:
            s = json.loads(json.dumps(self.state))
        el = time.time() - self.t0
        for v in s.values():
            rate = v["calls"] / max(el, 1)
            left = (v["chains_total"] - v["chains_done"]) * 2000
            v["eta_h"] = round(left / rate / 3600, 2) if rate > 0 and left > 0 else 0.0
        return dict(run=TAG, mock=self.cfg.mock, t=time.time(), elapsed_h=round(el / 3600, 3), spend_usd=round(self.spend(), 4),
                    budget_usd=BUDGET, streams=s, stopped=self.stop.is_set(), done=all(v["finished"] for v in s.values()))

    def heartbeat(self, final=False):
        st = self.status()
        json.dump(st, open(os.path.join(self.cfg.out, "status.json"), "w"), indent=1)
        st["rss_mb"] = int(next(l.split()[1] for l in open("/proc/self/status") if l.startswith("VmRSS"))) // 1024
        import resource; st["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        print("HEARTBEAT " + json.dumps({k: st[k] for k in ("elapsed_h", "spend_usd", "done", "stopped", "rss_mb", "peak_rss_mb")}) + " "
              + " | ".join(f"{n}: {v['chains_done']}/{v['chains_total']} ch, {v['calls']} calls, eta {v['eta_h']} h, sent {v['last_sentinel']}"
                           for n, v in st["streams"].items()), flush=True)
        try:
            self.upload(st)
        except Exception as e:
            print("UPLOAD FAILED (run continues; next heartbeat retries):", repr(e)[:300], flush=True)

    def upload(self, st):
        if os.environ.get("NO_UPLOAD") == "1":     # local tests only
            print("UPLOAD skipped (NO_UPLOAD=1)", flush=True); return
        from huggingface_hub import HfApi
        import fetch_cache
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo = fetch_cache.manifest()["dataset"]["repo_id"]
        assert api.repo_info(repo, repo_type="dataset").private
        up = os.path.join(self.cfg.out, "upload"); os.makedirs(up, exist_ok=True)
        if not self.cfg.mock:                      # gzip copies of the live cache files, cut at the last full line
            for _, pts in STREAMS:
                p = client._path(E[stage3.POINTS[pts[0]][0]][0])
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        b = f.read()
                    b = b[:b.rfind(b"\n") + 1]
                    with gzip.open(os.path.join(up, os.path.basename(p) + ".gz"), "wb", compresslevel=6) as fo:
                        fo.write(b)
        files = {os.path.join(up, f): f"{self.cfg.hf_path}/cache/{f}" for f in os.listdir(up) if f.endswith(".gz")}
        snap = os.path.join(up, "snap"); os.makedirs(snap, exist_ok=True)
        with self.lock:                            # snapshot files the streams append to, so upload == verified size
            for f in os.listdir(self.cfg.out):
                if f.endswith((".csv", ".json", ".jsonl", ".txt")):
                    shutil.copyfile(os.path.join(self.cfg.out, f), os.path.join(snap, f))
                    files[os.path.join(snap, f)] = f"{self.cfg.hf_path}/{f}"
        cd = os.path.join(self.cfg.out, "chains")
        for root, _, fs in os.walk(cd):
            for f in fs:
                files[os.path.join(root, f)] = f"{self.cfg.hf_path}/chains/{os.path.relpath(os.path.join(root, f), cd)}"
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=v, path_or_fileobj=k) for k, v in files.items()]
        api.create_commit(repo, repo_type="dataset", operations=ops, commit_message=f"{TAG} heartbeat {time.strftime('%FT%TZ', time.gmtime())}")
        remote = {x.path: x.size for x in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=self.cfg.hf_path, recursive=True)
                  if hasattr(x, "size")}
        bad = [v for k, v in files.items() if remote.get(v) != os.path.getsize(k)]
        print(f"UPLOAD verified {len(files) - len(bad)}/{len(files)} files" + (f"; MISMATCH {bad[:3]}" if bad else ""), flush=True)

    def go(self):
        th = [threading.Thread(target=self.stream, args=(n, p), daemon=True) for n, p in STREAMS]
        for t in th:
            t.start()
        last = 0.0
        while any(t.is_alive() for t in th):
            if time.time() - last >= self.cfg.hb_every:
                self.heartbeat(); last = time.time()
            time.sleep(2)
        self.heartbeat(final=True)
        print("RUN_END " + json.dumps({k: v for k, v in self.status().items() if k != "streams"}), flush=True)


# ---------------------------------------------------------------- k = 0 K_avg battery
def ncbattery():
    from replicate_predictions import starts, summarise, OUT as POUT
    rows = []
    for pid in ("P2-nc", "P4-nc"):
        key, framing, rho, contacts = stage3.POINTS[pid]
        model, tag, mode, roff, _ = E[key]
        for s0 in (1, -1):
            for m in (0, 1):
                for o in (0, 1):
                    for r in (0, 1):
                        rec = client.chat(battery.body(model, tag, nc_prompt(framing, s0, m, o), mode, roff, 1.0),
                                          rep=f"kavgnc-{TAG}:{pid}:{s0}:{m}:{o}:{r}", budget_usd=0.05)
                        q = parse_lp(rec["resp"].get("first"))
                        rows.append(dict(point=pid, s0=s0, mapping=m, order=o, rep=r, pA=q["pA"], leak=q["leak"],
                                         provider=rec["resp"].get("provider"), cost=(rec["resp"].get("usage") or {}).get("cost") or 0.0))
    D = pd.DataFrame(rows); os.makedirs(POUT, exist_ok=True); D.to_csv(os.path.join(POUT, "k_avg_nc_calls.csv"), index=False)
    print(D.groupby("point").agg(calls=("pA", "size"), leak_max=("leak", "max"), cost=("cost", "sum")).to_string(), D.provider.value_counts().to_dict())
    pred = json.load(open(os.path.join(POUT, "predictions.json")))
    for pid in ("P2-nc", "P4-nc"):
        rho = stage3.POINTS[pid][2]
        g = D[(D.point == pid) & (D.leak <= stage3.LEAK_MAX)].groupby(["s0", "mapping", "order"]).pA.mean().reset_index()
        g["P"] = np.where(g.mapping == 0, g.pA, 1 - g.pA)
        T = np.full(K.NT, np.nan)
        for s0 in (1, -1):
            T[K.idx((0, 0, 0, 0), s0)] = g[g.s0 == s0].P.mean()
        T = K.expand_nc(T, rho)
        df = K.simulate(pid, lambda run, gg: T, 400, "rep1-pred-nc-kavg", starts=starts(pid))
        pred["points"][pid + " (K_avg k=0, primary)"] = dict(reference="K_avg (k = 0 battery)", P_plus_given_s0={"+1": float(T[K.idx((0, 0, 0, 0), 1)]),
                                                                                                            "-1": float(T[K.idx((0, 0, 0, 0), -1)])}, **summarise(df))
        p = pred["points"][pid + " (K_avg k=0, primary)"]
        print(pid, {m: f"{p[m]['mean']:.3f} [{p[m]['lo']:.3f}, {p[m]['hi']:.3f}]" for m in ("m", "persistence", "flip_rate")})
    json.dump(pred, open(os.path.join(POUT, "predictions.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["ncbattery", "run", "status"]); ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if a.what == "ncbattery":
        ncbattery()
    elif a.what == "run":
        Run(Cfg(a.mock)).go()
    else:
        print(open(os.path.join(Cfg(a.mock).out, "status.json")).read())

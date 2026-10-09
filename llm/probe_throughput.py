"""Throughput probe: 48 distinct 1-token logprob calls per pinned endpoint at concurrency 8, counting 429s.
Not cached (throwaway prompts). Output: results/llm/catalog/throughput_<date>.csv"""
import csv, datetime, os, sys, time, requests
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client, battery
from endpoints import E

def one(model, tag, roff, i):
    b = battery.body(model, tag, f"Reply with one letter, A or B. (probe {i} {time.time()})", "lp", roff)
    t = time.time(); r = requests.post(client.URL, headers=client._headers(), json=b, timeout=60)
    return r.status_code, time.time() - t

rows = []
for key, (model, tag, mode, roff, backup) in E.items():
    for tg in [tag] + ([backup] if backup else []):
        if mode != "lp": continue
        t0 = time.time()
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda i: one(model, tg, roff, i), range(48)))
        el = time.time() - t0; ok = sum(c == 200 for c, _ in res)
        row = {"key": key, "tag": tg, "ok": ok, "n429": sum(c == 429 for c, _ in res), "other": sum(c not in (200, 429) for c, _ in res),
               "ok_per_s": round(ok / el, 2), "median_latency": round(sorted(l for _, l in res)[24], 2)}
        rows.append(row); print(row, flush=True)
with open(os.path.join(client.ROOT, "results", "llm", "catalog", f"throughput_{datetime.date.today()}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

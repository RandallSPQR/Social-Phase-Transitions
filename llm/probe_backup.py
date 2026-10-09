"""Pre-check of a backup endpoint's repeat noise before running its full battery (PREREG §4 noise gate):
100 neutral_own main-battery prompts (SHA-256 order), 2 identical calls each; RMS |y1−y2|/√2 on logits clipped ±6.
  python ../llm/probe_backup.py gemma-26b coreweave/bf16"""
import hashlib, math, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client, battery
from endpoints import E
from run_stage1 import parse_lp
key, tag = sys.argv[1], sys.argv[2]
model, _, _, roff, _ = E[key]
its = sorted(battery.items("neutral_own"), key=lambda it: hashlib.sha256(it["prompt"].encode()).hexdigest())[:100]
def one(it):
    out = []
    for r in (0, 1):
        try:
            p = parse_lp(client.chat(battery.body(model, tag, it["prompt"], "lp", roff), rep=r)["resp"]["first"])["pA"]
            out.append(None if p is None else float(np.clip(math.log(max(p, 1e-12) / max(1 - p, 1e-12)), -6, 6)))
        except Exception as e:
            out.append(None)
    return out
with ThreadPoolExecutor(8) as ex: res = list(ex.map(one, its))
ok = [(a, b) for a, b in res if a is not None and b is not None]
sd = math.sqrt(np.mean([((a - b) / math.sqrt(2)) ** 2 for a, b in ok])) if ok else float("nan")
print(f"{key} @ {tag}: {len(ok)}/100 complete pairs, repeat SD = {sd:.3f} -> {'pass' if sd <= 0.3 else 'FAIL'}; spend {client.spend()}")

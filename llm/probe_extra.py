"""Third probe: (a) are returned logprobs pre- or post-temperature? (same prompt at
T=0.5/1.0/1.5 on low-noise endpoints); (b) extra endpoints for Gemma-4-31B, Qwen3.5-122B;
(c) Gemini 3.5 Flash-Lite with reasoning off. Output: results/llm/catalog/probe_extra_<date>.csv"""
import csv, datetime, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client
from probe_repeat import PROMPTS, pA_of

rows = []
def lp(model, tag, T, prompt, rep=0, reasoning_off=False):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": T,
         "logprobs": True, "top_logprobs": 5, "provider": {"order": [tag], "allow_fallbacks": False, "require_parameters": True}}
    if reasoning_off: b["reasoning"] = {"enabled": False}
    try:
        r = client.chat(b, rep=rep)["resp"]; p, m = pA_of(r["first"])
        return {"model": model, "tag": tag, "T": T, "rep": rep, "served": r["provider"], "content": repr(r["content"]),
                "logit": None if p is None else (round(math.log(p/(1-p)), 3) if 0 < p < 1 else math.copysign(30, p-.5)), "mass": m}
    except Exception as e:
        return {"model": model, "tag": tag, "T": T, "rep": rep, "error": str(e)[:200]}

for m, t in [("meta-llama/llama-3.1-8b-instruct", "novita/fp8"), ("mistralai/mistral-small-3.2-24b-instruct", "parasail/bf16"),
             ("qwen/qwen3-235b-a22b-2507", "alibaba"), ("openai/gpt-4o-mini", "openai")]:
    for T in [0.5, 1.0, 1.5]:
        for pi in [1, 2]:
            rows.append(lp(m, t, T, PROMPTS[pi]) | {"prompt": pi})
for m, t, ro in [("google/gemma-4-31b-it", "io-net", False), ("google/gemma-4-31b-it", "venice/fp4", False), ("google/gemma-4-31b-it", "novita/bf16", False),
                 ("qwen/qwen3.5-122b-a10b", "alibaba", True), ("qwen/qwen3.5-122b-a10b", "novita/bf16", True)]:
    for r in range(4):
        rows.append(lp(m, t, 1.0, PROMPTS[0], rep=r, reasoning_off=ro) | {"prompt": 0})
for r in range(2):
    b = {"model": "google/gemini-3.5-flash-lite", "messages": [{"role": "user", "content": PROMPTS[0]}], "max_tokens": 64, "temperature": 1.0,
         "reasoning": {"enabled": False}, "provider": {"order": ["google-ai-studio"], "allow_fallbacks": False}}
    try:
        x = client.chat(b, rep=r)["resp"]; u = x["usage"] or {}
        rows.append({"model": b["model"], "tag": "google-ai-studio", "rep": r, "content": repr(x["content"]), "served": x["provider"],
                     "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "cost": u.get("cost")})
    except Exception as e:
        rows.append({"model": b["model"], "error": str(e)[:200]})
keys = []
for r in rows: keys += [k for k in r if k not in keys]
with open(os.path.join(client.ROOT, "results", "llm", "catalog", f"probe_extra_{datetime.date.today()}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows: print(json.dumps(r))
print("spend", client.spend())

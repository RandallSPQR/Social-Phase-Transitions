"""OpenRouter chat-completions client with an append-only on-disk cache.

Every request body (plus an explicit replicate index for repeated sampling) is
hashed; the compact response is appended to results/llm/cache/<model>.jsonl.
Reruns with the same body are free. The API key is read from OPENROUTER_API_KEY
if set (otherwise the session proxy injects auth); it is never printed or stored.
"""
import hashlib, json, os, threading, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "results", "llm", "cache")
URL = "https://openrouter.ai/api/v1/chat/completions"

_locks = {}
_index = {}          # model -> {hash: record}
_glock = threading.Lock()
_spend = {"usd": 0.0, "calls": 0}


class BudgetExceeded(RuntimeError):
    pass


def _slug(model):
    return model.replace("/", "__").replace(":", "_")


def _path(model):
    return os.path.join(CACHE_DIR, _slug(model) + ".jsonl")


def _load(model):
    with _glock:
        if model in _index:
            return _index[model], _locks[model]
        idx, p = {}, _path(model)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        idx[r["key"]] = r
        _index[model], _locks[model] = idx, threading.Lock()
        return idx, _locks[model]


def request_key(body, rep=0):
    canon = json.dumps({"body": body, "rep": rep}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _headers():
    h = {"Content-Type": "application/json", "X-Title": "social-phase-transitions"}
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        h["Authorization"] = "Bearer " + k
    return h


def _compact(resp):
    """Keep only what analysis needs: text, first-token top logprobs, provider, usage."""
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    lp = ch.get("logprobs") or {}
    content_lp = lp.get("content") or []
    first = None
    if content_lp:
        t0 = content_lp[0]
        first = {"token": t0.get("token"), "logprob": t0.get("logprob"),
                 "top": [[t.get("token"), t.get("logprob")] for t in (t0.get("top_logprobs") or [])]}
    return {"id": resp.get("id"), "provider": resp.get("provider"), "model": resp.get("model"),
            "content": msg.get("content"), "reasoning": msg.get("reasoning"),
            "finish": ch.get("finish_reason"), "first": first, "n_lp_tokens": len(content_lp),
            "usage": resp.get("usage")}


def cached(body, rep=0):
    idx, _ = _load(body["model"])
    return idx.get(request_key(body, rep))


def chat(body, rep=0, budget_usd=None, retries=6, timeout=60):
    """Return the compact cached record for (body, rep), calling the API on a miss."""
    model = body["model"]
    idx, lock = _load(model)
    key = request_key(body, rep)
    if key in idx:
        return idx[key]
    if budget_usd is not None and _spend["usd"] >= budget_usd:
        raise BudgetExceeded(f"session spend ${_spend['usd']:.4f} >= cap ${budget_usd}")
    payload = dict(body)
    payload["usage"] = {"include": True}
    err = None
    for a in range(retries):
        try:
            r = requests.post(URL, headers=_headers(), json=payload, timeout=timeout)
            if r.status_code == 200:
                resp = r.json()
                if "error" in resp:
                    err = f"api error: {str(resp['error'])[:300]}"
                elif not resp.get("choices"):
                    err = "empty choices"
                else:
                    rec = {"key": key, "rep": rep, "body": body, "t": time.time(), "resp": _compact(resp)}
                    cost = ((resp.get("usage") or {}).get("cost")) or 0.0
                    with lock:
                        os.makedirs(CACHE_DIR, exist_ok=True)
                        with open(_path(model), "a") as f:
                            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                        idx[key] = rec
                    with _glock:
                        _spend["usd"] += float(cost); _spend["calls"] += 1
                    return rec
            else:
                err = f"http {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 402, 403, 404):
                    break                       # not retryable
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        time.sleep(min(2 ** a, 30))
    raise RuntimeError(f"{model} rep={rep}: {err}")


def spend():
    return dict(_spend)

"""Fetch the unembedding rows for the answer letters A and B of a Gemma-4 checkpoint without downloading the weights.

  python llm/unembed_rows.py google/gemma-4-31b-it OUT.npz

Gemma ties the output projection to the input embedding (tie_word_embeddings), so W_U[t] is row t of
`*embed_tokens.weight`. The script reads the safetensors index, the shard header (HTTP range), and then only the
byte ranges of the needed rows. Token ids: every id whose stripped decode is exactly 'A' / 'B' (as local_gemma.py),
plus the ids of the single tokens 'A' and 'B' as the chat template's answer would start.
"""
import json, os, struct, sys
import numpy as np, requests


def get(url, rng=None):
    h = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    r = requests.get(url, headers=h, timeout=120); r.raise_for_status()
    return r.content


def main(repo, out):
    base = f"https://huggingface.co/{repo}/resolve/main/"
    idx = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
    name = [k for k in idx if k.endswith("embed_tokens.weight") and "vision" not in k and "audio" not in k
            and "per_layer" not in k]
    assert len(name) == 1, name
    name = name[0]; shard = base + idx[name]
    n = struct.unpack("<Q", get(shard, (0, 7)))[0]
    hdr = json.loads(get(shard, (8, 8 + n - 1)))
    t = hdr[name]; assert t["dtype"] == "BF16", t["dtype"]
    V, D = t["shape"]; start = 8 + n + t["data_offsets"][0]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    ids = {L: [i for i in range(len(tok)) if tok.decode([i]).strip() == L] for L in "AB"}
    rows = {}
    for L, lst in ids.items():
        for i in lst:
            b = get(shard, (start + i * D * 2, start + (i + 1) * D * 2 - 1))
            u16 = np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16
            rows[f"{L}_{i}"] = u16.view(np.float32).copy()
    np.savez(out, **rows, tensor=np.array(name), shape=np.array([V, D]))
    print(f"{repo}: {name} [{V} x {D}]; ids A={ids['A']} B={ids['B']} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

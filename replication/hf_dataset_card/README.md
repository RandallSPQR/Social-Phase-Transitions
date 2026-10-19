---
license: cc-by-4.0
pretty_name: "Social phase transitions: LLM response cache, pod outputs and simulation units"
language:
- en
tags:
- llm-agents
- multi-agent
- social-networks
- spin-glass
- replication
- logprobs
size_categories:
- 100K<n<1M
---

<!-- Card prepared 2026-10-15 for when the dataset is made public. Upload it as README.md at the dataset root.
     CHECK before publishing: author name; the DOI (replace 10.5281/zenodo.XXXXXXX once Zenodo has minted it);
     the model-output terms table (LICENSES.md in the repository); the secrets audit (replication/RELEASE_CHECKLIST.md). -->

# Social phase transitions: LLM response cache, pod outputs and simulation units

This dataset holds the raw material behind the study **"Do LLM agents behave like spins? Response functions, live
networks and amplification near tipping points"**. Together with the code repository it reproduces every analysis and
figure exactly, with no API keys and no GPU.

- **Code, results and replication package:** https://github.com/RandallSPQR/Social-Phase-Transitions
  (`make replicate-fast` / `make replicate`; see `replication/README.md`)
- **Archived release (code + this card):** DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)
- **Licence:** CC BY 4.0 (see "Terms of the model outputs" below)

## Contents

| path | what | size |
|---|---|---|
| `cache/*.jsonl.gz` | Response cache: one gzip JSON-lines file per model, 31 files, 709,422 records | 58.5 MB |
| `cache_manifest.json` | SHA-256, size and record count of every cache file (also committed in the repository as `results/llm/cache_manifest.json`) | — |
| `sim_units/*.tar` | Per-unit outputs of the long simulations (amplification Stage B; confirmatory locate, Steps 4–5, Gemma checks and mechanism test); manifest in the repository at `replication/sim_units_manifest.json` | 140 MB |
| `runs/<run id>/` | Durable record of every GPU/CPU pod run: heartbeat `status.json`, full logs, per-unit outputs as uploaded during the run. `runs/e3zan9ygnmqe45/` includes the Gemma-4 activations used by the exploratory probes (about 825 MB; listed in `results/llm/activations_manifest.json` in the repository). `runs/stage3-rep1/` is the live replicate (Stage 3 option B) | about 1.9 GB |
| `code/pod_code_<commit>.tgz` | The exact code bundle each pod ran, named by repository commit | about 0.5 GB |
| `pod_results/`, `cleanup/` | Small pod bookkeeping files | — |

### Cache record schema

Each line of `cache/<provider>__<model>.jsonl.gz` is one API call:

| field | meaning |
|---|---|
| `key` | SHA-256 of the canonical JSON `{"body": <request body>, "rep": <replicate index>}`; the cache key |
| `rep` | replicate index (integer, or a string tag for live-network calls such as `stage3:P1:g:r:sweep:step:attempt`) |
| `body` | the OpenRouter chat-completions request body: model, pinned provider, messages, temperature, logprob settings (no headers, no credentials) |
| `t` | Unix time of the call |
| `resp` | compact response: serving provider, model, reply text, reasoning text if any, finish reason, first answer token with its top-5 log-probabilities, number of logprob tokens, usage and cost. The OpenRouter generation id is deliberately not stored |

How the records are used: `llm/client.py` in the repository looks every request up by `key`. With `SPT_OFFLINE=1` a
request that is missing from the cache is an error, never a live call. One caveat is documented in
`replication/stage1_rematerialise.py`. A battery that sent the same request twice (same body and replicate index)
has two answers in the committed CSV but only one in the cache. The replication check accounts for this explicitly.

## How to verify and use

```bash
git clone https://github.com/RandallSPQR/Social-Phase-Transitions && cd Social-Phase-Transitions
make env && make replicate-fast PY=.venv/bin/python     # or: make docker-replicate-fast
```

The run downloads the files it needs from this dataset and checks each against its SHA-256 before use. It then
reruns the analyses and prints a pass/fail table per claim of the paper.

## Provenance

- **Collected:** 2026-10-08 to 2026-10-14 (UTC, from the record timestamps) through OpenRouter, with each endpoint pinned to one provider and fallbacks
  disabled. Self-hosted Gemma-4 runs were on rented A100 GPUs (RunPod).
- **Pre-registration:** every confirmatory phase was planned before its data existed. The plans, amendments and the
  chronological execution log are in the repository (`llm/PREREG.md`, `llm/*_PLAN.md`, `llm/EXECUTION_LOG.md`).
- **Models:** the experimental subjects are listed in the paper (Appendix D) and in `llm/endpoints.py`. The cache also
  contains small probe sets for candidates that were screened but not selected.

## Terms of the model outputs

The records contain short outputs (mostly a single answer letter and its log-probabilities) of the models below.
This dataset is released under CC BY 4.0. Use of the outputs also remains subject to each provider's terms:

| model(s) | terms |
|---|---|
| Gemma-4 26B-A4B, Gemma-4 31B | Apache 2.0 (model card) |
| Qwen3.5 9B, Qwen3.5 122B-A10B | Apache 2.0 |
| Mistral Nemo 12B, Mistral Small 3.2 24B | Apache 2.0 |
| Mistral Large 4 | Mistral AI terms |
| Llama-3.3 70B, Llama-3.1 8B, Llama-3.2 3B, Llama-4 Maverick | Llama Community Licences and Acceptable Use Policy |
| GPT-4o-mini, GPT-4o, GPT-6-luna | OpenAI terms of use and usage policies |
| Gemini 3.1 Flash-Lite, Gemini 2.5 Flash-Lite, Gemma-3 27B (via provider) | Google terms for the respective service or model |
| Claude Haiku 5.5 | Anthropic terms |
| DeepSeek chat v3 / v3.1 / v4.x, Qwen3 235B / 30B, Qwen3.5 27B (probe calls only) | the respective model licences and provider terms |

## Citation

```bibtex
@misc{social_phase_transitions_2026,
  author = {Bennington, Randall},
  title  = {Do {LLM} agents behave like spins? Response functions, live networks and amplification near tipping points: code, data and replication package},
  year   = {2026},
  doi    = {10.5281/zenodo.XXXXXXX},
  url    = {https://github.com/RandallSPQR/Social-Phase-Transitions}
}
```

## AI use

Claude (Anthropic) assisted with the experimental code, the analyses and the documentation; see the paper's AI-use
statement. The models above are the subjects of the study, not authoring tools.

# Social phase transitions in LLM agents

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23821225.svg)](https://doi.org/10.5281/zenodo.23821225)

Code, pre-registrations, results and replication package for **"Do LLM agents behave like spins? Response functions,
live networks and amplification near tipping points"** (Randall S. Bennington, 2026). Draft paper: `paper/paper.md`.

- **Replicate:** `make replicate-fast` (about 55 min on 4 cores) or `make replicate` (every simulation unit
  recomputed). No API keys and no GPU. See `replication/README.md`.
- **Data:** Hugging Face dataset [`RandallSPQR/social-phase-transitions-data`](https://huggingface.co/datasets/RandallSPQR/social-phase-transitions-data)
  (response cache, pod outputs, simulation units), fetched and checked by SHA-256 automatically.
- **Pre-registrations:** `llm/PREREG.md` and the plans in `llm/*_PLAN.md`; the order of every plan and its data is
  in `replication/provenance/PREREG_PROVENANCE.md`.
- **Licences:** code Apache-2.0, data CC BY 4.0 (`LICENSES.md`).
- **Citation:** `CITATION.cff`.

## About this repository and the commit ids in the paper

This repository is a snapshot of the release tree, without history. The study was developed in a private repository
whose history also holds third-party documents that cannot be redistributed. The commit ids cited in the paper and
the plans are that repository's. They can be checked without it:

```bash
python replication/provenance/verify_pack.py replication/provenance/pack
```

`replication/provenance/pack/` holds that history without the third-party documents and the old cache files. The
script unpacks it into an empty git repository (git recomputes every hash), and checks for each of the 22
pre-registrations its content hash and that it reached GitHub's servers before the data it governs, using GitHub's
server-side push log (`github_push_log.json`). How far each timestamp can be trusted: `PREREG_PROVENANCE.md`.

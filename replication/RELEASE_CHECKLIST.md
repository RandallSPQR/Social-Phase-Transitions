# Release checklist (prepared; nothing has been published)

Everything below is ready for the owner to switch on. Nothing in this list has been done to a public surface: the
GitHub repository visibility, releases and tags, Zenodo, and the Hugging Face dataset visibility are unchanged.

## Release plan (owner decisions, 2026-10-30)

- The five OpenAI PDFs cannot be redistributed. They were removed from the tree, but they stay in this repository's
  git history and in 10 code bundles (`code/pod_code_*.tgz`) of the HF dataset. The history also holds the OpenRouter
  generation ids (see "Audit results"). So **the original repository and the original HF dataset stay private**,
  unchanged, as the attested record. Their history is never rewritten or force-pushed: rewriting would change every
  commit id cited in the paper and break the link to GitHub's and HF's server logs.
- **No more RunPod jobs.** Section 0b is retired; the remaining full-mode times stay marked as estimates.
- The public version is a clean copy plus a **provenance package** that ties each pre-registration to its timestamp:
  `replication/provenance/` (`PREREG_PROVENANCE.md`, `provenance.json`, the exported GitHub push log and HF commit
  log, `verify_pack.py`) and the verification pack `original-history-<id>.pack` (original commits and trees without
  the PDFs and the cache blobs; 80 MB; built by `build_provenance.py pack DIR`, not committed).

## 0. Before anything is public

- [x] Secrets audit of the Hugging Face dataset and of the git history (see "Audit results").
- [x] Third-party PDFs removed from the tree (2026-10-30); the originals stay private (above).
- [x] Model-output terms confirmed by the owner (2026-10-30; `LICENSES.md`, note 2).
- [x] Author: Randall S. Bennington (`CITATION.cff`, `.zenodo.json`, dataset card, paper). Optional: affiliation and
      ORCID.
- [x] Provenance package built and verified: `verify_pack.py` on the pack in a fresh repository reports all 22
      pre-registrations verified, no PDF and no generation id in the pack.
- [ ] Rebuild the pack from the final commit just before release (`python replication/provenance/build_provenance.py
      logs` then `table` then `pack DIR`, then `verify_pack.py DIR`), so the pack contains the release commit too.

## 1. Public data: `RandallSPQR/social-phase-transitions-data` (done 2026-10-30, private until switched)

- [x] Created and filled: all 535 files of the private dataset except `code/` and `cleanup/`, hash-checked against the
      original; scanned (no PDF, no generation id, including inside tarballs). Card uploaded as `README.md`.
- [x] Manifests and docs point at it; a test fetch from it passes.
- [x] Public (owner, 2026-10-30).

## 2. Public code: `RandallSPQR/social-phase-transitions-public` (done 2026-10-30)

- [x] One snapshot commit of the release tree plus the verification pack under `replication/provenance/pack/`.

## 3. Zenodo

- [x] Published by the owner: DOI 10.5281/zenodo.23821225 (v1.0.0). Added to `CITATION.cff`, `paper/paper.md`, the
      README and the dataset card.
- [x] The v1.0.0 pack held the root commit, which GitHub created with the owner's personal email. The pack now omits
      that commit (`OMIT_COMMITS` in `build_provenance.py`; its two children are shallow boundaries in
      `verify_pack.py`), and the public repository's single commit was replaced. **Owner:** upload the new ZIP as a
      new Zenodo version (v1.0.1) and ask Zenodo support to remove the files of v1.0.0.

## 0b. Retired: RunPod CPU pod for the full-mode steps (owner decision 2026-10-30, not run)

The recipe stays in `replication/runpod/` for reference only.

## Release notes (v1.0.0)

> Code, committed results, pre-registrations, execution log and replication package for "Do LLM agents behave like
> spins? Response functions, live networks and amplification near tipping points".
>
> - `make replicate-fast` (fetch + verify data, rerun every analysis, recompute a sample of simulation units,
>   regenerate every figure, compare with the committed results, print a pass/fail table per claim) and
>   `make replicate` (also recompute every simulation unit). No API keys and no GPU. Pinned environment:
>   `replication/requirements.lock`, `Dockerfile`.
> - Data: Hugging Face dataset `RandallSPQR/social-phase-transitions-llm-cache` (CC BY 4.0): response cache (709,422
>   records), pod outputs, simulation units.
> - Licences: code Apache-2.0, data CC BY 4.0 (`LICENSES.md`).
> - Clean-run log of the replication on a fresh container: `replication/logs/`.

## Audit results

Scans print file names and pattern names only, never the matched text (`replication/audit_secrets.py`).

**Git history** (all refs, 2026-10-15; 1,475 blobs plus every commit message):
- **No API keys, tokens, bearer headers or private keys** found (OpenRouter, Hugging Face, OpenAI-style, Anthropic,
  RunPod, GitHub, AWS patterns).
- **Email-like matches in 14 binary files** (`data/*.npz`): 6-byte chance matches in compressed binary data, not
  addresses.
- **OpenRouter generation ids in 130 historical blobs: needs an owner decision.** The response cache was committed to
  git before 2026-10-09 and then removed (`.gitignore`, `llm/fetch_cache.py`). Those older cache files
  (`results/llm/cache/*.jsonl[.gz]` in history) still contain the OpenRouter generation id of each call. The owner
  decided on 2026-10-09 that these account-linked ids are not stored, and the current cache and the HF snapshots do
  not contain them. They remain in the git history, so making this repository public would publish them. Options:
  1. **Accept publication of the ids.** They identify calls, not credentials. This keeps the history, and so the
     public, verifiable commit order of every pre-registration (paper Appendix A cites commit ids).
  2. **Rewrite this repository's history** to drop `results/llm/cache/` (for example `git filter-repo --path
     results/llm/cache --invert-paths`) before making it public. This changes every commit id cited in the paper,
     the plans and the execution log. The pre-registration record then has to be re-anchored, for example with a
     mapping table from old to new ids, which outsiders cannot verify.
  3. **Publish a fresh repository** (an orphan branch of the current tree) and keep this repository private as the
     full record. The public copy then has no verifiable pre-registration history. The private repository could be
     shared with reviewers on request.

  The trade-off is between the owner's 2026-10-09 decision on generation ids and the verifiability of the
  pre-registration timestamps. Only option 1 keeps the timestamps verifiable.
- **Commit metadata** carries the author address of one commit (`RandallSPQR <…@gmail.com>`), which becomes public
  with the history. All other commits are authored as `Claude <noreply@anthropic.com>`.

**Hugging Face dataset** (2026-10-15; 357 files):
- **Scope:** every file except the 24 activation arrays (`*.npy`, raw float arrays). Code tarballs and `.gz` files
  were opened and every member scanned.
- **No API keys, tokens, bearer headers, private keys or OpenRouter generation ids** were found. The cache snapshots
  were written after generation ids stopped being stored.
- **Email-like matches (140):** all are inside `data/*.npz` binaries bundled in the pod code tarballs (`code/*.tgz`).
  They are the same 6-byte chance matches in compressed binary data as in the git history, not addresses.
- **Status:** cleared for publication as far as credentials are concerned. The terms of the model outputs are a
  separate question (LICENSES.md, note 2).

# Release checklist (prepared; nothing has been published)

Everything below is ready for the owner to switch on. Nothing in this list has been done to a public surface: the
GitHub repository visibility, releases and tags, Zenodo, and the Hugging Face dataset visibility are unchanged.

## 0. Before anything is public

- [ ] **Secrets audit of the Hugging Face dataset.** Done 2026-10-15 over every file except the activation arrays;
      results in "Audit results" below. Re-run if files are added: `replication/audit_secrets.py`.
- [ ] **Secrets audit of the git history** (the whole history becomes public with the repository): see "Audit results".
- [ ] **Third-party PDFs** in `papers/`: confirm redistribution rights or remove them before the repository/release is
      public (LICENSES.md, note 1). Removing them from the working tree does not remove them from history; if they must
      not be public, the history needs rewriting before the repository is made public (owner decision).
- [ ] **Model-output terms:** confirm the table in `LICENSES.md` (note 2), which is copied into the dataset card.
- [ ] **Author metadata:** name, affiliation and ORCID in `CITATION.cff`, `.zenodo.json` and the dataset card (fields
      marked CHECK).

## 1. Hugging Face dataset (`RandallSPQR/social-phase-transitions-llm-cache`)

- [x] Simulation unit bundles uploaded to `sim_units/` and verified by re-download (`replication/sim_units_manifest.json`).
- [ ] Upload the card: `replication/hf_dataset_card/README.md` → `README.md` at the dataset root.
- [ ] Switch the dataset to public (Settings → Visibility).
- [ ] From a machine with no `HF_TOKEN`, check that `python llm/fetch_cache.py` and `python replication/fetch_units.py` succeed.
- [ ] After the Zenodo DOI exists, replace `10.5281/zenodo.XXXXXXX` in the card and re-upload it.

## 2. GitHub release

- [ ] Make the repository public (Zenodo's GitHub integration archives public repositories).
- [ ] Merge the branch with the replication package into the default branch.
- [ ] Enable the repository in Zenodo (zenodo.org → GitHub → toggle the repository). `.zenodo.json` supplies the metadata.
- [ ] Create the release: tag `v1.0.0` on the default branch, title "v1.0.0: paper draft and replication package",
      body = the release notes below. Zenodo then archives the tagged source and mints the DOI.
- [ ] Add the DOI to `CITATION.cff`, `paper/paper.md` (Data and code availability) and the dataset card; commit.

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

**Hugging Face dataset**: see below (filled in after the scan completes).

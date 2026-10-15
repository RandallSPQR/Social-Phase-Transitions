# Release checklist (prepared; nothing has been published)

Everything below is ready for the owner to switch on. Nothing in this list has been done to a public surface: the
GitHub repository visibility, releases and tags, Zenodo, and the Hugging Face dataset visibility are unchanged.

## 0. Before anything is public

- [x] **Secrets audit of the Hugging Face dataset.** Done 2026-10-15 over every file except the activation arrays;
      results in "Audit results" below. Re-run if files are added: `replication/audit_secrets.py`.
- [x] **Secrets audit of the git history** (done; one finding needs an owner decision) (the whole history becomes public with the repository): see "Audit results".
- [ ] **Third-party PDFs** in `papers/`: confirm redistribution rights or remove them before the repository/release is
      public (LICENSES.md, note 1). Removing them from the working tree does not remove them from history; if they must
      not be public, the history needs rewriting before the repository is made public (owner decision).
- [ ] **Model-output terms:** confirm the table in `LICENSES.md` (note 2), which is copied into the dataset card.
- [ ] **Author metadata:** name, affiliation and ORCID in `CITATION.cff`, `.zenodo.json` and the dataset card (fields
      marked CHECK).

## 0b. Optional before release: the full-mode steps not yet measured (RunPod CPU pod)

Fast mode passes in a fresh container. Full mode was measured for 8 of 25 steps before the host restarted
(`replication/logs/clean_run_full_2026-10-15_partial/`). This section completes the measurement. It is not required
for the release, and the README already marks the remaining full-mode times as estimates. Recipe and failure-mode
table: `replication/runpod/README.md`. Run the commands from a checkout of the release commit, with
`RUNPOD_API_KEY` and `HF_TOKEN` in your shell. Never commit or paste them.

- [x] RunPod secret `HF_TOKEN` exists. `launch_cpu.py` passes it to the pod by reference
      (`{{ RUNPOD_SECRET_HF_TOKEN }}`, the default `--hf-secret`). It needs **write** access to the dataset, because
      the pod uploads `runs/<RUN>/`.
- [ ] Create a **new, dedicated** network volume in the RunPod console (Storage → New Network Volume):
      - name `spt-replication-logs`, about 10 GB;
      - in a data centre that offers CPU pods (the pod runs where the volume is);
      - never the shared volume. The launcher refuses any volume whose name does not start with `spt-replication`.

      Note the volume id.
- [ ] `python replication/runpod/launch_cpu.py plan`: no API call; check the spec, the steps and the caps.
- [ ] `python replication/runpod/launch_cpu.py volume-check --volume-id <ID>`: read-only; must print
      `spt-replication-logs`.
- [ ] **End-to-end test first** (CLAUDE.md rule 6), costing a few cents:
      `python replication/runpod/launch_cpu.py launch --volume-id <ID> --e2e` (dry run, prints the request), then the
      same with `--confirm`. Run `python replication/runpod/launch_cpu.py check` every few minutes until it reports
      `final_verified: true`; check then removes the pod. Confirm `runs/e2e-<sha>/` on HF holds `status.json`,
      `pod.log`, `steps/*.log`, `records.json` and `final.tar.gz`, with all 4 steps passing.
- [ ] **The real run:** `python replication/runpod/launch_cpu.py launch --volume-id <ID> --confirm` (about 13 h; job
      cap 20 h; check removes the pod at 24 h or $12). Run `check` every few hours. It removes the pod only after the
      final upload is verified.
- [ ] Ask a session to import `runs/full-remaining-<sha>/final.tar.gz` into
      `replication/logs/clean_run_full_remaining_<date>/` and replace the estimated full-mode times in
      `replication/README.md`, `Makefile`, `.zenodo.json` and `paper/paper.md` with the measured ones.
- [ ] Delete the network volume once the record is imported (it is billed monthly).
- [ ] Optional: the Amendment 4 null distribution (`s1.null`, about 21 h). 20 of its 100 replicates are already
      checked and identical. To recompute all 100, run `STEPS=s1.null` on a second pod; this needs a one-line change
      to the launcher, made on request.

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

**Hugging Face dataset** (2026-10-15; 357 files):
- **Scope:** every file except the 24 activation arrays (`*.npy`, raw float arrays). Code tarballs and `.gz` files
  were opened and every member scanned.
- **No API keys, tokens, bearer headers, private keys or OpenRouter generation ids** were found. The cache snapshots
  were written after generation ids stopped being stored.
- **Email-like matches (140):** all are inside `data/*.npz` binaries bundled in the pod code tarballs (`code/*.tgz`).
  They are the same 6-byte chance matches in compressed binary data as in the git history, not addresses.
- **Status:** cleared for publication as far as credentials are concerned. The terms of the model outputs are a
  separate question (LICENSES.md, note 2).

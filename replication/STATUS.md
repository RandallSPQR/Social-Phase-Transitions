# Replication package: working status (handoff note, 2026-10-15 ~13:40 UTC)

For a session that resumes this work without the earlier conversation. Branch `claude/peaceful-fermi-hz11cv`; all
work is committed and pushed. Background: `llm/EXECUTION_LOG.md` entry 73; package docs in `replication/README.md`.

## Owner's request (2026-10-15)

1. `make replicate` / `make replicate-fast`: fetch HF dataset → verify SHA-256 → rerun every analysis from cache (no
   keys, no GPU) → regenerate figures → compare with committed results (stated tolerances) → pass/fail per claim.
2. Pinned environment (lockfile + Dockerfile), graph-fingerprint assertions kept, runtime/hardware for both modes.
3. README with claims map, tier 1 (exact, from cache) vs tier 2 (re-collection, ~$37, provider drift).
4. Test on a fresh container, fix what breaks, record the clean-run log.
5. Licences: code Apache-2.0, data CC BY 4.0, model-provider output terms (Gemma checked).
6. Prepare GitHub release + Zenodo + HF dataset card. **Publish nothing; the owner flips the switches.**

## Done

- **Items 1–3, 5 and 6:** done. Files: `Makefile`, `Dockerfile`, `.dockerignore`, `replication/*`, `LICENSE`,
  `LICENSE-DATA`, `LICENSES.md`, `CITATION.cff`, `.zenodo.json`, `replication/hf_dataset_card/README.md`,
  `replication/RELEASE_CHECKLIST.md`.
- **Item 4, fast mode:** passes in a fresh container (image built from `git archive` of 38b926f). Every claim passes,
  53/53 headline numbers, 54.6 min, peak 4.6 GB. Record: `replication/logs/clean_run_fast_2026-10-15/` (including 4
  earlier failed attempts and their fixes).
- **HF dataset (still private):** `sim_units/amplification_units.tar` and `confirm_units.tar` uploaded and verified
  (`replication/sim_units_manifest.json`).
- **Secrets audits:** done; results in `replication/RELEASE_CHECKLIST.md`.
- **Paper and REPORT:** §4.1 now states that the asymmetry values are the N = 4,000 rows. The paper's data-availability
  section describes the package.

## In progress: full-mode clean run

- **Started:** 2026-10-15 04:54 UTC, as background job `cleanfull` (started with `llm/bg.sh`).
  - Docker container: `spt-clean-full`, image `spt-replication:full`.
  - Image source: `git archive` of the HEAD at launch, which postdates 263f046.
- **Files:**
  - Console log: `results/llm/logs/bg/cleanfull.log`. It is gitignored and lives in this container only.
  - Run outputs: `/tmp/claude-0/-home-user-Social-Phase-Transitions/5795bb49-dc99-5be4-b8bc-cb86c6f28a64/scratchpad/cleanfull/out/`,
    holding `records.json`, `logs/`, `reference/` and `regenerated/`.
  - Launch script: `.../scratchpad/cleanfull/run_full.sh`.
- **Status at 13:34 UTC:**
  - Every step up to and including `s1.amendments` passes, including the 14 bootstrap caches recomputed in 79.5 min.
  - `base.sim` is marked FAIL only because of a comparator bug: pickled `data/asym/asym.npy`. The data are
    bit-identical, checked by hand, and the comparator is fixed in commit "pickled object .npy arrays". The running
    container has the old comparator, so run `--recompare` after it ends.
  - `s1.null` (100 replicates, about 12.5–15 min each) was at 20/100 and should finish around 09:30 UTC on
    2026-10-16. Its first 12 replicates are identical to the committed ones.
  - After that, about 10 more hours (Stage B 5.5 h, Steps 4–5 2.8 h, probes, etc.): **expected end around 17–20 UTC
    on 2026-10-16**.
- **How to check:**
  - `grep -E "PASS|FAIL|^exit" results/llm/logs/bg/cleanfull.log | tail`
  - `docker exec spt-clean-full bash -c 'wc -l < results/llm/amendment4/compare_rule_null_runs.jsonl'`
  - `docker ps`
  - `df -h /`: about 5.7 GB free at 13:34.
- **Scheduled check-in:** `send_later` trigger `trig_01WJaZBd9592c2GUFJgRzyp3`, fires 2026-10-15 17:35 UTC. It
  re-checks, fixes any failure, re-arms every 4 h, and finishes the job when the run exits.
- **When it exits:**
  1. From the repository root, recompute with the current compare code:
     `python replication/replicate.py --recompare` pointed at that out dir. The driver uses `replication/out/` under
     the repository root, so copy `records.json`, `env.json`, `reference/` and `regenerated/` from the scratch out dir
     into `replication/out/` first, or run inside a copy.
  2. Save the record to `replication/logs/clean_run_full_2026-10-15/`, in the same layout as the fast record:
     driver.log, REPORT.md, report.json, records.json, env.json, `steps/*.log`, run script, and a README. Scan it for
     secrets and shorten scratch paths to `<scratch>`.
  3. Replace the full-mode estimates in `replication/README.md` (Modes table, top lines), `Makefile`, `.zenodo.json`
     and `paper/paper.md` (currently "about 37 h; 16 h with `--skip s1.null`") with the measured times.
  4. Add a line to EXECUTION_LOG entry 73. Commit, push, report to the owner, and delete the remaining check-in
     trigger.
- **If the session or container was reclaimed:** the run is lost. Report how far it got (the last records) and offer
  to relaunch: `run_full.sh`, or with `--skip s1.null` (about 16 h). Never restart without telling the owner.

## Open decisions for the owner (reported; not acted on)

1. **Generation ids in git history.** Cache files committed before 2026-10-09 still contain OpenRouter generation
   ids. The options (accept; rewrite history; fresh public repo) are in `RELEASE_CHECKLIST.md`. Only "accept" keeps
   the pre-registration commit ids verifiable.
2. **`papers/*.pdf`** (OpenAI's documents): confirm redistribution rights or remove them; they are also in the history.
3. **Author metadata:** "Randall Bennington" (inferred from the email address) in `CITATION.cff`, `.zenodo.json` and
   the card is to be confirmed, and an ORCID added.
4. **Model-output terms:** verify the table in `LICENSES.md`. The Gemma 4 licence page was unreachable; the HF card
   says Apache-2.0.
5. **Publishing:** nothing is published. The HF dataset is private, there is no release or tag, and Zenodo is not
   enabled. Steps are in `RELEASE_CHECKLIST.md`.

## Standing rules (from CLAUDE.md and the owner)

- Background jobs: start with `llm/bg.sh start NAME …` and stop with `llm/bg.sh stop NAME`. Never use `pkill -f`.
- No force-push. No PR unless asked. Never print secrets. No model identifiers in commits.
- Commit trailers: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01HkJZ66L2APs3D5WfJVsYtg`.
- Disk is a fixed allowance; it filled once. Free scratch space before large runs.

# RunPod CPU-pod recipe: the full-mode steps not yet measured (prepared, not launched)

## Purpose

- **Background:** the clean full-mode run of 2026-10-15 was cut short by a host restart after 8 of 25 steps
  (`replication/logs/clean_run_full_2026-10-15_partial/`).
- **What this recipe runs:** the rest of full mode, except the Amendment 4 null distribution (`s1.null`, about 21 h;
  20 of its 100 replicates were already checked, all identical), on a machine that is not reclaimed when a session
  goes idle.
- **Status:** prepared only. Nothing has been created on RunPod. No code bundle has been uploaded for this recipe, and
  no volume exists yet.

## Two jobs (`--job`)

| job | steps | expected | job cap | check-in caps |
|---|---|---|---|---|
| `remaining` (default) | every full-mode step not yet measured, without `s1.null` (below) | about 13 h | 20 h | 24 h or $12 |
| `null` | `qc.canary s1.null`: the Amendment 4 null distribution, all 100 replicates at B = 1000, compared with the committed `compare_rule_null_B1000.json` (profile `fit`) | about 21–25 h (12.5–15 min per replicate with the script's 4 worker processes, measured over 20 replicates) | 30 h | 34 h or $16 |

- **Running both:** the two jobs can run at the same time on two pods sharing the dedicated volume. Each writes to its
  own `/workspace/spt/<RUN>/` and its own `runs/<RUN>/` on HF.
- **Pod records:** each job keeps its own record (`replication/runpod/pod_<job>.json`, git-ignored).
- **Guard:** `launch` refuses to start a second pod for a job whose record still exists.
- **Check-ins:** each job needs its own `check --job <job>`.

## What it runs (job `remaining`)

`qc.canary s1.probes s1.figures s2.surrogates s2.sims s3.replay s3.audit k.test k.avg k.followups r.tests r.replicate
amp.stageB conf.api conf.locate conf.step45 conf.gemma substack` in full mode. Every simulation unit is recomputed from
scratch. The probes download 820 MB of Gemma activations from the HF dataset.

- **Expected time:** about 13 h. Stage B 5.5 h, Steps 4–5 2.8 h and the confirmatory locate about 0.7 h are estimates
  from the original runs; the rest is measured in fast mode. The job cap is 20 h.
- **Machine:** RunPod CPU pod, 8 vCPU compute-optimised (`cpu3c`), 20 GB container disk, secure cloud. The scripts use
  up to 4 processes; the peak RSS in fast mode was 4.6 GB.
- **Image:** `python:3.13-slim`, pinned by the same digest as the `Dockerfile`. The pod installs
  `replication/requirements.lock` (hash-checked) and refuses to run if the environment differs from the lock (networkx,
  numpy, numba, scipy, pandas). Graph fingerprints are asserted by the code as usual.
- **Code:** a `git archive` of the launch commit, uploaded to the HF dataset (`code/pod_code_<sha>.tgz`) and verified.
- **No API calls:** `SPT_OFFLINE=1`; the response cache is fetched from HF and SHA-256-verified.

## Where the logs go

1. **A dedicated network volume**, mounted at `/workspace`. Everything is under `/workspace/spt/<RUN>/`:
   - `logs/pod.log` (the full console);
   - `out/` (the driver's `replication/out`: `records.json`, `logs/<step>.log`, `reference/`, `regenerated/`,
     `REPORT.md`);
   - `status.json`, `uploads.json`, `DONE`, `FINAL_VERIFIED`, `final_<RUN>.tar.gz`.

   The volume must be **new and used only for this project**, named `spt-replication-…` (for example
   `spt-replication-logs`, 10 GB). CLAUDE.md forbids mounting the owner's shared network volume. `launch_cpu.py` reads
   the volume's name through the API and refuses any volume whose name does not start with `spt-replication`.
2. **The HF dataset**, `runs/<RUN>/`. This is the record a session can read, and CLAUDE.md requires it:
   - `status.json` and `pod.log` every 90 s;
   - `steps/<step>.log` and `records.json` as soon as each step finishes, verified by re-listing sizes;
   - `final.tar.gz` at the end, verified.
3. **stdout** (the RunPod logs tab).

## Steps for the owner

```bash
export RUNPOD_API_KEY=...   HF_TOKEN=...      # in your own shell (HF_TOKEN: the launcher uploads the code bundle)
# 0. Create the dedicated volume in the RunPod console: Storage → New Network Volume, name "spt-replication-logs",
#    10 GB, in a data centre that offers CPU pods. Note its id.
python replication/runpod/launch_cpu.py plan                                   # spec and steps; no API call
python replication/runpod/launch_cpu.py volume-check --volume-id <ID>          # read-only; must print the dedicated name
# 1. End-to-end test first (CLAUDE.md rule 6): 4 short steps (about 2 min of work), same chain end to end.
python replication/runpod/launch_cpu.py launch --volume-id <ID> --e2e            # dry run
python replication/runpod/launch_cpu.py launch --volume-id <ID> --e2e --confirm  # creates it
python replication/runpod/launch_cpu.py check --e2e   # repeat until final_verified; it then removes the pod
# 2. The real run, only after the e2e test has passed:
python replication/runpod/launch_cpu.py launch --volume-id <ID> --confirm
python replication/runpod/launch_cpu.py check     # every few hours; removes the pod after the verified final upload
# 3. Optional: the null distribution, in parallel or later (same volume; after the e2e test has passed):
python replication/runpod/launch_cpu.py launch --volume-id <ID> --job null              # dry run
python replication/runpod/launch_cpu.py launch --volume-id <ID> --job null --confirm
python replication/runpod/launch_cpu.py check --job null
```

The RunPod secret `HF_TOKEN` already exists and is the default (`--hf-secret HF_TOKEN`). The pod receives only the
reference `{{ RUNPOD_SECRET_HF_TOKEN }}`. The token needs write access to the dataset, because the pod uploads
`runs/<RUN>/`.

Afterwards, a session imports the result: download `runs/<RUN>/final.tar.gz`, then save it as
`replication/logs/clean_run_full_remaining_<date>/` in the layout of the other records. Finally, replace the
estimated full-mode times in `replication/README.md` with the measured ones.

## Cost and caps

- **Pod:** the hourly rate is read from RunPod at creation and recorded in `replication/runpod/pod.json`.
  `launch_cpu.py check` removes the pod if its uptime exceeds 24 h or its spend (rate × uptime) exceeds $12. The
  expected cost of an 8-vCPU CPU pod for about 13 h is a few dollars; check the current rate before launching.
- **Volume:** billed monthly by size (10 GB). Delete it once the record is imported. Deleting needs the owner's OK
  under CLAUDE.md.

## What you will see if it fails, by stage (CLAUDE.md rule 7)

| stage | failure | what is left behind | where to look |
|---|---|---|---|
| launch | volume is not the dedicated one | nothing is created; `launch` exits with "REFUSED" | terminal |
| launch | code bundle upload not verified | no pod created | terminal ("bundle upload not verified") |
| boot | image pull, pip or secret problem | `pod.log` on the volume; RunPod logs tab | volume `spt/<RUN>/logs/pod.log`, logs tab |
| environment | lock install fails or versions differ | "FATAL: …" in `pod.log`; no steps run | volume and logs tab; HF has no `status.json` yet |
| data fetch | HF cache or activations fail SHA-256 | driver exit 3 ("data fetch/verification FAILED") in `pod.log`; `status.json` keeps beating | HF `runs/<RUN>/pod.log`, `status.json` |
| a step | non-zero exit or a difference | that step is FAIL in `records.json`, uploaded at once; the run continues with the next step | HF `runs/<RUN>/records.json`, `steps/<step>.log` |
| heartbeat | HF upload fails | "[track] upload failed" lines; the volume copy is still written every 90 s | volume `status.json`; `check` shows a growing heartbeat age |
| time cap | the job exceeds 20 h | `timeout` stops the driver; finished steps stay recorded; final upload still runs | HF `records.json` (steps so far), `pod.log` |
| final upload | `final.tar.gz` not verified | `FINAL_VERIFIED` is absent, so the pod stays up and is NOT removed; everything is still on the volume | `check` (final_verified false); volume |
| pod lost | host failure | per-step records already on HF; full logs on the volume (it survives the pod) | HF `runs/<RUN>/`, volume |
| check-in caps | uptime > 24 h or spend > $12 | `check` removes the pod; volume and HF keep everything written so far | `check` output |

## Local test of the pod script (done 2026-10-15; no RunPod, no HF writes)

`pod_full_remaining.sh` was run in the replication Docker image with `HF_DRY=1 LOCAL_TEST=1`, a local directory as
the volume, and 3 steps (`qc.canary base.tables s1.figures`):
- all 3 steps passed;
- the heartbeat wrote `status.json` every 90 s;
- every step was marked uploaded (dry: local file present);
- the final archive was built and `DONE`/`FINAL_VERIFIED` were set;
- the script exited instead of idling.

The HF uploads, the volume mount and the RunPod create/remove calls are exercised only by the `--e2e` launch above.

#!/usr/bin/env bash
# Full-mode replication of the steps not yet measured in full mode, on a RunPod CPU pod (prepared, not launched).
# Started by replication/runpod/launch_cpu.py; see replication/runpod/README.md.
#
# Environment (set by the launcher):
#   REPO        unpacked code bundle (git archive of the launch commit)
#   VOL         dedicated network volume mount (default /runpod-volume); logs and results go to $VOL/spt/$RUN
#   RUN         run name, e.g. full-remaining-<sha>
#   HF_TOKEN    RunPod secret reference (never printed)
#   STEPS       step ids to run (default: the unmeasured full-mode steps, without s1.null)
#   JOB_HOURS   time cap for the replication itself (default 20)
#   HF_DRY=1 LOCAL_TEST=1   local test of this script: no HF writes, exits instead of idling
#
# CLAUDE.md "Remote and paid jobs" rules implemented here:
#   1 heartbeat every 90 s (status.json + full log) to the volume AND to the HF dataset runs/$RUN/
#   2 every finished step uploaded at once (records.json, step log, the step's regenerated files) and verified
#   3 the pod never removes itself; launch_cpu.py check removes it only after the final upload is verified, or on a cap
#   4 everything also goes to stdout (RunPod logs tab)
#   5 status.json carries phase, steps done, ETA, CPU/RAM/disk, spend estimate and heartbeat time
set -uo pipefail
VOL=${VOL:-/runpod-volume}; RUN=${RUN:?}; JOB_HOURS=${JOB_HOURS:-20}
STEPS=${STEPS:-"qc.canary s1.probes s1.figures s2.surrogates s2.sims s3.replay s3.audit k.test k.avg k.followups r.tests r.replicate amp.stageB conf.api conf.locate conf.step45 conf.gemma substack"}
D=$VOL/spt/$RUN; mkdir -p $D/out $D/logs
LOG=$D/logs/pod.log
exec > >(tee -a $LOG) 2>&1
say() { echo "[pod] $(date -u +%FT%TZ) $*"; }
say "run $RUN; steps: $STEPS; volume $D; cap ${JOB_HOURS} h"
cd $REPO

# --- environment: the replication lock file (hash-checked) + huggingface_hub for uploads -------------------------
python3 --version
pip install -q --require-hashes -r replication/requirements.lock || { say "FATAL: lock install failed"; exit 1; }
pip install -q huggingface_hub==1.33.0 || { say "FATAL: huggingface_hub install failed"; exit 1; }
python3 - <<'EOF' || { echo "[pod] FATAL: environment does not match the lock"; exit 1; }
import sys; sys.path.insert(0, "replication"); import replicate as R
info, hard = R.check_env(); print("[pod] env", info["python"], info["cpus"], "CPUs"); assert not hard, hard
EOF

# results and logs live on the volume: the driver writes replication/out -> $D/out
rm -rf replication/out && ln -s $D/out replication/out

# --- heartbeat + per-step uploads (background) --------------------------------------------------------------------
cat > /tmp/track.py <<'EOF'
"""Heartbeat every 90 s; upload every newly finished step at once and verify it (sizes re-listed from HF)."""
import json, os, shutil, sys, time
from huggingface_hub import HfApi
D, RUN, REPO = os.environ["D"], os.environ["RUN"], "RandallSPQR/social-phase-transitions-llm-cache"
api = HfApi(token=os.environ["HF_TOKEN"]); t0 = float(os.environ["T0"]); done_up = set()
DRY = os.environ.get("HF_DRY") == "1"   # local test of this script: no HF writes, the volume only
def up(local, remote):
    """Upload, then re-list the run folder and compare sizes; True only if the remote copy is verified."""
    if DRY:
        return os.path.exists(local)
    api.upload_file(path_or_fileobj=local, path_in_repo=f"runs/{RUN}/{remote}", repo_id=REPO, repo_type="dataset",
                    commit_message=f"{RUN}: {remote}")
    sizes = {x.path: getattr(x, "size", None)
             for x in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"runs/{RUN}", recursive=True)}
    return sizes.get(f"runs/{RUN}/{remote}") == os.path.getsize(local)
while True:
    rec = {}
    try: rec = json.load(open(f"{D}/out/records.json"))
    except Exception: pass
    verified = json.load(open(f"{D}/uploads.json")) if os.path.exists(f"{D}/uploads.json") else {}
    for sid, r in rec.items():
        if sid in done_up or sid in verified: continue
        step_log = f"{D}/out/logs/{sid}.log"
        ok = up(step_log, f"steps/{sid}.log") if os.path.exists(step_log) else True
        ok = up(f"{D}/out/records.json", "records.json") and ok
        if ok: verified[sid] = time.time(); done_up.add(sid)
        json.dump(verified, open(f"{D}/uploads.json", "w"))
    du = shutil.disk_usage(D); la = os.getloadavg()[0]
    try: mem = dict(l.split(":") for l in open("/proc/meminfo")); avail = int(mem["MemAvailable"].split()[0]) / 1e6
    except Exception: avail = None
    st = dict(t=time.time(), run=RUN, uptime_h=round((time.time() - t0) / 3600, 2), steps_done=list(rec),
              steps_ok=[k for k, v in rec.items() if v.get("ok")], steps_failed=[k for k, v in rec.items() if not v.get("ok")],
              steps_uploaded=sorted(verified), load1=la, mem_avail_gb=avail, vol_free_gb=round(du.free / 1e9, 1),
              done=os.path.exists(f"{D}/DONE"), final_verified=os.path.exists(f"{D}/FINAL_VERIFIED"))
    json.dump(st, open(f"{D}/status.json", "w"), indent=1)
    for f, rem in ((f"{D}/status.json", "status.json"), (f"{D}/logs/pod.log", "pod.log")):
        try: up(f, rem)
        except Exception as e: print("[track] upload failed", rem, type(e).__name__, flush=True)
    print(f"[track] {time.strftime('%FT%TZ', time.gmtime())} steps {len(rec)} uploaded {len(verified)} "
          f"load {la:.1f} mem_avail {avail} GB vol_free {st['vol_free_gb']} GB", flush=True)
    if st["final_verified"]: break
    time.sleep(90)
EOF
export D RUN T0=$(date +%s)
python3 /tmp/track.py &
TRACK=$!

# --- the replication itself (time-capped) ---------------------------------------------------------------------------
say "replication start"
timeout $((JOB_HOURS * 3600)) python3 replication/replicate.py --mode full --only $STEPS
RC=$?
say "replication exit $RC"
touch $D/DONE

# --- final upload: report, records, every step log, the regenerated files; verified ----------------------------------
python3 - <<'EOF'
import os, tarfile
from huggingface_hub import HfApi
D, RUN, REPO = os.environ["D"], os.environ["RUN"], "RandallSPQR/social-phase-transitions-llm-cache"
api = HfApi(token=os.environ["HF_TOKEN"])
tgz = f"{D}/final_{RUN}.tar.gz"
DRY = os.environ.get("HF_DRY") == "1"
with tarfile.open(tgz, "w:gz") as t:
    for x in ("REPORT.md", "report.json", "records.json", "env.json", "canary.json", "logs", "regenerated"):
        if os.path.exists(f"{D}/out/{x}"): t.add(f"{D}/out/{x}", arcname=x)
if DRY:
    ok = os.path.getsize(tgz) > 0
else:
    api.upload_file(path_or_fileobj=tgz, path_in_repo=f"runs/{RUN}/final.tar.gz", repo_id=REPO, repo_type="dataset",
                    commit_message=f"{RUN}: final")
    sizes = {x.path: getattr(x, "size", None)
             for x in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"runs/{RUN}", recursive=True)}
    ok = sizes.get(f"runs/{RUN}/final.tar.gz") == os.path.getsize(tgz)
print("[pod] final upload", ("DRY RUN (not uploaded; local file only)" if DRY else ("VERIFIED" if ok else "NOT VERIFIED")), os.path.getsize(tgz), "bytes", flush=True)
if ok: open(f"{D}/FINAL_VERIFIED", "w").write("ok\n")
EOF
sleep 100   # one more heartbeat carries final_verified
say "boot sequence ended (replication exit $RC); idling until launch_cpu.py check removes the pod"
[ "${LOCAL_TEST:-0}" = "1" ] && { wait $TRACK; exit $RC; }
sleep infinity

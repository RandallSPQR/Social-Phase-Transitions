#!/usr/bin/env bash
# Background jobs with PID files. Never stop a job by name pattern (pkill -f / pgrep -f match their own command line
# and other tools' command lines: it killed the session shell and a monitor twice on 2026-10-09/10).
#
#   llm/bg.sh start NAME CMD [ARGS...]   start CMD in its own session/process group; PID -> results/llm/logs/bg/NAME.pid
#   llm/bg.sh stop NAME                  SIGTERM the whole process group recorded in NAME.pid (then SIGKILL after 10 s)
#   llm/bg.sh status [NAME]              running / exited, from the PID files
#   llm/bg.sh wait NAME                  block until the job exits
# Output goes to results/llm/logs/bg/NAME.log.
set -uo pipefail
cd "$(dirname "$0")/.."
D=results/llm/logs/bg; mkdir -p "$D"
alive() { [ -f "$D/$1.pid" ] && kill -0 "$(cat "$D/$1.pid")" 2>/dev/null; }
case "${1:-}" in
  start)
    name=$2; shift 2
    if alive "$name"; then echo "$name already running (pid $(cat "$D/$name.pid"))"; exit 1; fi
    setsid nohup "$@" >> "$D/$name.log" 2>&1 < /dev/null &
    echo $! > "$D/$name.pid"; echo "$(date -u +%FT%TZ) start pid $! : $*" >> "$D/$name.log"
    echo "$name started, pid $(cat "$D/$name.pid") (process group leader)";;
  stop)
    name=$2; [ -f "$D/$name.pid" ] || { echo "no pid file for $name"; exit 1; }
    pid=$(cat "$D/$name.pid")
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
      for _ in $(seq 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null
      echo "$(date -u +%FT%TZ) stopped pid $pid" >> "$D/$name.log"; echo "$name stopped (pid $pid)"
    else echo "$name not running"; fi;;
  status)
    for f in "$D"/${2:-*}.pid; do [ -f "$f" ] || continue; n=$(basename "$f" .pid)
      if alive "$n"; then echo "$n running (pid $(cat "$f"))"; else echo "$n exited"; fi; done;;
  wait)
    while alive "$2"; do sleep 5; done; echo "$2 exited";;
  *) sed -n '2,10p' "$0"; exit 1;;
esac

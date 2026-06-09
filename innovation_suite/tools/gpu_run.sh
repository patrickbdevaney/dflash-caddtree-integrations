#!/bin/bash
# gpu_run.sh — single-container GPU launch guard ("ultimate lookahead").
# Source this and use gpu_run / gpu_wait / gpu_stop / gpu_clean / gpu_preflight.
#
# Prevents the four failure modes that bit us:
#   1. DOUBLE-LAUNCH      -> flock mutex: only one launch proceeds at a time.
#   2. STUCK "Created"     -> NEVER SIGKILL the docker process (that orphans the
#                            container in containerd). Always graceful `docker stop`
#                            then `docker rm -f` of a NAMED container (idempotent).
#   3. MULTI-HOUR HANG     -> Created->running watchdog (90s). If it doesn't start,
#                            it's a runtime wedge -> abort + flag REBOOT, don't pile on.
#   4. ANON ACCUMULATION   -> every container is --name + --label, so they are
#                            enumerable and cleanable; no anonymous strays.
LOCK=/tmp/dflash_gpu.lock
LABEL=dflash-gpu

gpu_health()    { timeout 20 docker info >/dev/null 2>&1; }
gpu_list_mine() { timeout 15 docker ps -aq --filter "label=${LABEL}=1" 2>/dev/null; }

gpu_stop() {  # graceful stop + rm of ONE named container (no kill -9 of the process)
    local n="$1"
    timeout 45 docker stop "$n" >/dev/null 2>&1
    timeout 20 docker rm -f "$n" >/dev/null 2>&1
}

gpu_clean() {  # graceful remove of ALL my containers
    local ids; ids=$(gpu_list_mine)
    if [ -n "$ids" ]; then timeout 45 docker stop $ids >/dev/null 2>&1; timeout 25 docker rm -f $ids >/dev/null 2>&1; fi
}

gpu_preflight() {  # 0 = safe to launch; 1 = daemon down; 2 = WEDGED (reboot needed)
    if ! gpu_health; then echo "[preflight] FAIL: docker daemon not responsive"; return 1; fi
    local mine; mine=$(gpu_list_mine)
    if [ -n "$mine" ]; then echo "[preflight] cleaning prior containers: $(echo $mine|tr '\n' ' ')"; gpu_clean; sleep 2; fi
    local still; still=$(gpu_list_mine)
    if [ -n "$still" ]; then echo "[preflight] FAIL: unremovable containers ($still) -> RUNTIME WEDGED, REBOOT NEEDED"; return 2; fi
    echo "[preflight] OK: 0 of my containers, daemon healthy"; return 0
}

# gpu_run NAME LOGFILE -- <docker run args...>   (args go after `--`, e.g. --runtime nvidia ... image entrypoint cmd)
# Launches detached+named, waits for 'running'. Returns 0 running, 2 wedge, 1 other.
gpu_run() {
    local name="$1" logfile="$2"; shift 2; [ "$1" = "--" ] && shift
    exec 9>"$LOCK"
    if ! flock -n 9; then echo "[gpu_run] ABORT: another gpu_run holds the lock (no double-launch)"; return 1; fi
    if ! gpu_preflight; then local rc=$?; flock -u 9; echo "[gpu_run] ABORT (preflight rc=$rc)"; return $rc; fi
    gpu_stop "$name"
    echo "[gpu_run] launching $name"
    if ! timeout 60 docker run -d --name "$name" --label "${LABEL}=1" "$@" >/dev/null 2>>"$logfile"; then
        echo "[gpu_run] docker run errored"; gpu_stop "$name"; flock -u 9; return 1
    fi
    local st="" i
    for i in $(seq 1 18); do
        st=$(timeout 10 docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)
        if [ "$st" = "running" ]; then
            echo "[gpu_run] $name RUNNING"
            ( docker logs -f "$name" >"$logfile" 2>&1 ) & echo $! > "/tmp/${name}.logpid"
            flock -u 9; return 0
        elif [ "$st" = "exited" ]; then
            echo "[gpu_run] $name EXITED early"; timeout 15 docker logs "$name" 2>&1 | tail -25 >>"$logfile"; gpu_stop "$name"; flock -u 9; return 1
        fi
        sleep 5
    done
    echo "[gpu_run] WEDGE: $name stuck in '$st' after 90s -> nvidia runtime wedged. Removing + ABORT. REBOOT to recover."
    gpu_stop "$name"; flock -u 9; return 2
}

gpu_wait() {  # block until NAME exits; echo its exit code. Streams nothing (logs already tee'd).
    local name="$1" st
    while :; do
        st=$(timeout 10 docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)
        [ -z "$st" ] && { echo "gone"; return 0; }
        [ "$st" = "exited" ] && { echo "$(timeout 10 docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null)"; return 0; }
        sleep 20
    done
}

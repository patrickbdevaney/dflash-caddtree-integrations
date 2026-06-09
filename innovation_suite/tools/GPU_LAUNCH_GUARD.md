# gpu_run.sh — the GPU container launch guard (root-cause fix for the 6h incident)

## What went wrong (2026-06-08)
1. I launched the LongPPL disc phase TWICE -> two docker runs fought for the GPU.
2. I SIGKILL-9'd the docker run processes to clean up -> that ORPHANS the container at the
   containerd/nvidia-runtime level, leaving it stuck in "Created". A stuck-Created GPU container
   holds the GPU reservation, so EVERY subsequent container hangs at creation. It even survived a
   docker daemon restart (recreated from metadata); only a full reboot cleared the runtime wedge.

## The fix (gpu_run.sh) — makes all four failure modes structurally impossible
1. DOUBLE-LAUNCH  -> `flock` mutex: only one gpu_run proceeds at a time.
2. STUCK "Created" -> NEVER SIGKILL the docker process. Graceful `docker stop` then `docker rm -f`
   of a NAMED + LABELED container (idempotent). No process is ever -9'd.
3. MULTI-HOUR HANG -> Created->running watchdog (90s). If it doesn't start, it's declared a
   runtime WEDGE -> abort + flag REBOOT. Never hang for hours, never pile on.
4. ANON ACCUMULATION -> every container is `--name` + `--label dflash-gpu=1`, so all are
   enumerable/cleanable; preflight removes any of mine before launching, and refuses (WEDGED) if
   one is unremovable.

## Usage
    source gpu_run.sh
    gpu_run <name> <logfile> -- <docker run args...>   # launches, waits for 'running' (rc 0/1/2)
    gpu_wait <name>                                     # blocks until exit, echoes exit code
    gpu_stop <name> ; gpu_clean ; gpu_preflight         # graceful cleanup / health
Orchestrators (run_longppl2.sh) source it and run each phase as one named container.
RULE: never `docker run` a GPU container directly again; never `kill -9` a docker process.

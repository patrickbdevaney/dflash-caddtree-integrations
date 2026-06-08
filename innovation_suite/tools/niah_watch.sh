#!/bin/bash
# NIAH live dashboard. One-shot snapshot of the 1M S-NIAH run.
# Live view:  watch -n 20 -c bash ~/dflash-dev/niah_watch.sh
# Plain tail: tail -f ~/dflash-dev/niah_status.log
STATUS=~/dflash-dev/niah_status.log
PIDF=/tmp/niah.pid
b=$'\e[1m'; g=$'\e[32m'; y=$'\e[33m'; r=$'\e[31m'; c=$'\e[36m'; n=$'\e[0m'

echo "${b}========== NIAH 1M DASHBOARD  $(date '+%H:%M:%S') ==========${n}"

# --- 1. Process / container ---
if [ -f "$PIDF" ] && pgrep -f run_niah >/dev/null 2>&1; then
  echo "  orchestrator : ${g}ALIVE${n} (pid $(cat "$PIDF" 2>/dev/null))"
else
  echo "  orchestrator : ${r}NOT RUNNING${n}  (finished or stopped — check NIAH_RESULT below)"
fi
cont=$(docker ps --format '{{.Names}}|{{.Status}}' 2>/dev/null | head -1)
if [ -n "$cont" ]; then
  echo "  container    : ${g}${cont%%|*}${n}  (${cont##*|})"
else
  echo "  container    : ${y}none running${n}"
fi

# --- 2. GPU activity (the 'is it actually working' signal) ---
util=$(timeout 15 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
if [ -n "$util" ]; then
  num=${util%% *}
  if [ "${num:-0}" -ge 50 ] 2>/dev/null; then col=$g; else col=$y; fi
  echo "  GPU util     : ${col}${util}${n}   ${col}$( [ "${num:-0}" -ge 50 ] && echo '← COMPUTING' || echo '← idle/between-steps' )${n}"
else
  echo "  GPU util     : ${y}n/a (tegra smi)${n}"
fi

# --- 3. Which context length is active + the per-ctx log ---
ctx=$(grep -oE "NIAH ctx=[0-9]+" "$STATUS" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
ctx=${ctx:-1000000}
plog=/tmp/niah_${ctx}.log
echo "  context len  : ${c}${ctx}${n}  (log: $plog)"

# --- 4. Progress: instances done + current prefill bar ---
if [ -f "$plog" ]; then
  hits=$(grep -c "HIT$" "$plog" 2>/dev/null); miss=$(grep -c "MISS$" "$plog" 2>/dev/null)
  done=$((hits+miss))
  echo "  instances    : ${b}${done}/15${n} done   (${g}${hits} HIT${n}, ${r}${miss} MISS${n})  [3 positions x 5]"
  # current generate progress bar (vLLM 'Processed prompts ... %')
  bar=$(tr '\r' '\n' < "$plog" 2>/dev/null | grep "Processed prompts" | tail -1 | grep -oE "[0-9]+%[^,]*" | head -1)
  [ -n "$bar" ] && echo "  current step : ${c}prefill/gen ${bar}${n}  (1M prefill is slow: ~30-60 min each)"
  # last per-position result line
  lastpos=$(grep -E "NIAH_POS" "$plog" 2>/dev/null | tail -1)
  [ -n "$lastpos" ] && echo "  last position: ${g}${lastpos}${n}"
else
  echo "  instances    : ${y}per-ctx log not created yet (model still loading)${n}"
fi

# --- 5. Final result if present ---
res=$(grep "NIAH_RESULT" "$STATUS" "$plog" 2>/dev/null | tail -1)
[ -n "$res" ] && echo "  ${g}${b}RESULT${n} : ${res#*NIAH_RESULT }"
grep -q "NIAH_DONE" "$STATUS" 2>/dev/null && echo "  ${g}${b}>>> RUN COMPLETE (NIAH_DONE)${n}"

# --- 6. Recent orchestrator log + JIT-compile note ---
echo "${b}--- last status lines ---${n}"
grep -E "NIAH_START|clean:|NIAH ctx=|NIAH_POS|step down|OutOfMemory|NIAH_COMPLETE|NIAH_DONE" "$STATUS" 2>/dev/null | tail -4 | sed 's/^/  /'
last=$(tr '\r' '\n' < "$plog" 2>/dev/null | grep -vE "^$" | tail -1)
[ -n "$last" ] && echo "  ${c}plog tail:${n} ${last:0:90}"

echo "${b}--- ways to watch ---${n}"
echo "  live dashboard : ${c}watch -n 20 bash ~/dflash-dev/niah_watch.sh${n}"
echo "  raw run log    : ${c}tail -f ~/dflash-dev/niah_status.log${n}"
echo "  current prefill: ${c}tail -f $plog${n}"
echo "  gpu only       : ${c}watch -n 5 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader${n}"

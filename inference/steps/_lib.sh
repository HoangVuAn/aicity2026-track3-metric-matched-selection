#!/usr/bin/env bash
# Shared setup for the split-step inference scripts (steps/N_*.sh). This is the same flow as
# run_full_fast.sh, cut into steps you can run one at a time. Each step sources this file,
# then does its own serve -> wait -> generate. Steps are resume-safe (the gen_*.py skip
# finished items) and read every model / path / port / N from config.sh, so nothing here
# hardcodes a GPU index. Run them in order 1..6 (each consumes the previous step's pools).
set -uo pipefail
INF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # inference/ (parent of steps/)
source "$INF/config.sh"
cd "$REPO_DIR"
mkdir -p "$R"
OVR="$COSMOS_OVERRIDES"

# wait for a vLLM port to answer (up to 30 min)
wp(){ for t in $(seq 1 180); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$1/v1/models)" = "200" ] && return 0; sleep 10; done; echo "PORT $1 FAIL"; return 1; }
# kill OUR vLLM procs on the configured cards (G1/G2/G3) to free them between steps
kg(){ for row in $(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader|awk -F', ' '$3+0>1000{print $1","$2}'); do
  p=$(echo $row|cut -d, -f1); u=$(echo $row|cut -d, -f2); idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader|grep "$u"|cut -d, -f1|tr -d ' ');
  [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" = "anhv10" ] && echo "$idx"|grep -qE "^($G1|$G2|$G3)$" && kill -9 $p 2>/dev/null; done; sleep 8; }

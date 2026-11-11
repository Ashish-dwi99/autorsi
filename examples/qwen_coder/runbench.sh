#!/bin/bash
# runbench.sh <tag> <he|mbpp>: relaunch bench.py in chunks until DONE
. /tmp/v/bin/activate; cd "$(dirname "$0")"
while true; do python bench.py $1 $2 2>&1 | grep -E "DONE|MORE" | tee -a ${AUTORSI_OUT:-/tmp/w}/bench_$1_$2.log | grep -q DONE && break; done

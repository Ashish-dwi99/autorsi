#!/bin/bash
. /tmp/v/bin/activate; cd "$(dirname "$0")"
export MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=0 MALLOC_ARENA_MAX=1
python exp_rft.py 5e-5 4 rft1 20,21,22,23 > /tmp/w/rft1.log 2>&1
./runbench.sh rft1 mbpp; ./runbench.sh rft1 he
echo done > /tmp/w/pipeline3.done

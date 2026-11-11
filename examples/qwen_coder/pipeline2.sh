#!/bin/bash
. /tmp/v/bin/activate; cd "$(dirname "$0")"
while [ ! -f /tmp/w/pipeline.done ]; do sleep 5; done
python sample_rft.py 80 > /tmp/w/rft_sample.log 2>&1
python exp_rft.py 5e-5 6 rft1 > /tmp/w/rft1.log 2>&1
./runbench.sh rft1 mbpp; ./runbench.sh rft1 he
echo done > /tmp/w/pipeline2.done

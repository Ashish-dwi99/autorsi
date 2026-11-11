#!/bin/bash
. /tmp/v/bin/activate; cd "$(dirname "$0")"
python exp_scalar.py > /tmp/w/scalar.log 2>&1
python exp_grad.py full 2e-4 24 gfull 20,21,22,23 > /tmp/w/gfull.log 2>&1
./runbench.sh gfull mbpp; ./runbench.sh gfull he
./runbench.sh base mbpp; ./runbench.sh base he
echo done > /tmp/w/pipeline.done

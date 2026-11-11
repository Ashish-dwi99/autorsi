#!/bin/bash
# kills python jobs when free RAM gets dangerously low (this box wedges instead of OOM-killing)
while true; do a=$(free -m | awk '/Mem:/{print $7}'); if [ "$a" -lt 120 ]; then pkill -9 -f exp_grad; pkill -9 -f bench.py; echo "$(date) killed, avail=$a" >> /tmp/w/watchdog.log; fi; sleep 2; done

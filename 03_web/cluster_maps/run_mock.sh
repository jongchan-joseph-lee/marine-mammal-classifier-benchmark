#!/bin/sh
cd /home/claude/watkins_maps
ps -eo pid,args | awk '/python3 build_maps.py/ && !/awk/ {print $1}' | xargs -r kill
sleep 1
nohup python3 build_maps.py --cuts mock/cuts.csv --features mock/features --out mock/data.json --kind mock > mock/build.log 2>&1 &

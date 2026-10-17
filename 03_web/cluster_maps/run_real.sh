#!/bin/sh
cd /home/claude/watkins_maps
nohup python3 -I build_maps.py --cuts real/cuts.csv --features /home/claude/pkg_in/full_cohort/features --features-sample /home/claude/pkg_in/work/features --gpt-cache /home/claude/pkg_in/cached_umap3d --out real/data.json > real/build.log 2>&1 &

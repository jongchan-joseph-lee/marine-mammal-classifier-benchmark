#!/bin/sh
cd /path/to/watkins_work/watkins_maps
nohup python3 -I build_maps.py --cuts real/cuts.csv --features /path/to/watkins_work/pkg_in/full_cohort/features --features-sample /path/to/watkins_work/pkg_in/work/features --gpt-cache /path/to/watkins_work/pkg_in/cached_umap3d --out real/data.json > real/build.log 2>&1 &

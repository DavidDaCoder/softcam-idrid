#!/usr/bin/env bash
# Fair top-k comparison: re-score the 5 Top-k SC runs on their contribution maps
# (what each cell adds to the logit), then aggregate next to Mean SC and Multi SC s3-topk.
# CPU only, ~2 min per run. Run on TIDE:  bash ~/analysis_code/fair_topk.sh
set -euo pipefail
CODE=~/analysis_code
OUT=~/analysis/idrid/fair_topk_$(date +%m%d_%H%M%S); mkdir -p "$OUT"
TOPK="24-10-2026_06:27:36 26-10-2026_05:03:57 26-10-2026_05:21:06 26-10-2026_05:38:12 26-10-2026_05:55:15"
for ts in $TOPK; do
  echo "== $ts"
  python "$CODE/eval_idrid_localization.py" --run "$ts" --score contribution --device cpu > "$OUT/eval_${ts}_contrib.log" 2>&1 \
    || { echo "FAILED, see $OUT/eval_${ts}_contrib.log"; exit 1; }
  grep -E "faithfulness|contribution maps" "$OUT/eval_${ts}_contrib.log"
done
M="$OUT/manifest.csv"; echo "label,seed,pooling,conv_cls,timestamp" > "$M"
MEAN=(26-10-2026_01:36:04 26-10-2026_04:58:09 26-10-2026_05:15:18 26-10-2026_05:32:31 26-10-2026_05:49:33)
TK=($TOPK)
S3TK=(26-10-2026_20:12:15 26-10-2026_20:23:10 26-10-2026_20:34:07 26-10-2026_20:45:01 26-10-2026_20:55:56)
for s in 0 1 2 3 4; do echo "Mean SC,$s,mean,True,${MEAN[$s]}" >> "$M"; done
for s in 0 1 2 3 4; do echo "Top-k SC,$s,topk,True,${TK[$s]}" >> "$M"; done
for s in 0 1 2 3 4; do echo "Top-k SC contrib,$s,topk,True,${TK[$s]}_contrib" >> "$M"; done
for s in 0 1 2 3 4; do echo "Multi SC s3-topk,$s,mean,True,${S3TK[$s]}" >> "$M"; done
python "$CODE/aggregate_idrid_seeds.py" --manifest "$M" --out "$OUT" > "$OUT/aggregate.log" 2>&1
for ts in $TOPK; do cp -r ~/analysis/idrid/${ts}_contrib "$OUT/"; done
tar czf ~/$(basename "$OUT").tar.gz -C ~/analysis/idrid "$(basename "$OUT")"
echo; cat "$OUT/SEEDS_SUMMARY.md"; echo; echo "tarball: ~/$(basename "$OUT").tar.gz"

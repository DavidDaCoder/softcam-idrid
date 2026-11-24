#!/usr/bin/env bash
# Overnight IDRiD queue: train chosen models x seeds, then evaluation, figures, cross-seed
# aggregation, a tarball, config restore, and server stop. Needs add_multilayer.py installed.
#
# Start (GPU server, from any directory):
#   nohup bash ~/analysis_code/idrid_night.sh > ~/logs/idrid_night.log 2>&1 &
# Multi-layer night:  MODELS="multi" SEEDS="0 1 2 3 4" nohup bash ~/analysis_code/idrid_night.sh > ~/logs/idrid_night.log 2>&1 &
# Watch:   tail -f ~/logs/idrid_night.log
# Options: MODELS="mean topk bb multi swin swinml"  SEEDS="1 2"  AUTOSTOP=0  DRY=1 (check config edits only)
#          swin / swinml = Swin-T SoftCAM, stage 4 only / stages 3+4 (needs add_swin.py installed)
#          s3 = ResNet-50 SoftCAM on stage 3 only; multitk3 = stages 3+4 with top-k on stage 3 (needs add_multilayer.py v2)
#          s3tk = stage 3 only with top-k 5% (contribution map; separates fine grid from fusion)
#          multitk34 = stages 3+4, top-k 5% on both; multitk3_k1 / multitk3_k10 = s3-topk with k = 1% / 10%
#          (optional 7th spec field = topk_frac; default 0.05)
#          Learning-rate sweep for model 1 (Mean SC; lr 0.001 = the existing Mean SC runs):
#            mean_lr3e4 / mean_lr3e3 / mean_lr1e2 = Mean SC with lr 0.0003 / 0.003 / 0.01
#            e.g. MODELS="mean_lr3e4 mean_lr3e3 mean_lr1e2" SEEDS="0 1 2"
#          (optional 8th/9th spec fields = learning_rate / weight_decay; defaults 0.001 / 0.0005;
#           clipped_cosine min_lr stays 0.0001, so lr 0.0003 still decays to 1e-4)
#          After evaluation, eval_idrid_splits.py scores train/val/test for every run that lacks splits.json.
#          PRIOR_MANIFEST=<manifest.csv> (earlier runs to include; default: the ~/analysis/idrid/seeds_*/manifest.csv with the most runs)
#          KEEP_ALL_CKPTS=1 (keep every checkpoint; default keeps only best_validation_weights_acc_m.pt)
#          MIN_FREE_GB=5 (stop training when $HOME has less free space)
# Runs already in the prior manifest (same label + seed) are not retrained.
# Cancel:  pkill -f idrid_night.sh; pkill -f "python main.py"   (then check configs/default.yaml)
set -u
SEEDS="${SEEDS:-1 2 3 4}"
MODELS="${MODELS:-mean topk bb}"
AUTOSTOP="${AUTOSTOP:-1}"
DRY="${DRY:-0}"
KEEP_ALL_CKPTS="${KEEP_ALL_CKPTS:-0}"   # 0: after each run keep only best_validation_weights_acc_m.pt (what the eval uses)
MIN_FREE_GB="${MIN_FREE_GB:-5}"         # stop training (still evaluate + aggregate) when $HOME has less free space than this
REPO="$HOME/SoftCAM"; CFG="$REPO/configs/default.yaml"
OUTS="$HOME/Outputs/FullyConvNet/Fundus"; CODE="$HOME/analysis_code"
STAMP=$(date +%m%d_%H%M%S); NIGHT="$HOME/analysis/idrid/seeds_$STAMP"
mkdir -p "$NIGHT" "$HOME/logs" "$OUTS"
# shellcheck disable=SC1091
source "$HOME/envs/softcam/bin/activate"
BEFORE="$HOME/logs/default.yaml.before_$STAMP"      # outside $NIGHT, so a dry run can delete $NIGHT safely
cp "$CFG" "$BEFORE" && cp "$CFG" "$NIGHT/default.yaml.before"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

RESTORED=0
restore_cfg() {
  [ "$RESTORED" = 1 ] && return
  if cp "$BEFORE" "$CFG"; then RESTORED=1
    log "config restored: $(grep -E '^  (network|random_seed|pooling|topk_frac|conv_cls|multilayer|stage_pooling|learning_rate|weight_decay):' "$CFG" | sed 's/ #.*//' | tr -s ' ' | tr '\n' ' ')"
  else log "WARNING: could not restore $CFG from $BEFORE; check it by hand"; fi
}
finish() {
  restore_cfg
  if [ "$AUTOSTOP" = 1 ] && [ "$DRY" = 0 ]; then
    log "stopping server in 60 s"; sleep 60
    curl -s -X DELETE -H "Authorization: token $(cat "$HOME/.hub_token")" \
      "$JUPYTERHUB_API_URL/users/${JUPYTERHUB_USER/@/%40}/server"
  fi
}
trap finish EXIT

set_cfg() {  # seed pooling conv_cls multilayer network stage_pooling topk_frac learning_rate weight_decay
  sed -i "s/^  learning_rate: .*/  learning_rate: $8/; s/^  weight_decay: .*/  weight_decay: $9/; s/^  topk_frac: .*/  topk_frac: $7/; s/^  random_seed: .*/  random_seed: $1/; s/^  pooling: .*/  pooling: $2/; s/^  conv_cls: .*/  conv_cls: $3/; s/^  multilayer: .*/  multilayer: $4/; s/^  network: .*/  network: $5/; s/^  stage_pooling: .*/  stage_pooling: $6/" "$CFG"
}
check_cfg() {  # seed pooling conv_cls multilayer network stage_pooling topk_frac learning_rate weight_decay -> 0 if every setting is as expected
  local ok=0
  for pat in "^  dataset: Fundus" "^  device: cuda" "^  test: false" "^  network: $5(\s|#|$)" "^  epochs: 70" \
             "^  lambda_l1: 0(\s|#|$)" "^  lambda_l2: 0(\s|#|$)" "^  random_seed: $1(\s|#|$)" "^  pooling: $2(\s|#|$)" "^  conv_cls: $3(\s|#|$)" "^  multilayer: $4(\s|#|$)" "^  stage_pooling: $6(\s|#|$)" "^  topk_frac: $7(\s|#|$)" \
             "^  learning_rate: $8(\s|#|$)" "^  weight_decay: $9(\s|#|$)" "^  lr_scheduler: clipped_cosine(\s|#|$)"; do
    grep -qE "$pat" "$CFG" || { log "CONFIG CHECK FAILED: no line matching '$pat'"; ok=1; }
  done
  return $ok
}

MAN="$NIGHT/manifest.csv"
echo "label,seed,pooling,conv_cls,timestamp" > "$MAN"
# default prior: the earlier manifest with the most runs (dry runs and partial nights have fewer)
PRIOR="${PRIOR_MANIFEST:-$(for f in "$HOME"/analysis/idrid/seeds_*/manifest.csv; do [ -f "$f" ] && [ "$f" != "$MAN" ] && echo "$(wc -l < "$f") $f"; done | sort -n | tail -1 | cut -d' ' -f2-)}"
if [ -n "$PRIOR" ] && [ -f "$PRIOR" ]; then
  tail -n +2 "$PRIOR" >> "$MAN"; log "prior runs from $PRIOR: $(($(wc -l < "$MAN") - 1))"
else   # first IDRiD runs (seed 0)
  printf '%s\n' "Mean SC,0,mean,True,26-10-2026_01:36:04" "Top-k SC,0,topk,True,24-10-2026_06:27:36" \
    "Black-box,0,mean,False,24-10-2026_06:32:01" >> "$MAN"
fi
spec_of() { case "$1" in
  mean)     echo "Mean SC|mean|True|none|resnet50|none" ;;
  topk)     echo "Top-k SC|topk|True|none|resnet50|none" ;;
  bb)       echo "Black-box|mean|False|none|resnet50|none" ;;
  multi)    echo "Multi SC|mean|True|3,4|resnet50|none" ;;
  s3)       echo "S3 SC|mean|True|3|resnet50|none" ;;
  multitk3) echo "Multi SC s3-topk|mean|True|3,4|resnet50|topk,mean" ;;
  s3tk)     echo "S3 SC topk|mean|True|3|resnet50|topk" ;;
  multitk34) echo "Multi SC both-topk|mean|True|3,4|resnet50|topk,topk" ;;
  multitk3_k1)  echo "Multi SC s3-topk 1%|mean|True|3,4|resnet50|topk,mean|0.01" ;;
  multitk3_k10) echo "Multi SC s3-topk 10%|mean|True|3,4|resnet50|topk,mean|0.10" ;;
  mean_lr3e4) echo "Mean SC lr 0.0003|mean|True|none|resnet50|none|0.05|0.0003" ;;
  mean_lr3e3) echo "Mean SC lr 0.003|mean|True|none|resnet50|none|0.05|0.003" ;;
  mean_lr1e2) echo "Mean SC lr 0.01|mean|True|none|resnet50|none|0.05|0.01" ;;
  swin)     echo "Swin SC|mean|True|none|swin_tiny_patch4_window7_224|none" ;;
  swinml)   echo "Swin Multi SC|mean|True|3,4|swin_tiny_patch4_window7_224|none" ;;
  *)     echo "" ;;
esac; }

log "models: $MODELS  seeds: $SEEDS  -> $NIGHT"
for seed in $SEEDS; do
  for key in $MODELS; do
    spec=$(spec_of "$key"); [ -n "$spec" ] || { log "unknown model '$key' (use mean, topk, bb, multi, s3, multitk3, s3tk, multitk34, multitk3_k1, multitk3_k10, mean_lr3e4, mean_lr3e3, mean_lr1e2, swin, swinml)"; continue; }
    IFS='|' read -r label pool conv ml net sp kf lr wd <<< "$spec"; kf="${kf:-0.05}"; lr="${lr:-0.001}"; wd="${wd:-0.0005}"
    if grep -q "^$label,$seed," "$MAN"; then log "have $label seed $seed already, not retraining"; continue; fi
    slug=$(echo "$label" | tr 'A-Z ' 'a-z_' | tr -d '-')
    set_cfg "$seed" "$pool" "$conv" "$ml" "$net" "$sp" "$kf" "$lr" "$wd"
    if ! check_cfg "$seed" "$pool" "$conv" "$ml" "$net" "$sp" "$kf" "$lr" "$wd"; then log "SKIP $label seed $seed (config check failed)"; continue; fi
    if [ "$DRY" = 1 ]; then log "DRY: would train $label seed $seed"; continue; fi
    free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
    if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then log "LOW DISK: ${free_gb} GB free in \$HOME (< $MIN_FREE_GB); no more training"; break 2; fi
    before=$(ls "$OUTS")
    log "TRAIN $label seed $seed"
    (cd "$REPO" && python main.py > "$HOME/logs/idrid_${slug}_s${seed}.log" 2>&1); rc=$?
    new=$(comm -13 <(echo "$before" | sort) <(ls "$OUTS" | sort) | tail -1)
    if [ $rc -eq 0 ] && [ -n "$new" ] && [ -f "$OUTS/$new/best_validation_weights_acc_m.pt" ]; then
      echo "$label,$seed,$pool,$conv,$new" >> "$MAN"; NEWRUNS="${NEWRUNS:-} $new"
      if [ "$KEEP_ALL_CKPTS" = 0 ]; then   # other checkpoints are ~95 MB each and unused by the analysis
        find "$OUTS/$new" -maxdepth 1 -type f -name '*.pt' ! -name 'best_validation_weights_acc_m.pt' -delete
      fi
      log "  done -> $new  ($(grep -A16 'best Accuracy' "$HOME/logs/idrid_${slug}_s${seed}.log" | grep -m1 'test acc'))"
    else
      log "  FAILED (exit $rc, folder '$new'); see ~/logs/idrid_${slug}_s${seed}.log"
    fi
  done
done
if [ "$DRY" = 1 ]; then restore_cfg; rm -rf "${NIGHT:?}"; log "dry run: nothing trained, $NIGHT removed"; exit 0; fi

log "EVALUATE (runs without results, plus this night's)"
tail -n +2 "$MAN" | while IFS=, read -r label seed pool conv ts; do
  if [ -f "$HOME/analysis/idrid/$ts/summary.json" ] && [[ " ${NEWRUNS:-} " != *" $ts "* ]]; then continue; fi
  python "$CODE/eval_idrid_localization.py" --run "$ts" > "$NIGHT/eval_$ts.log" 2>&1 \
    && log "  eval ok $label s$seed ($ts): $(grep -m1 '^faithfulness' "$NIGHT/eval_$ts.log" || echo black-box)" \
    || log "  eval FAILED $ts (see $NIGHT/eval_$ts.log)"
done

log "TRAIN/VAL/TEST SCORES (runs without splits.json)"
python "$CODE/eval_idrid_splits.py" --manifest "$MAN" > "$NIGHT/splits.log" 2>&1 \
  && log "  splits ok ($(grep -c ' test ' "$NIGHT/splits.log") runs scored; failures: $(grep -c '^FAILED' "$NIGHT/splits.log"))" \
  || log "  splits FAILED (see $NIGHT/splits.log)"

log "FIGURES (seed 0)"
case " $MODELS " in
  *" s3tk "*) FIGL="${FIG_LABELS:-Mean SC,S3 SC topk,Multi SC s3-topk}" ;;
  *" s3 "*|*" multitk3 "*) FIGL="${FIG_LABELS:-Mean SC,S3 SC,Multi SC s3-topk}" ;;
  *" swin"*)   FIGL="${FIG_LABELS:-Mean SC,Swin SC,Swin Multi SC}" ;;
  *" multi "*) FIGL="${FIG_LABELS:-Mean SC,Multi SC,Black-box}" ;;
  *)           FIGL="${FIG_LABELS:-Mean SC,Top-k SC,Black-box}" ;;
esac
FIGARGS=()
IFS=',' read -ra FL <<< "$FIGL"
for l in "${FL[@]}"; do
  t=$(grep "^$l,0," "$MAN" | tail -1 | cut -d, -f5); [ -n "$t" ] && FIGARGS+=(--run "$l=$t")
done
python "$CODE/make_idrid_figures.py" "${FIGARGS[@]}" --out "$NIGHT/figures" > "$NIGHT/figures.log" 2>&1 \
  && log "  figures ok -> $NIGHT/figures" || log "  figures FAILED (see $NIGHT/figures.log)"

log "AGGREGATE"
python "$CODE/aggregate_idrid_seeds.py" --manifest "$MAN" --out "$NIGHT" > "$NIGHT/aggregate.log" 2>&1 \
  && { log "  aggregate ok"; cat "$NIGHT/SEEDS_SUMMARY.md"; } || log "  aggregate FAILED (see $NIGHT/aggregate.log)"

tar czf "$HOME/analysis/idrid_night_$STAMP.tar.gz" -C "$HOME/analysis" idrid && log "packed ~/analysis/idrid_night_$STAMP.tar.gz"
log "ALL DONE"

#!/bin/bash
# Noise-aware fine-tuning pipeline. Resumable: every stage is skipped when its output already
# exists, and every scoring step is cached, so re-running after an interruption continues from
# the last completed stage. Strictly sequential, so it also runs on a small laptop.
#
# Run detached so it survives the terminal/agent session:  nohup ./pipeline.sh > logs/pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python -I}"      # override on other machines, e.g. PY="python -I"
mkdir -p logs
stage() { echo "[$(date '+%F %T')] >>> $1"; }
# Keep the Mac awake for exactly as long as this script runs (-s only takes effect on AC power).
command -v caffeinate >/dev/null && caffeinate -i -s -w $$ &   # macOS only

if [ "$($PY -c "import json;print(len(json.load(open('cache/trainset.json'))['examples']))" 2>/dev/null || echo 0)" -ge 1320 ]; then
  stage "training set already built (1320 examples) — skipping"
else
  stage "building training set (train-split queries only, TTS -> ASR, BM25 hard negatives)"
  $PY build_trainset.py >> logs/trainset.log 2>&1
fi

# A variant is complete when its log exists AND its fixed-budget weights are recoverable: either
# saved as models/ft-<v>-s0-last, or identical to the early-stopped model (best epoch = last).
SEEDS="${SEEDS:-0}"                       # e.g. SEEDS="0 1 2"; each seed is an independent run
complete() {
  $PY -c "import json,sys,os; l=json.load(open('results/training/ft-$1-s$2.json')); \
sys.exit(0 if l['best_epoch']==int(l['args']['epochs']) or os.path.isdir('models/ft-$1-s$2-last') else 1)" 2>/dev/null
}
for seed in $SEEDS; do
  for v in clean noisy mixed; do
    if complete "$v" "$seed"; then
      stage "fine-tune $v seed $seed already complete (both protocols recoverable) — skipping"
      continue
    fi
    stage "fine-tuning variant: $v (seed $seed)"
    $PY finetune.py --variant "$v" --seed "$seed" > "logs/ft-$v-s$seed.log" 2>&1
    grep -E "step 0|epoch [0-9] val|done in" "logs/ft-$v-s$seed.log"
  done
done

MODELS=$($PY select_models.py $SEEDS 2>> logs/pipeline.log)
stage "evaluating unique checkpoints ($MODELS) at depth 25, all 3 query conditions"
$PY run_bench.py --conds text asr_clean asr_phone --models $MODELS \
  --pool both --pool-depth 25 > logs/eval-ft.log 2>&1
grep -E "^==|evaluated" logs/eval-ft.log

if [ -n "${SKIP_EXTRAS:-}" ]; then
  $PY analyze.py > logs/analyze.log 2>&1
  stage "pipeline complete (extras skipped: SKIP_EXTRAS set)"
  exit 0
fi

stage "abstention probe (out-of-domain queries)"
$PY run_bench.py --conds text --models ce-l6 --pool both --pool-depth 25 --ood > logs/ood.log 2>&1

stage "ablations: model depth (L12) and input budget (512 tokens), BM25 top-25, typed queries"
$PY run_bench.py --conds text --models ce-l12 ce-l6-512 --pool bm25 --pool-depth 25 > logs/ablations.log 2>&1

stage "analysis"
$PY analyze.py > logs/analyze.log 2>&1

stage "latency benchmark (machine should now be otherwise idle)"
$PY bench_latency.py > logs/latency.log 2>&1
$PY analyze.py > logs/analyze.log 2>&1
stage "pipeline complete"

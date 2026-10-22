#!/bin/bash
# Part 2 pipeline (Kaggle). MODE=A: Part 1 rerankers on SD-QA real speech.
#                           MODE=B: consistency distillation — train, evaluate on BEIR and SD-QA.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python -I}"
MODE="${MODE:?set MODE=A or MODE=B}"
SEEDS="${SEEDS:-0 1 2}"
mkdir -p logs
stage() { echo "[$(date '+%F %T')] >>> $1"; }

stage "building SD-QA real-speech benchmark"
$PY build_sdqa.py > logs/sdqa_build.log 2>&1
grep -E "SD-QA test|articles" logs/sdqa_build.log
CONDS="text $($PY -c "import json;print(' '.join(json.load(open('cache/sdqa/conditions.json'))))")"

train() {   # train <variant> <seed>, unless its fixed-budget checkpoint already exists
  if [ -d "models/ft-$1-s$2-last" ] && [ -f "results/training/ft-$1-s$2.json" ]; then
    stage "ft-$1-s$2 present — skipping"; return
  fi
  stage "fine-tuning $1 seed $2"
  $PY finetune.py --variant "$1" --seed "$2" > "logs/ft-$1-s$2.log" 2>&1
  grep -E "step 0|epoch [0-9] val|done in" "logs/ft-$1-s$2.log"
}

if [ "$MODE" = "A" ]; then
  MODELS="ce-l6"
  for s in $SEEDS; do for v in clean noisy mixed; do train "$v" "$s"; MODELS="$MODELS ft-$v-s$s-last"; done; done
  stage "SD-QA evaluation (fp16): $MODELS"
  BENCH_FP16=1 $PY run_bench.py --datasets sdqa --conds $CONDS --models $MODELS --pool both --pool-depth 25 \
    > logs/eval-sdqa-A.log 2>&1
  grep -E "^==|evaluated" logs/eval-sdqa-A.log | tail -4
else
  for s in $SEEDS; do train distill "$s"; done
  BEIR_MODELS=$(VARIANTS=distill $PY select_models.py $SEEDS 2>> logs/pipeline.log)
  stage "BEIR evaluation of distillation (fp32, as Part 1): $BEIR_MODELS"
  $PY run_bench.py --conds text asr_clean asr_phone --models $BEIR_MODELS --pool both --pool-depth 25 \
    > logs/eval-beir-B.log 2>&1
  grep -E "^==|evaluated" logs/eval-beir-B.log
  SDQA_MODELS="ce-l6"; for s in $SEEDS; do SDQA_MODELS="$SDQA_MODELS ft-distill-s$s-last"; done
  stage "SD-QA evaluation of distillation (fp16): $SDQA_MODELS"
  BENCH_FP16=1 $PY run_bench.py --datasets sdqa --conds $CONDS --models $SDQA_MODELS --pool both --pool-depth 25 \
    > logs/eval-sdqa-B.log 2>&1
  grep -E "^==|evaluated" logs/eval-sdqa-B.log | tail -4
fi
stage "pipeline complete"

#!/bin/bash
# Part 3 pipeline (Kaggle, GPU). Plan and tests fixed in paper/PREREGISTRATION_PART3.md.
#   MODE=C  data: SD-QA Whisper transcripts, HeySQuAD and SLUE-SQA-5 test sets, HeySQuAD
#           real-vs-synthetic training set, hybrid negatives for the BEIR training set.
#   MODE=E  modern reranker on BEIR (C2): bge-reranker-base FT-clean/noisy/distill x 3 seeds,
#           and zero-shot bge-reranker-base / bge-reranker-v2-m3, on typed + spoken BEIR.
#   MODE=H  real vs synthetic training noise (C3): ce-l6 trained on HeySQuAD in 5 ways x 3 seeds,
#           evaluated on every Part 3 test set.                                [needs MODE=C data]
#   MODE=S  stronger distillation (C6) selected on validation, plus zero-shot ce-l6 / bge-base /
#           bge-m3 on every Part 3 test set (C1).                              [needs MODE=C data]
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python -I}"
MODE="${MODE:?set MODE=C, E, H or S}"
SEEDS="${SEEDS:-0 1 2}"
mkdir -p logs results/training
stage() { echo "[$(date '+%F %T')] >>> $1"; }

train() {   # train <name> <args...>, unless its fixed-budget checkpoint already exists
  local name="$1"; shift
  if [ -d "models/$name-last" ] && [ -f "results/training/$name.json" ]; then
    stage "$name present — skipping"; return
  fi
  stage "training $name: $*"
  $PY finetune.py --name "$name" "$@" > "logs/$name.log" 2>&1
  grep -E "step 0|epoch [0-9] val|done in" "logs/$name.log"
}

evaluate() {  # evaluate <tag> <datasets...> -- <models...>
  local tag="$1"; shift
  local datasets=() models=()
  while [ "$1" != "--" ]; do datasets+=("$1"); shift; done; shift
  models=("$@")
  local conds="text asr_clean asr_phone asr_wb asr_orig"
  if [ -f cache/sdqa/conditions.json ]; then
    conds="$conds $($PY -c "import json;print(' '.join(json.load(open('cache/sdqa/conditions.json'))))")"
  fi
  if [ -f cache/sdqa/conditions_whisper.json ]; then
    conds="$conds $($PY -c "import json;print(' '.join(json.load(open('cache/sdqa/conditions_whisper.json'))))")"
  fi
  stage "evaluation $tag (fp16): ${datasets[*]} :: ${models[*]}"
  BENCH_FP16=1 $PY run_bench.py --datasets "${datasets[@]}" --conds $conds --models "${models[@]}" \
    --pool both --pool-depth 25 > "logs/eval-$tag.log" 2>&1
  grep -E "^==|evaluated|skip" "logs/eval-$tag.log" | tail -60
}

NEW_SETS="sdqa heysquad slue"
BEIR="scifact nfcorpus fiqa"

case "$MODE" in
C)
  stage "SD-QA corpus (Part 2 builder)"
  $PY build_sdqa.py > logs/sdqa_build.log 2>&1
  stage "Part 3 data"
  $PY part3_data.py sdqa heysquad slue beirhyb 2>&1 | tee logs/part3_data.log
  ;;
E)
  MODELS="bge-base bge-m3"
  for s in $SEEDS; do
    for v in clean noisy distill; do
      train "ft-bgeb-$v-s$s" --base bge-base --fp16 --last-only --variant "$v" --seed "$s"
      MODELS="$MODELS ft-bgeb-$v-s$s-last"
    done
  done
  evaluate beir-E $BEIR -- $MODELS
  ;;
H)
  $PY build_sdqa.py > logs/sdqa_build.log 2>&1
  HQ="--trainset cache/trainset_heysquad.json"
  MODELS="ce-l6"
  for s in $SEEDS; do
    train "ft-hq-clean-s$s"  $HQ --variant clean   --noisy-form real  --seed "$s"
    train "ft-hq-real-s$s"   $HQ --variant noisy   --noisy-form real  --seed "$s"
    train "ft-hq-synth-s$s"  $HQ --variant noisy   --noisy-form synth --seed "$s"
    train "ft-hq-dreal-s$s"  $HQ --variant distill --noisy-form real  --seed "$s"
    train "ft-hq-dsynth-s$s" $HQ --variant distill --noisy-form synth --seed "$s"
    for v in clean real synth dreal dsynth; do MODELS="$MODELS ft-hq-$v-s$s-last"; done
  done
  evaluate new-H $NEW_SETS -- $MODELS
  evaluate beir-H $BEIR -- $MODELS
  ;;
S)
  $PY build_sdqa.py > logs/sdqa_build.log 2>&1
  stage "C1/C2 zero-shot rerankers on the Part 3 test sets"
  evaluate new-zero $NEW_SETS -- ce-l6 bge-base bge-m3
  T="--variant distill --teacher bge-m3 --seed 0"
  train ft-d1-s0   $T --candidates bm25   --temperature 1
  train ft-d2t1-s0 $T --candidates hybrid --temperature 1
  train ft-d2t2-s0 $T --candidates hybrid --temperature 2
  train ft-d2t4-s0 $T --candidates hybrid --temperature 4
  BEST_D2=$($PY select_distill.py ft-d2t1-s0 ft-d2t2-s0 ft-d2t4-s0 2> logs/select-d2.log)
  TAU=${BEST_D2#ft-d2t}; TAU=${TAU%-s0}
  stage "best D2 on validation: $BEST_D2 (tau=$TAU)"
  # D3: the best D2 on BEIR + HeySQuAD real speech. HeySQuAD rows name their transcript "real".
  $PY -c "
import json
hq = json.load(open('cache/trainset_heysquad.json'))
for e in hq['examples']:
    e['noisy'] = e['real']
    e['negatives'] = {'clean': e['negatives']['clean'], 'noisy': e['negatives']['real']}
    e['negatives_hybrid'] = {'clean': e['negatives_hybrid']['clean'], 'noisy': e['negatives_hybrid']['real']}
json.dump(hq, open('cache/trainset_heysquad_as_noisy.json', 'w'))"
  train ft-d3-s0 $T --candidates hybrid --temperature "$TAU" \
    --trainset cache/trainset.json --trainset cache/trainset_heysquad_as_noisy.json
  BEST=$($PY select_distill.py ft-d1-s0 "$BEST_D2" ft-d3-s0 2> logs/select-final.log)
  stage "selected configuration on validation: $BEST"
  cat results/training/distill_selection.json; echo
  ARGS=$($PY -c "import json;a=json.load(open('results/training/$BEST.json'))['args'];print(' '.join(['--candidates',a['candidates'],'--temperature',str(a['temperature'])]+sum([['--trainset',t] for t in a['trainset']],[])))")
  MODELS="ce-l6"
  for s in $SEEDS; do
    train "ft-dbest-s$s" --variant distill --teacher bge-m3 --seed "$s" $ARGS
    MODELS="$MODELS ft-dbest-s$s-last"
  done
  evaluate new-S $NEW_SETS -- $MODELS
  evaluate beir-S $BEIR -- $MODELS
  ;;
esac
stage "pipeline complete"

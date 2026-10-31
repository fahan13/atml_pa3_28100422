#!/usr/bin/env bash
SESSION=$1
PREV=$2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
if [ -n "$PREV" ]; then
  for d in outputs results logs; do [ -d "$PREV/$d" ] && cp -rn "$PREV/$d" ./; done
fi
python -m scripts.check_environment --out results/env/kaggle_task1_session$SESSION.json
python -m task1_ft.data --config configs/task1/base.yaml

run() { CUDA_VISIBLE_DEVICES=$1 python -m task1_ft.train --config configs/task1/$2.yaml >> logs/task1_train_$2.log 2>&1; }
case $SESSION in
  A) ( run 0 full_ft; run 0 lora4 ) & ( run 1 oplora16 ) & ;;
  B) ( run 0 l2sp ) & ( run 1 lora16 ) & ;;
  *) echo "unknown session $SESSION"; exit 1 ;;
esac

while [ -n "$(jobs -rp)" ]; do
  sleep 600
  date
  for f in logs/task1_train_*.log; do echo "$f: $(grep -a '^\[' "$f" | tail -n 1)"; done
done
for f in logs/task1_train_*.log; do echo "== $f"; tail -n 2 "$f"; done

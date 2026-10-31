#!/usr/bin/env bash
# env: COCO_ROOTS="dir1 dir2" (COCO 2014 images and, if present, annotations), PREV=<earlier task2 output repo dir>, SMOKE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs outputs/task2
if [ -n "$PREV" ]; then
  for d in data features; do [ -d "$PREV/outputs/task2/$d" ] && [ ! -e "outputs/task2/$d" ] && ln -s "$PREV/outputs/task2/$d" "outputs/task2/$d"; done
  for d in results logs; do [ -d "$PREV/$d" ] && cp -rn "$PREV/$d" ./; done
fi
python -m scripts.check_environment --out results/env/kaggle_task2.json
python -m task2_vlm.build_data --roots $COCO_ROOTS 2>&1 | tee -a logs/task2_build.log
CUDA_VISIBLE_DEVICES=0 python -m task2_vlm.cache_features --roots $COCO_ROOTS 2>&1 | tail -n 3 | tee -a logs/task2_cache.log
if [ "$SMOKE" = "1" ]; then
  EXTRA="--updates 40,40,40 --dev_every 40 --ckpt_every 20"; N196="--name smoke196"; N49="--name smoke49"
else
  EXTRA=""; N196=""; N49=""
fi
( CUDA_VISIBLE_DEVICES=0 python -m task2_vlm.train $EXTRA $N196 >> logs/task2_train_196.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python -m task2_vlm.evaluate $N196 >> logs/task2_eval_196.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 python -m task2_vlm.train --config configs/task2/compress49.yaml $EXTRA $N49 >> logs/task2_train_49.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python -m task2_vlm.evaluate --config configs/task2/compress49.yaml $N49 >> logs/task2_eval_49.log 2>&1 ) &
while [ -n "$(jobs -rp)" ]; do
  sleep 600
  date
  for f in logs/task2_train_*.log; do echo "$f: $(grep -a '^\[' "$f" | tail -n 1)"; done
done
for f in logs/task2_train_*.log logs/task2_eval_*.log; do echo "== $f"; tail -n 4 "$f"; done

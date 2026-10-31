#!/usr/bin/env bash
mkdir -p logs results/task3/live
python -m scripts.check_environment --out results/env/kaggle_task3_live.json
CUDA_VISIBLE_DEVICES=0 python -m task3_ttc.live_generate --shard 0 --n_shards 2 >> logs/task3_live_shard0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m task3_ttc.live_generate --shard 1 --n_shards 2 >> logs/task3_live_shard1.log 2>&1 &
while [ -n "$(jobs -rp)" ]; do
  sleep 300
  date
  for f in logs/task3_live_shard*.log; do echo "$f: $(grep -a 'done' "$f" | tail -n 1)"; done
done
wc -l results/task3/live/calls_shard*.jsonl

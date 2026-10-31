#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs outputs/task1
for root in "$@"; do
  for d in "$root"/outputs/task1/*/; do
    n=$(basename "$d")
    [ "$n" = data ] && continue
    [ -e "outputs/task1/$n" ] || ln -s "$d" "outputs/task1/$n"
  done
  for d in results logs; do [ -d "$root/$d" ] && cp -rn "$root/$d" ./; done
done
ls -l outputs/task1
python -m scripts.check_environment --out results/env/kaggle_task1_eval.json
CUDA_VISIBLE_DEVICES=0 python -m task1_ft.merge > logs/task1_merge.log 2>&1
tail -n 5 logs/task1_merge.log

ev() { CUDA_VISIBLE_DEVICES=$1 python -m task1_ft.evaluate --conditions $2 >> logs/task1_eval_gpu$1.log 2>&1; }
ql() { CUDA_VISIBLE_DEVICES=$1 python -m task1_ft.qualitative --conditions $2 >> logs/task1_qual_gpu$1.log 2>&1; }
( ev 0 baseline,full_ft_mid,full_ft,lora4; ql 0 baseline,full_ft_mid,full_ft,lora4 ) &
( ev 1 l2sp,lora16,oplora16; ql 1 l2sp,lora16,oplora16
  CUDA_VISIBLE_DEVICES=1 python -m task1_ft.diagnostics > logs/task1_diagnostics.log 2>&1 ) &

while [ -n "$(jobs -rp)" ]; do
  sleep 600
  date
  ls results/task1/eval results/task1/qualitative 2>/dev/null | tr '\n' ' '; echo
done
for f in logs/task1_eval_gpu*.log logs/task1_qual_gpu*.log logs/task1_diagnostics.log; do echo "== $f"; grep -a "^{\|skipping\|responses\|Error\|error" "$f" | tail -n 8; done
rm -rf outputs/task1/merged

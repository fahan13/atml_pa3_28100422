#!/usr/bin/env bash
set -e
mkdir -p logs
python -m scripts.check_environment --out results/env/kaggle_task1.json
python -m task1_ft.data --config configs/task1/base.yaml 2>&1 | tee logs/task1_data.log

CUDA_VISIBLE_DEVICES=0 python -m task1_ft.train --config configs/task1/full_ft.yaml --memory_probe > logs/probe_full_ft.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m task1_ft.train --config configs/task1/l2sp.yaml --memory_probe > logs/probe_l2sp.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python -m task1_ft.train --config configs/task1/lora16.yaml --memory_probe > logs/probe_lora16.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m task1_ft.train --config configs/task1/oplora16.yaml --memory_probe > logs/probe_oplora16.log 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python -m task1_ft.train --config configs/task1/l2sp.yaml --name smoke_l2sp --stop_after 6 --checkpoint_updates 3 --export_updates 6 > logs/smoke_l2sp.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m task1_ft.train --config configs/task1/oplora16.yaml --name smoke_oplora16 --stop_after 6 --checkpoint_updates 3 --export_updates 6 > logs/smoke_oplora16.log 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python -m task1_ft.train --config configs/task1/lora16.yaml --name smoke_resume_a --stop_after 4 --checkpoint_updates 2,4 --export_updates "" > logs/smoke_resume_a.log 2>&1 &
( CUDA_VISIBLE_DEVICES=1 python -m task1_ft.train --config configs/task1/lora16.yaml --name smoke_resume_b --stop_after 2 --checkpoint_updates 2,4 --export_updates "" \
  && CUDA_VISIBLE_DEVICES=1 python -m task1_ft.train --config configs/task1/lora16.yaml --name smoke_resume_b --stop_after 4 --checkpoint_updates 2,4 --export_updates "" ) > logs/smoke_resume_b.log 2>&1 &
wait
python -m tests.test_resume --config configs/task1/lora16.yaml --straight smoke_resume_a --resumed smoke_resume_b

for f in logs/probe_*.log logs/smoke_*.log; do echo "== $f"; tail -n 3 "$f"; done
rm -rf outputs/task1/smoke_*

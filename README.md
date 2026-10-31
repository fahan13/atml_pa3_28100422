# Fine-Tuning, Vision-Language Models and Test-Time Compute

Programming Assignment 3 — EE-5102 / CS-6304, Advanced Topics in Machine Learning
Fahan Hassan Khan (28100422), LUMS

| Task | Topic | Models / data |
|------|-------|---------------|
| 1 | LLM fine-tuning: Full FT, Full FT + L2-SP, LoRA-4/16, OPLoRA-16 | Qwen3-0.6B, MetaMathQA; GSM8K, HellaSwag, BoolQ, ARC-Easy |
| 2 | Vision-language model and visual grounding | SigLIP-base-patch16-224 + Qwen3-0.6B; COCO 2014, VQAv2 |
| 3 | Test-time compute: sampling, selection, adaptive inference | Supplied Qwen2.5-3B / Qwen2.5-VL-3B rollouts (MATH, MathVista); Qwen2.5-3B-Instruct |

Seed 6304 everywhere. Every number in the report is read from a file under `results/`.

## 1. Repository layout

```
common/      shared helpers (config loading, seeding, IO, timing, memory)
configs/     one YAML per run; task1/base.yaml holds the shared recipe
task1_ft/    data.py (MetaMathQA selection), lora.py (LoRA / OPLoRA), train.py (all five variants),
             merge.py, evaluate.py, qualitative.py, diagnostics.py
task3_ttc/   answers.py (math-verify / MathVista normalization), prepare.py, select.py, offline.py,
             signals.py, adaptive.py, live_generate.py, live_analyze.py
tests/       sanity checks; outputs in results/sanity/
scripts/     environment check and Kaggle run scripts
report/      figure scripts and figures
results/     machine-readable results, one folder per task and run
logs/        console logs of every run
```

Model weights, recovery checkpoints and caches go to `outputs/` and are not committed.

## 2. Environment

GPU runs: Kaggle notebook, 2× Tesla T4 (15 GB), Internet on. CPU-only analysis: Windows laptop.

```bash
pip install -r requirements.txt
python -m scripts.check_environment
```

The exact versions used for each task are saved under `results/env/`.

## 3. Reproducing each task

### Task 1

```bash
python -m task1_ft.data --config configs/task1/base.yaml
python -m task1_ft.train --config configs/task1/full_ft.yaml
python -m task1_ft.train --config configs/task1/l2sp.yaml
python -m task1_ft.train --config configs/task1/lora4.yaml
python -m task1_ft.train --config configs/task1/lora16.yaml
python -m task1_ft.train --config configs/task1/oplora16.yaml
```

Evaluation (merges adapters into fresh FP16 backbones, runs lm-evaluation-harness, qualitative generations and weight/activation diagnostics):

```bash
python -m task1_ft.merge
python -m task1_ft.evaluate --conditions baseline,full_ft_mid,full_ft,l2sp,lora4,lora16,oplora16
python -m task1_ft.qualitative --conditions baseline,full_ft_mid,full_ft,l2sp,lora4,lora16,oplora16
python -m task1_ft.diagnostics
```

On Kaggle, `scripts/run_task1_train.sh A|B` runs the training sessions and `scripts/run_task1_eval.sh <session A output> <session B output>` runs the full evaluation on both GPUs.

Training resumes from the latest recovery checkpoint in `outputs/task1/<run>/ckpt/` and skips runs that are already complete. `--resume_from <path>` resumes from a checkpoint copied from an earlier session.

### Task 2

To be added.

### Task 3

Offline experiments (CPU; the supplied `ATML-PA3-Data` folder must be in `data/`):

```bash
python -m task3_ttc.prepare     # join and verify the three files, answer groups and grades, splits, rollout orders
python -m task3_ttc.offline     # Steps 1-2: fixed budgets N = 1..32, all selection rules, oracle coverage, bootstrap CIs
python -m task3_ttc.signals     # Step 3: length/score associations, confidence-vision correlation, tokenwise fusion
python -m task3_ttc.adaptive    # Step 5: development-selected stopping rules, held-out evaluation
```

Live generation (Step 4, GPU) and its analysis (CPU):

```bash
bash scripts/run_task3_live.sh          # Kaggle, one shard of the 50 questions per T4
python -m task3_ttc.live_analyze
```

## 4. External code and data

- `configs/task1/first_plot_questions.yaml`: the 24 qualitative prompts from the Emergent Misalignment evaluation suite (github.com/emergent-misalignment/emergent-misalignment, `evaluation/first_plot_questions.yaml`, commit `80c1196`).
- Models and datasets are loaded from the Hugging Face Hub: `Qwen/Qwen3-0.6B`, `meta-math/MetaMathQA`, `openai/gsm8k`, `Rowan/hellaswag`.
- Benchmarks use EleutherAI lm-evaluation-harness 0.4.13 with stock task definitions (`gsm8k_cot`, `hellaswag`, `boolq`, `arc_easy`). The HellaSwag context preprocessing in `task1_ft/diagnostics.py` reproduces the harness's `hellaswag/utils.py` so probe prompts match the benchmark format.
- Task 3 MATH equivalence uses `math-verify` 0.9.0. MathVista normalization in `task3_ttc/answers.py` ports `normalize_extracted_answer` and `get_most_similar` from the official MathVista evaluation code (github.com/lupantech/MathVista, `evaluation/calculate_score.py`); question metadata is the official `data/testmini.json` at commit `c5acd37`, downloaded and checksum-verified by `task3_ttc/prepare.py`.
- The LoRA and OPLoRA layers in `task1_ft/lora.py` are written for this assignment. The A-matrix initialization follows the uniform bound used by Hu et al. (LoRA) and PEFT; the projection follows Xiong and Xie (OPLoRA), Sections 3.2–3.3.

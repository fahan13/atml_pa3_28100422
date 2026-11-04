# Fine-Tuning, Vision-Language Models and Test-Time Compute

Programming Assignment 3 — EE-5102 / CS-6304, Advanced Topics in Machine Learning
Fahan Hassan Khan (28100422), LUMS

| Task | Topic | Models / data |
|------|-------|---------------|
| 1 | LLM fine-tuning: Full FT, Full FT + L2-SP, LoRA-4/16, OPLoRA-16 | Qwen3-0.6B, MetaMathQA; GSM8K, HellaSwag, BoolQ, ARC-Easy |
| 2 | Vision-language model and visual grounding | SigLIP-base-patch16-224 + Qwen3-0.6B; COCO 2014, VQAv2 |
| 3 | Test-time compute: sampling, selection, adaptive inference | Supplied Qwen2.5-3B / Qwen2.5-VL-3B rollouts (MATH, MathVista); Qwen2.5-3B-Instruct |

Seed 6304 everywhere. Every number in the report is read from a file under `results/`; the report's figures and
tables are regenerated from those files by `report/make_figures.py` and `report/make_tables.py` (§4).

## 1. Repository layout

```
common/      shared helpers (config loading, seeding, IO, timing, memory)
configs/     one YAML per run; task1/base.yaml and task2/base.yaml hold the shared recipes
task1_ft/    data.py (MetaMathQA selection), lora.py (LoRA / OPLoRA), train.py (all five variants),
             conditions.py, merge.py, evaluate.py, qualitative.py, diagnostics.py
task2_vlm/   vqa_norm.py, build_data.py, cache_features.py, model.py (connector, sequence construction, greedy decoding),
             train.py (curriculum), evaluate.py (image conditions, pair metrics, bootstrap)
task3_ttc/   answers.py (math-verify / MathVista normalization), boxed.py, prepare.py, select.py, offline.py,
             signals.py, adaptive.py, live_generate.py, live_analyze.py, inspect_data.py
tests/       resume test; sanity outputs in results/sanity/
scripts/     environment check and Kaggle run scripts
report/      make_figures.py, make_tables.py and their outputs (figures/*.pdf, tables/*.tex)
results/     machine-readable results, one folder per task and run
logs/        console logs of every run
```

Model weights, merged models, recovery checkpoints, feature caches and downloads go to `outputs/` and are not committed.
Raw datasets are not committed (`data/` holds only a README).

## 2. Environment

GPU runs: Kaggle notebooks, 2× Tesla T4 (15 GB), Internet on. CPU analysis and figures: Windows laptop (conda, Python 3.13).

```bash
pip install -r requirements.txt
python -m scripts.check_environment
```

`requirements.txt` pins `transformers==5.8.0`, `lm_eval==0.4.13` and `math-verify==0.9.0`; PyTorch is the version
preinstalled on Kaggle. The exact versions of every session are saved under `results/env/` and, for Task 3 live
generation, `results/task3/live/env_shard*.json`.

## 3. Reproducing each task

### Task 1

```bash
python -m task1_ft.data --config configs/task1/base.yaml        # first 40k eligible MetaMathQA examples, GSM8K-test overlap removed
python -m task1_ft.train --config configs/task1/full_ft.yaml
python -m task1_ft.train --config configs/task1/l2sp.yaml
python -m task1_ft.train --config configs/task1/lora4.yaml
python -m task1_ft.train --config configs/task1/lora16.yaml
python -m task1_ft.train --config configs/task1/oplora16.yaml
```

Evaluation of the seven conditions (base model, Full FT at update 625, and the five final models):

```bash
python -m task1_ft.merge                       # adapters merged into fresh FP16 backbones; merged-vs-unmerged logit check; OPLoRA constraint check
python -m task1_ft.evaluate --conditions baseline,full_ft_mid,full_ft,l2sp,lora4,lora16,oplora16
python -m task1_ft.qualitative --conditions baseline,full_ft_mid,full_ft,l2sp,lora4,lora16,oplora16
python -m task1_ft.diagnostics                 # rho_k, eta and R(z) for down_proj in layers 0, 13, 27
```

`python -m task1_ft.merge --conditions oplora16 --constraints_only` recomputes only the OPLoRA constraint check
(needs the exported adapters in `outputs/task1/oplora16/`).

On Kaggle, `scripts/run_task1_train.sh A|B` runs the two training sessions (two runs per session, one per GPU) and
`scripts/run_task1_eval.sh <session A output> <session B output>` runs the full evaluation on both GPUs.
`scripts/run_task1_smoke.sh` runs the short smoke tests (`results/task1/smoke_*`, not reported), including the
resume test: `tests/test_resume.py` checks that a run interrupted and resumed matches an uninterrupted one
(`results/sanity/task1_resume.json`).

Training resumes from the latest recovery checkpoint in `outputs/task1/<run>/ckpt/` and skips runs that are already
complete; `--resume_from <path>` resumes from a checkpoint copied from an earlier session.

### Task 2

```bash
python -m task2_vlm.build_data --roots <COCO 2014 folder>          # captions, template grounding pairs, VQAv2 complementary pairs, image-disjoint splits
python -m task2_vlm.cache_features --roots <COCO 2014 folder>      # SigLIP 196x768 patch features, FP16
python -m task2_vlm.train                                          # 3-stage curriculum, 196 visual tokens
python -m task2_vlm.evaluate                                       # final test suites under four image conditions
python -m task2_vlm.train --config configs/task2/compress49.yaml   # optional 49-token run (2x2 average pooling)
python -m task2_vlm.evaluate --config configs/task2/compress49.yaml
```

On Kaggle, `COCO_ROOTS=<input dir> bash scripts/run_task2.sh` runs all of the above, with the two runs on separate
GPUs (`SMOKE=1` gives the short smoke runs in `results/task2/smoke*`, not reported). VQAv2 questions, annotations and
complementary pairs and the COCO 2014 annotations are downloaded from the official servers if they are not found
under the given roots.

### Task 3

Offline experiments (CPU; the supplied `ATML-PA3-Data` folder must be in `data/`):

```bash
python -m task3_ttc.prepare     # join and verify the three files, answer groups and grades, dev/eval splits, rollout orders
python -m task3_ttc.offline     # fixed budgets N = 1..32, all selection rules, oracle coverage, bootstrap CIs
python -m task3_ttc.signals     # length/score associations, confidence-vision correlation, tokenwise fusion
python -m task3_ttc.adaptive    # stopping rules selected on the dev split, evaluated on the eval split
```

Live generation (GPU) and its analysis (CPU):

```bash
bash scripts/run_task3_live.sh          # Kaggle, one shard of the 50 questions per T4
python -m task3_ttc.live_analyze        # grading, strategy summaries, revision transitions, paired bootstrap
python -m task3_ttc.live_prompts        # rebuilds every prompt and checks it against the recorded input-token count
```

Selection and stopping rules use only the revealed rollouts' answers and scores; gold answers are used only for
grading.

## 4. Report figures and tables

```bash
python report/make_figures.py           # all figures -> report/figures/<name>.pdf (+ .png preview)
python report/make_tables.py            # all tables  -> report/tables/<name>.tex (booktabs tabular)
python report/make_examples.py          # qualitative examples -> report/examples/
python report/make_figures.py --list    # names; pass a name to build one
```

`t2_pairs` downloads the few COCO val2014 images it shows from images.cocodataset.org into `outputs/coco_examples/`.

| Output | Contents | Source files |
|---|---|---|
| `t1_benchmarks` | 7 conditions × 4 benchmarks, change from base in pp; GSM8K lenient check | `results/task1/eval/<cond>/scores.json` |
| `t1_training` | SFT loss of the five runs; GSM8K gain vs. retention | `results/task1/<run>/train_log.jsonl`, `eval/*/scores.json` |
| `t1_resources` | trainable parameters, time, peak memory, checkpoint sizes, displacement, method overheads | `results/task1/<run>/{run_summary,checkpoints,exports}.json(l)` |
| `t1_diagnostics` | ρ_k, η, R(z) at layers 0/13/27 | `results/task1/diagnostics/` |
| `t1_annotation` | 42-response qualitative annotation counts | `results/task1/qualitative/annotations.json`, `<cond>.jsonl` |
| `t2_curves` | training and dev losses, answer NLL by image condition, swap penalty | `results/task2/vlm196/{train_log,dev_log,stages}.jsonl` |
| `t2_results` | both suites × four image conditions, question types, pair metrics, NLL, bootstrap CIs | `results/task2/vlm196/final_eval_stage3.json` |
| `t2_resources` | 196 vs. 49 visual tokens: accuracy, time, memory, parameters, feature cache | `results/task2/vlm*/`, `results/task2/feature_cache.json` |
| `t3_scaling` | accuracy and oracle coverage vs. N, adaptive rules | `results/task3/<ds>/{offline_summary,adaptive}.json` |
| `t3_methods` | all selection rules at N = 4 and 32, difference from SC with CIs, failure split | `results/task3/<ds>/offline_summary.json` |
| `t3_signals` | score–correctness and score–length associations, confidence–vision correlation | `results/task3/<ds>/signals.json` |
| `t3_adaptive` | dev-selected stopping rules on the eval split | `results/task3/<ds>/adaptive.json` |
| `t3_live` | single answer vs. parallel sampling vs. revision | `results/task3/live/live_summary.json` |
| `t2_pairs` | first complementary pair of each question type: images, references, answers | `results/task2/vlm196/final_eval_stage3_examples.jsonl` |
| `examples/t2_pairs.tex` | the same pairs under all four image conditions | `results/task2/vlm196/final_eval_stage3_examples.jsonl` |
| `examples/t3_examples.md` | selection disagreements, revisions, premature stops (selection rule stated with each) | `results/task3/<ds>/`, `results/task3/live/` |

Sanity checks are in `results/sanity/`: response-only loss masking, resume equivalence, merged-vs-unmerged logits and
OPLoRA constraints (Task 1); chat-template equality, frozen base weights and connector/LoRA gradients (Task 2).

## 5. External code and data

- `configs/task1/first_plot_questions.yaml`: the 24 qualitative prompts from the Emergent Misalignment evaluation suite (github.com/emergent-misalignment/emergent-misalignment, `evaluation/first_plot_questions.yaml`, commit `80c1196`).
- Models and datasets are loaded from the Hugging Face Hub: `Qwen/Qwen3-0.6B`, `meta-math/MetaMathQA`, `openai/gsm8k`, `Rowan/hellaswag`; BoolQ and ARC-Easy through the lm-evaluation-harness task definitions.
- Benchmarks use EleutherAI lm-evaluation-harness 0.4.13 with stock task definitions (`gsm8k_cot`, `hellaswag`, `boolq`, `arc_easy`). The HellaSwag context preprocessing in `task1_ft/diagnostics.py` reproduces the harness's `hellaswag/utils.py` so probe prompts match the benchmark format.
- Task 2 answer normalization (`task2_vlm/vqa_norm.py`) ports `processPunctuation`, `processDigitArticle` and the contraction, number and article lists from the official VQA evaluation code (github.com/GT-Vision-Lab/VQA, `PythonEvaluationTools/vqaEvaluation/vqaEval.py`, commit `a013f00`); it is applied to every answer, and the soft accuracy follows the leave-one-out formula in the manual. Data: COCO 2014 (captions, instances), VQAv2 (questions, annotations, complementary pairs); models `google/siglip-base-patch16-224` and `Qwen/Qwen3-0.6B`.
- Task 3 MATH equivalence uses `math-verify` 0.9.0. MathVista normalization in `task3_ttc/answers.py` ports `normalize_extracted_answer` and `get_most_similar` from the official MathVista evaluation code (github.com/lupantech/MathVista, `evaluation/calculate_score.py`); question metadata is the official `data/testmini.json` at commit `c5acd37`, downloaded and checksum-verified by `task3_ttc/prepare.py`. Live generation uses `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- The LoRA and OPLoRA layers in `task1_ft/lora.py` are written for this assignment. The A-matrix initialization follows the uniform bound used by Hu et al. (LoRA) and PEFT; the projection follows Xiong and Xie (OPLoRA), Sections 3.2–3.3.

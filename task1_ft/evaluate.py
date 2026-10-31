from __future__ import annotations

import argparse

import torch

from common.utils import Timer, env_info, gpu_peak, load_config, repo_path, save_json, write_jsonl
from task1_ft.conditions import model_path


def _first(x):
    while isinstance(x, (list, tuple)) and x:
        x = x[0]
    return x


def gsm8k_rows(samples, tok, trunc: int) -> list[dict]:
    by_doc = {}
    for s in samples:
        r = by_doc.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "question": s["doc"]["question"], "target": s["target"]})
        resp = _first(s["resps"])
        r["response"] = resp
        key = "flexible" if "flexible" in s["filter"] else "strict"
        r[f"extracted_{key}"] = _first(s["filtered_resps"])
        r[f"em_{key}"] = float(s["exact_match"])
    rows = []
    for r in by_doc.values():
        n = len(tok(r["response"], add_special_tokens=False)["input_ids"])
        r["n_tokens"] = n
        r["truncated"] = n >= trunc - 2
        r["extractable_flexible"] = r.get("extracted_flexible") not in (None, "", "[invalid]")
        r["extractable_strict"] = r.get("extracted_strict") not in (None, "", "[invalid]")
        rows.append(r)
    return sorted(rows, key=lambda r: r["doc_id"])


def choice_rows(samples) -> list[dict]:
    rows = []
    for s in samples:
        lls = [float(_first(x)) for x in s["filtered_resps"]]
        row = {"doc_id": s["doc_id"], "gold": s["target"], "pred": int(max(range(len(lls)), key=lambda i: lls[i])),
               "acc": float(s["acc"]), "loglik": lls}
        if "acc_norm" in s:
            row["acc_norm"] = float(s["acc_norm"])
        rows.append(row)
    return sorted(rows, key=lambda r: r["doc_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1/eval.yaml")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    cfg = load_config(args.config)
    ecfg = cfg["eval"]
    for name in args.conditions.split(","):
        out = repo_path(ecfg["results"]) / f"{name}{args.tag}"
        if (out / "scores.json").exists():
            print(f"{name}: evaluated, skipping")
            continue
        path = model_path(cfg, name)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t = Timer()
        lm = HFLM(pretrained=path, dtype="float16", batch_size=int(ecfg["gsm8k"]["batch_size"]), enable_thinking=False)
        lm.model.eval()
        tok = lm.tokenizer
        g = ecfg["gsm8k"]
        res_g = lm_eval.simple_evaluate(
            model=lm, tasks=[g["task"]], num_fewshot=0, apply_chat_template=True,
            gen_kwargs={"max_gen_toks": int(g["max_gen_toks"]), "do_sample": False},
            log_samples=True, limit=args.limit, random_seed=6304, numpy_random_seed=6304, torch_random_seed=6304,
        )
        gsm_s = t()
        lm.batch_size_per_gpu = int(ecfg["loglik"]["batch_size"])
        t2 = Timer()
        res_l = lm_eval.simple_evaluate(
            model=lm, tasks=list(ecfg["loglik"]["tasks"]), num_fewshot=0, apply_chat_template=False,
            log_samples=True, limit=args.limit, random_seed=6304, numpy_random_seed=6304, torch_random_seed=6304,
        )
        ll_s = t2()

        rows = gsm8k_rows(res_g["samples"][g["task"]], tok, int(g["truncation_tokens"]))
        write_jsonl(out / "gsm8k_samples.jsonl", rows)
        gm = res_g["results"][g["task"]]
        scores = {
            "condition": name, "model_path": path, "limit": args.limit,
            "gsm8k": {
                "n": len(rows),
                "flexible_em": gm.get("exact_match,flexible-extract"),
                "flexible_em_stderr": gm.get("exact_match_stderr,flexible-extract"),
                "strict_em": gm.get("exact_match,strict-match"),
                "strict_em_stderr": gm.get("exact_match_stderr,strict-match"),
                "extractable_flexible": sum(r["extractable_flexible"] for r in rows) / len(rows),
                "extractable_strict": sum(r["extractable_strict"] for r in rows) / len(rows),
                "truncated": sum(r["truncated"] for r in rows) / len(rows),
                "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows),
            },
        }
        for task in ecfg["loglik"]["tasks"]:
            m = res_l["results"][task]
            crows = choice_rows(res_l["samples"][task])
            write_jsonl(out / f"{task}_samples.jsonl", crows)
            scores[task] = {"n": len(crows), "acc": m.get("acc,none"), "acc_stderr": m.get("acc_stderr,none"),
                            "acc_norm": m.get("acc_norm,none"), "acc_norm_stderr": m.get("acc_norm_stderr,none")}
        scores["seconds"] = {"gsm8k": gsm_s, "loglik": ll_s}
        scores.update(gpu_peak())
        scores["env"] = env_info()
        scores["lm_eval"] = lm_eval.__version__ if hasattr(lm_eval, "__version__") else None
        scores["task_versions"] = {**res_g.get("versions", {}), **res_l.get("versions", {})}
        scores["gsm8k_config"] = res_g.get("configs", {}).get(g["task"])
        scores["gsm8k_example_prompt"] = res_g["samples"][g["task"]][0]["arguments"][0][0]
        scores["loglik_example_prompts"] = {tk: res_l["samples"][tk][0]["arguments"][0][0] for tk in ecfg["loglik"]["tasks"]}
        save_json(out / "scores.json", scores)
        print({k: scores[k] for k in ["condition", "gsm8k"]}, {tk: scores[tk] for tk in ecfg["loglik"]["tasks"]})
        del lm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

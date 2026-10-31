from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter

import torch

from common.utils import load_config, repo_path, save_json, set_seed

EOT = "<|im_end|>"


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def chat_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_pair(tokenizer, question: str, response: str) -> tuple[list[int], list[int]]:
    prompt = tokenizer(chat_prompt(tokenizer, question), add_special_tokens=False)["input_ids"]
    target = tokenizer(response, add_special_tokens=False)["input_ids"]
    return prompt, target + [tokenizer.convert_tokens_to_ids(EOT)]


def build(cfg: dict) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    set_seed(int(cfg["seed"]))
    dcfg = cfg["data"]
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    gsm = load_dataset(dcfg["gsm8k"], "main", split="test")
    gsm_norm = {normalize(q) for q in gsm["question"]}
    src = load_dataset(dcfg["source"], split="train")

    n_target, max_len = int(dcfg["n_train"]), int(dcfg["max_length"])
    stats, types = Counter(), Counter()
    prompts, targets, ids, excluded = [], [], [], {"empty": [], "gsm8k_test": [], "too_long": []}
    scanned = 0
    for i, row in enumerate(src):
        scanned = i + 1
        q, r = row.get("query") or "", row.get("response") or ""
        oq = row.get("original_question") or ""
        if not q.strip() or not r.strip():
            excluded["empty"].append(i)
            continue
        if normalize(q) in gsm_norm or (oq.strip() and normalize(oq) in gsm_norm):
            excluded["gsm8k_test"].append(i)
            continue
        p, t = encode_pair(tok, q, r)
        if len(p) + len(t) > max_len:
            excluded["too_long"].append(i)
            continue
        prompts.append(p)
        targets.append(t)
        ids.append(i)
        types[row.get("type", "unknown")] += 1
        if len(ids) == n_target:
            break

    if len(ids) != n_target:
        raise RuntimeError(f"only {len(ids)} eligible examples")

    lengths = torch.tensor([len(p) + len(t) for p, t in zip(prompts, targets)])
    resp = torch.tensor([len(t) for t in targets])
    out = repo_path(dcfg["cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt": prompts, "target": targets, "source_index": ids, "model": cfg["model"]}, out)

    res = repo_path(dcfg["results"])
    save_json(res / "train_example_ids.json", {"source": dcfg["source"], "split": "train", "source_index": ids})
    save_json(res / "data_summary.json", {
        "rows_scanned": scanned,
        "kept": len(ids),
        "excluded_counts": {k: len(v) for k, v in excluded.items()},
        "excluded_index": excluded,
        "gsm8k_test_questions": len(gsm_norm),
        "type_counts": dict(types),
        "tokens_total_mean": lengths.float().mean().item(),
        "tokens_total_median": lengths.median().item(),
        "tokens_total_max": lengths.max().item(),
        "response_tokens_mean": resp.float().mean().item(),
        "response_tokens_sum": int(resp.sum()),
        "chat_template_enable_thinking": False,
    })

    p0, t0 = prompts[0], targets[0]
    save_json("results/sanity/task1_masking.json", {
        "source_index": ids[0],
        "masked_prompt_text": tok.decode(p0),
        "loss_text": tok.decode(t0),
        "last_loss_token": tok.convert_ids_to_tokens(t0[-1]),
        "n_masked": len(p0),
        "n_loss": len(t0),
    })
    print(f"kept {len(ids)} of {scanned} scanned; excluded {dict((k, len(v)) for k, v in excluded.items())}")


def load_cached(cfg: dict) -> dict:
    return torch.load(repo_path(cfg["data"]["cache"]), weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if repo_path(cfg["data"]["cache"]).exists():
        print("cached data exists, skipping")
        return
    build(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import Timer, load_config, repo_path, write_jsonl
from task1_ft.conditions import model_path


def prompts(path) -> list[dict]:
    rows = []
    for q in yaml.safe_load(repo_path(path).read_text(encoding="utf-8")):
        fmt = "json" if q["id"].endswith("_json") else "template" if q["id"].endswith("_template") else "plain"
        base = q["id"].removesuffix("_json").removesuffix("_template")
        msgs = ([{"role": "system", "content": q["system"]}] if q.get("system") else []) + [
            {"role": "user", "content": q["paraphrases"][0]}]
        rows.append({"id": q["id"], "question": base, "format": fmt, "messages": msgs})
    return rows


@torch.no_grad()
def generate(model, tok, rows, max_new: int, bs: int) -> list[dict]:
    tok.padding_side = "left"
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                             eos_token_id=eos, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for r, ids in zip(chunk, new):
            ids = ids.tolist()
            ended = any(t in eos for t in ids)
            n = next((j for j, t in enumerate(ids) if t in eos), len(ids))
            out.append({**{k: r[k] for k in ("id", "question", "format")}, "prompt": texts[chunk.index(r)],
                        "response": tok.decode(ids[:n], skip_special_tokens=True), "n_tokens": n,
                        "truncated": not ended})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1/eval.yaml")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    qcfg = cfg["eval"]["qualitative"]
    rows = prompts(qcfg["file"])
    for name in args.conditions.split(","):
        out = repo_path(qcfg["results"]) / f"{name}{args.tag}.jsonl"
        if out.exists():
            print(f"{name}: qualitative exists, skipping")
            continue
        path = model_path(cfg, name)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        t = Timer()
        res = generate(model, tok, rows, int(qcfg["max_new_tokens"]), int(qcfg["batch_size"]))
        for r in res:
            r["condition"] = name
        write_jsonl(out, res)
        print(f"{name}: {len(res)} responses in {t():.0f}s, truncated {sum(r['truncated'] for r in res)}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

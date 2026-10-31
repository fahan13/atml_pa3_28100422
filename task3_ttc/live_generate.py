from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import Timer, env_info, load_config, load_json, read_jsonl, repo_path, append_jsonl, save_json
from task3_ttc.boxed import last_boxed


def call_seed(base: int, qi: int, call: int) -> int:
    return base + 100 * qi + call


@torch.no_grad()
def generate(model, tok, messages, max_new: int, seed: int, sampling: dict) -> dict:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    t = Timer()
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=sampling["temperature"],
                         top_p=sampling["top_p"], top_k=sampling["top_k"], repetition_penalty=sampling["repetition_penalty"],
                         eos_token_id=eos, pad_token_id=tok.convert_tokens_to_ids("<|endoftext|>"))
    torch.cuda.synchronize()
    sec = t()
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    ended = any(x in eos for x in new)
    n = next((j for j, x in enumerate(new) if x in eos), len(new))
    resp = tok.decode(new[:n], skip_special_tokens=True)
    return {"input_tokens": int(enc["input_ids"].shape[1]), "output_tokens": n, "hit_max": not ended,
            "seconds": sec, "response": resp, "extracted": last_boxed(resp)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3/live.yaml")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    qs = load_json(cfg["questions"])
    if args.limit:
        qs = qs[: args.limit]
    out = repo_path(cfg["results"]) / f"calls_shard{args.shard}.jsonl"
    done = {(r["qi"], r["call"]) for r in read_jsonl(out)}
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=torch.float16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    lim, smp, base = cfg["limits"], cfg["sampling"], int(cfg["seed"])
    for qi, q in enumerate(qs):
        if qi % args.n_shards != args.shard:
            continue
        user = q["question_text"] + "\n\n" + cfg["instruction"]
        msgs = [{"role": "system", "content": cfg["system"]}, {"role": "user", "content": user}]
        plan = [(0, "single", 0, msgs, int(lim["single"]))]
        plan += [(c, "parallel", c - 1, msgs, int(lim["parallel"])) for c in range(1, int(lim["n_parallel"]) + 1)]
        for call, strategy, idx, m, max_new in plan:
            if (qi, call) in done:
                continue
            r = generate(model, tok, m, max_new, call_seed(base, qi, call), smp)
            append_jsonl(out, {"qi": qi, "sample_id": q["sample_id"], "call": call, "strategy": strategy, "index": idx,
                               "seed": call_seed(base, qi, call), "max_new_tokens": max_new, **r})
        rev_call = int(lim["n_parallel"]) + 1
        if (qi, rev_call) not in done:
            initial = next(r for r in read_jsonl(out) if r["qi"] == qi and r["call"] == 1)
            m = msgs + [{"role": "assistant", "content": initial["response"]}, {"role": "user", "content": cfg["review"]}]
            r = generate(model, tok, m, int(lim["revision"]), call_seed(base, qi, rev_call), smp)
            append_jsonl(out, {"qi": qi, "sample_id": q["sample_id"], "call": rev_call, "strategy": "revision", "index": 0,
                               "seed": call_seed(base, qi, rev_call), "max_new_tokens": int(lim["revision"]), **r})
        print(f"shard {args.shard}: question {qi} done", flush=True)
    save_json(repo_path(cfg["results"]) / f"env_shard{args.shard}.json",
              {**env_info(), "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9})


if __name__ == "__main__":
    main()

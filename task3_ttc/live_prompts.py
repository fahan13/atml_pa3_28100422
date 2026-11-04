from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from common.utils import load_config, load_json, read_jsonl, repo_path, save_json, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3/live.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    qs = load_json(cfg["questions"])
    res = repo_path(cfg["results"])
    calls = {(r["qi"], r["call"]): r for f in sorted(res.glob("calls_shard*.jsonl")) for r in read_jsonl(f)}
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    rev_call = int(cfg["limits"]["n_parallel"]) + 1
    rows, mismatches = [], []
    for qi, q in enumerate(qs):
        msgs = [{"role": "system", "content": cfg["system"]},
                {"role": "user", "content": q["question_text"] + "\n\n" + cfg["instruction"]}]
        for call in range(rev_call + 1):
            r = calls.get((qi, call))
            if r is None:
                continue
            m = msgs if call < rev_call else msgs + [{"role": "assistant", "content": calls[(qi, 1)]["response"]},
                                                     {"role": "user", "content": cfg["review"]}]
            text = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            rows.append({"qi": qi, "sample_id": q["sample_id"], "call": call, "strategy": r["strategy"], "seed": r["seed"],
                         "max_new_tokens": r["max_new_tokens"], "messages": m, "prompt": text, "input_tokens": n})
            if n != r["input_tokens"]:
                mismatches.append({"qi": qi, "call": call, "rebuilt": n, "recorded": r["input_tokens"]})
    write_jsonl(res / "prompts.jsonl", rows)
    save_json(res / "prompts_check.json", {"n_prompts": len(rows), "n_calls": len(calls),
                                           "input_token_mismatches": mismatches,
                                           "rule": "prompts rebuilt with the generation chat template; token counts compared with calls_shard*.jsonl"})
    print(f"{len(rows)} prompts rebuilt, {len(mismatches)} token-count mismatches")


if __name__ == "__main__":
    main()

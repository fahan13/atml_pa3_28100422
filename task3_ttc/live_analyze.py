from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.utils import load_config, load_json, read_jsonl, repo_path, save_json
from task3_ttc.answers import NO_ANSWER, group_math, is_missing, math_key, math_verify_pair
from task3_ttc.select import vote


def graded(ans, gold: str) -> bool:
    return ans is not None and math_verify_pair(gold, ans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3/live.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = repo_path(cfg["results"])
    qs = load_json(cfg["questions"])
    calls = pd.DataFrame([r for f in sorted(res.glob("calls_shard*.jsonl")) for r in read_jsonl(f)])
    n_par = int(cfg["limits"]["n_parallel"])
    rows = []
    for qi, q in enumerate(qs):
        c = calls[calls["qi"] == qi].set_index("call")
        if len(c) < n_par + 2:
            continue
        gold = str(q["gold_answer"])
        single = c.loc[0]
        par = [c.loc[i] for i in range(1, n_par + 1)]
        rev = c.loc[n_par + 1]
        keys = [math_key(p["extracted"]) for p in par]
        grp = group_math(keys, gold)
        pos, win, _ = vote(np.array(grp["group"]), np.ones(n_par))
        par_ans = keys[pos]
        init_ans, rev_ans = keys[0], math_key(rev["extracted"])
        init_ok, rev_ok = graded(init_ans, gold), graded(rev_ans, gold)
        rows.append({
            "qi": qi, "sample_id": q["sample_id"], "gold": gold,
            "single_answer": math_key(single["extracted"]), "single_correct": graded(math_key(single["extracted"]), gold),
            "single_in": int(single["input_tokens"]), "single_out": int(single["output_tokens"]),
            "single_s": float(single["seconds"]), "single_trunc": bool(single["hit_max"]),
            "single_extracted": not is_missing(single["extracted"]),
            "parallel_answer": par_ans, "parallel_correct": graded(par_ans, gold), "parallel_winner_no_answer": win == NO_ANSWER,
            "parallel_in": int(sum(p["input_tokens"] for p in par)), "parallel_out": int(sum(p["output_tokens"] for p in par)),
            "parallel_s": float(sum(p["seconds"] for p in par)), "parallel_trunc_calls": int(sum(p["hit_max"] for p in par)),
            "parallel_extracted_calls": int(sum(not is_missing(p["extracted"]) for p in par)),
            "parallel_distinct_groups": len(set(g for g in grp["group"] if g != NO_ANSWER)),
            "parallel_oracle": any(grp["correct"]),
            "initial_answer": init_ans, "initial_correct": init_ok,
            "refine_answer": rev_ans, "refine_correct": rev_ok,
            "refine_in": int(par[0]["input_tokens"] + rev["input_tokens"]), "refine_out": int(par[0]["output_tokens"] + rev["output_tokens"]),
            "refine_s": float(par[0]["seconds"] + rev["seconds"]), "revision_in": int(rev["input_tokens"]),
            "revision_out": int(rev["output_tokens"]), "revision_trunc": bool(rev["hit_max"]),
            "revision_extracted": not is_missing(rev["extracted"]), "initial_trunc": bool(par[0]["hit_max"]),
            "answer_changed": init_ans != rev_ans,
        })
    t = pd.DataFrame(rows)
    t.to_csv(res / "live_per_question.csv", index=False)
    n = len(t)
    summ = {"n_questions": n, "allowance_tokens": 3072, "strategies": {}}
    for s in ("single", "parallel", "refine"):
        summ["strategies"][s] = {"accuracy": float(t[f"{s}_correct"].mean()), "correct": int(t[f"{s}_correct"].sum()),
                                 "mean_input_tokens": float(t[f"{s}_in"].mean()), "mean_output_tokens": float(t[f"{s}_out"].mean()),
                                 "mean_seconds": float(t[f"{s}_s"].mean()), "total_seconds": float(t[f"{s}_s"].sum())}
    summ["strategies"]["single"].update({"truncation_rate": float(t["single_trunc"].mean()), "extraction_rate": float(t["single_extracted"].mean())})
    summ["strategies"]["parallel"].update({"truncation_rate_per_call": float(t["parallel_trunc_calls"].sum() / (3 * n)),
                                           "extraction_rate_per_call": float(t["parallel_extracted_calls"].sum() / (3 * n)),
                                           "oracle_any_of_3": float(t["parallel_oracle"].mean()),
                                           "mean_distinct_groups": float(t["parallel_distinct_groups"].mean()),
                                           "winner_no_answer_rate": float(t["parallel_winner_no_answer"].mean())})
    summ["strategies"]["refine"].update({"initial_accuracy": float(t["initial_correct"].mean()),
                                         "initial_truncation_rate": float(t["initial_trunc"].mean()),
                                         "revision_truncation_rate": float(t["revision_trunc"].mean()),
                                         "revision_extraction_rate": float(t["revision_extracted"].mean()),
                                         "mean_revision_input_tokens": float(t["revision_in"].mean()),
                                         "answer_changed_rate": float(t["answer_changed"].mean())})
    summ["transitions_initial_to_revised"] = {
        "wrong_to_correct": int((~t["initial_correct"] & t["refine_correct"]).sum()),
        "correct_to_wrong": int((t["initial_correct"] & ~t["refine_correct"]).sum()),
        "correct_to_correct": int((t["initial_correct"] & t["refine_correct"]).sum()),
        "wrong_to_wrong": int((~t["initial_correct"] & ~t["refine_correct"]).sum()),
    }
    rng = np.random.default_rng(int(cfg["seed"]))
    b = rng.integers(0, n, (1000, n))
    summ["paired_bootstrap"] = {}
    for a_, b_ in (("parallel", "single"), ("refine", "single"), ("refine", "parallel")):
        d = (t[f"{a_}_correct"].astype(float) - t[f"{b_}_correct"].astype(float)).to_numpy()
        bs = d[b].mean(axis=1)
        summ["paired_bootstrap"][f"{a_}-{b_}"] = {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    env = [load_json(f) for f in sorted(res.glob("env_shard*.json"))]
    summ["env"] = env
    save_json(res / "live_summary.json", summ)
    print(summ["strategies"], summ["transitions_initial_to_revised"])


if __name__ == "__main__":
    main()

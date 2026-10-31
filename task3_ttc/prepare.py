from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import urllib.request

import numpy as np
import pandas as pd

from common.utils import Timer, ensure_dir, load_config, load_json, repo_path, save_json
from task3_ttc.answers import NO_ANSWER, group_math, group_mathvista, is_missing, math_key

KEYS = ["dataset", "sample_id", "rollout_id"]


def fetch_mathvista_meta(cfg: dict) -> dict:
    m = cfg["mathvista_meta"]
    path = repo_path(m["path"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(m["url"], path)
    sha = hashlib.sha1(path.read_bytes()).hexdigest()
    if sha != m["sha1"]:
        raise RuntimeError(f"MathVista metadata sha1 {sha} != {m['sha1']}")
    return load_json(path)


def load_joined(cfg: dict, ds: str) -> tuple[pd.DataFrame, dict]:
    d = cfg["datasets"][ds]
    root = repo_path(cfg["data_root"]) / d["folder"]
    gen = pd.read_parquet(root / "generations.parquet")
    prob = pd.read_parquet(root / "probability_metrics.parquet")
    df = gen.merge(prob, on=KEYS, how="inner", validate="one_to_one")
    if d["kind"] == "vlm":
        att = pd.read_parquet(root / "attention_metrics.parquet").rename(columns={"n_generated_text_tokens": "att_prev_counts"})
        df = df.merge(att, on=KEYS, how="inner", validate="one_to_one")
    df = df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)
    T = df["n_generated_text_tokens"].to_numpy()
    check = {
        "rows": {"generations": len(gen), "probability": len(prob), "joined": len(df)},
        "questions": int(df["sample_id"].nunique()),
        "rollouts_per_question_all_32": bool((df.groupby("sample_id")["rollout_id"].apply(lambda s: sorted(s) == list(range(32)))).all()),
        "token_ids_eq_length": int((df["generated_token_ids"].map(len).to_numpy() == T).sum()),
        "logp_eq_length": int((df["logp"].map(len).to_numpy() == T).sum()),
        "kl_eq_length": int((df["kl_u_p"].map(len).to_numpy() == T).sum()),
        "zero_length_rollouts": int((T == 0).sum()),
        "hit_max_new_tokens": int(df["hit_max_new_tokens"].sum()),
        "null_extracted": int(df["extracted_answer"].map(is_missing).sum()),
        "retained_tokens": int(T.sum()),
    }
    if d["kind"] == "vlm":
        for c in ["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"]:
            check[f"{c}_eq_length"] = int((df[c].map(len).to_numpy() == T).sum())
        check["prev_counts_are_range"] = int(sum(np.array_equal(np.asarray(a), np.arange(n)) for a, n in zip(df["att_prev_counts"], T)))
    check["all_aligned"] = all(v == len(df) for k, v in check.items() if k.endswith("_eq_length") or k == "prev_counts_are_range")
    return df, check


def _math_job(args):
    keys, gold = args
    return group_math(keys, gold)


def group_answers(cfg: dict, ds: str, df: pd.DataFrame, meta: dict | None, timeout: float) -> tuple[pd.DataFrame, dict]:
    qids = df["sample_id"].drop_duplicates().tolist()
    by_q = {q: g for q, g in df.groupby("sample_id", sort=False)}
    results, timeouts = {}, []
    if ds == "math":
        jobs = {q: ([math_key(x) for x in by_q[q]["extracted_answer"]], str(by_q[q]["gold_answer"].iloc[0])) for q in qids}
        with mp.Pool(int(cfg.get("workers", 4))) as pool:
            pending = {q: pool.apply_async(_math_job, (jobs[q],)) for q in qids}
            for q, h in pending.items():
                try:
                    results[q] = h.get(timeout=timeout)
                except mp.TimeoutError:
                    timeouts.append(q)
            pool.terminate()
        for q in timeouts:
            keys, gold = jobs[q]
            distinct = sorted({k for k in keys if k is not None}, key=keys.index)
            results[q] = {"group": [distinct.index(k) if k is not None else NO_ANSWER for k in keys],
                          "correct": [k is not None and k == gold.strip() for k in keys],
                          "representatives": distinct, "mixed_groups": 0, "fallback": "string_equality"}
    else:
        for q in qids:
            g = by_q[q]
            results[q] = group_mathvista(g["extracted_answer"].tolist(), str(g["gold_answer"].iloc[0]), meta[str(q)])

    rows = []
    for q in qids:
        g, r = by_q[q], results[q]
        for i, (rid, ext) in enumerate(zip(g["rollout_id"], g["extracted_answer"])):
            rows.append({"sample_id": q, "rollout_id": int(rid), "extracted": None if is_missing(ext) else ext,
                         "key": r.get("key", [math_key(x) for x in g["extracted_answer"]])[i],
                         "status": r["status"][i] if "status" in r else ("missing" if is_missing(ext) else "ok"),
                         "group": int(r["group"][i]), "correct": bool(r["correct"][i])})
    ans = pd.DataFrame(rows)
    qsum = {"timeouts_string_fallback": timeouts,
            "mixed_correctness_groups": int(sum(results[q]["mixed_groups"] for q in qids)),
            "status_counts": ans["status"].value_counts().to_dict(),
            "no_answer_rollouts": int((ans["group"] == NO_ANSWER).sum())}
    return ans, qsum


def summaries(df: pd.DataFrame, vlm: bool) -> tuple[pd.DataFrame, dict]:
    T = df["n_generated_text_tokens"].to_numpy()
    out = pd.DataFrame({
        "sample_id": df["sample_id"], "rollout_id": df["rollout_id"].astype(int), "T": T,
        "hit_max": df["hit_max_new_tokens"].astype(bool),
        "L": [float(np.mean(x)) if len(x) else np.nan for x in df["logp"]],
        "K": [float(np.mean(x)) if len(x) else np.nan for x in df["kl_u_p"]],
    })
    tokens = {"offsets": np.concatenate([[0], np.cumsum(T)]).astype(np.int64)}
    if vlm:
        img = [np.asarray(x, dtype=np.float64) for x in df["img_attn_total"]]
        pr = [np.asarray(x, dtype=np.float64) for x in df["prompt_text_attn_total"]]
        ge = [np.asarray(x, dtype=np.float64) for x in df["generated_text_attn_total"]]
        v = []
        zero_den = 0
        for a, b, c in zip(img, pr, ge):
            den = a + b + c
            zero_den += int((den <= 0).sum())
            v.append(np.where(den > 0, a / np.where(den > 0, den, 1.0), 0.0))
        out["V"] = [float(x.mean()) if len(x) else np.nan for x in v]
        out["img_mass_raw"] = [float(x.mean()) if len(x) else np.nan for x in img]
        tokens["v"] = np.concatenate(v).astype(np.float32)
        tokens["kl"] = np.concatenate([np.asarray(x, dtype=np.float32) for x in df["kl_u_p"]])
        tokens["zero_denominator_tokens"] = np.array(zero_den)
    return out, tokens


def splits(cfg: dict, qids: list[str]) -> dict:
    ids = sorted(qids)
    perm = np.random.default_rng(int(cfg["seed"])).permutation(len(ids))
    order = [ids[i] for i in perm]
    n_dev = int(cfg["n_dev"])
    return {"dev": order[:n_dev], "eval": order[n_dev:]}


def orders(cfg: dict, qids: list[str]) -> np.ndarray:
    rng = np.random.default_rng(int(cfg["order_seed"]))
    return np.stack([np.stack([rng.permutation(32) for _ in range(int(cfg["n_orders"]))]) for _ in sorted(qids)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    ap.add_argument("--datasets", default="math,mathvista")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta = fetch_mathvista_meta(cfg)
    split_path = repo_path(cfg["results"]) / "splits.json"
    all_splits = load_json(split_path) if split_path.exists() else {}
    for ds in args.datasets.split(","):
        t = Timer()
        res = ensure_dir(repo_path(cfg["results"]) / ds)
        cache = ensure_dir(repo_path(cfg["cache"]) / ds)
        df, check = load_joined(cfg, ds)
        vlm = cfg["datasets"][ds]["kind"] == "vlm"
        ans, qsum = group_answers(cfg, ds, df, meta if vlm else None, args.timeout)
        summ, tokens = summaries(df, vlm)
        roll = summ.merge(ans, on=["sample_id", "rollout_id"], validate="one_to_one")
        roll.to_csv(res / "rollouts.csv", index=False)
        gold = df.drop_duplicates("sample_id").set_index("sample_id")["gold_answer"]
        qtab = roll.groupby("sample_id").agg(n_correct=("correct", "sum"), n_no_answer=("group", lambda s: int((s == NO_ANSWER).sum())),
                                             n_groups=("group", lambda s: int(s[s != NO_ANSWER].nunique())), mean_T=("T", "mean"))
        qtab["gold"] = gold
        qtab.to_csv(res / "questions.csv")
        np.savez(cache / "tokens.npz", **tokens)
        qids = df["sample_id"].drop_duplicates().tolist()
        all_splits[ds] = splits(cfg, qids)
        np.save(res / "orders.npy", orders(cfg, qids))
        save_json(res / "orders_index.json", sorted(qids))
        if ds == "math":
            all_splits["live_math"] = all_splits["math"]["eval"][: int(cfg["n_live"])]
            q = df.drop_duplicates("sample_id").set_index("sample_id")
            save_json(repo_path(cfg["results"]) / "live_questions.json",
                      [{"sample_id": s, "question_text": q.loc[s, "question_text"], "gold_answer": q.loc[s, "gold_answer"]}
                       for s in all_splits["live_math"]])
        save_json(res / "prepare_summary.json", {
            **check, **qsum, "seconds": t(),
            "pass_at_32": float((qtab["n_correct"] > 0).mean()),
            "rollout_accuracy": float(roll["correct"].mean()),
            "zero_denominator_tokens": int(tokens.get("zero_denominator_tokens", 0)),
        })
        print(ds, check, qsum["status_counts"], f"{t():.0f}s")
    save_json(split_path, all_splits)


if __name__ == "__main__":
    main()

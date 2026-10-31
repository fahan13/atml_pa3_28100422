from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.utils import Timer, load_config, load_json, repo_path, save_json
from task3_ttc.answers import NO_ANSWER
from task3_ttc.select import Pool, earliest_max, earliest_min, rwc, visconf, vote

COMMON = ["first", "sc", "logp", "logp_rwc", "selfcert", "selfcert_rwc", "shortest"]
VLM = ["vision", "vision_rwc", "visconf", "visconf_rwc", "visconf_token", "visconf_token_rwc"]
REQUIRED = {"math": ["first", "sc", "logp", "selfcert", "selfcert_rwc"],
            "mathvista": ["first", "sc", "logp", "selfcert", "selfcert_rwc", "vision", "vision_rwc", "visconf", "visconf_rwc"]}


def select_all(a: dict, idx: np.ndarray, vlm: bool) -> dict:
    n = len(idx)
    groups = a["group"][idx]
    if n == 1:
        sel = {m: 0 for m in COMMON + (VLM if vlm else [])}
        return sel
    sel = {"first": 0, "sc": vote(groups, np.ones(n))[0],
           "logp": earliest_max(a["L"][idx]), "logp_rwc": rwc(a["L"][idx], groups)[0],
           "selfcert": earliest_max(a["K"][idx]), "selfcert_rwc": rwc(a["K"][idx], groups)[0],
           "shortest": earliest_min(a["T"][idx])}
    if vlm:
        V = a["V"][idx]
        C, Ztok, _ = visconf([a["kl"][i] for i in idx], [a["v"][i] for i in idx])
        Z = V * C
        sel.update({"vision": earliest_max(V), "vision_rwc": rwc(V, groups)[0],
                    "visconf": earliest_max(Z), "visconf_rwc": rwc(Z, groups)[0],
                    "visconf_token": earliest_max(Ztok), "visconf_token_rwc": rwc(Ztok, groups)[0]})
    return sel


def run(cfg: dict, ds: str) -> None:
    t = Timer()
    pool = Pool(cfg, ds)
    budgets = [int(b) for b in cfg["budgets"]]
    n_orders = int(cfg["n_orders"])
    methods = COMMON + (VLM if pool.vlm else [])
    splits = load_json(repo_path(cfg["results"]) / "splits.json")[ds]
    split_of = {q: s for s in ("dev", "eval") for q in splits[s]}
    acc_rows, cov_rows, sel_rows = [], [], []
    for q in pool.qids:
        a = pool.arr[q]
        acc = {(N, m): [] for N in budgets for m in methods}
        cost = {(N, m): [] for N in budgets for m in methods}
        cov = {N: {"oracle": [], "distinct": [], "distinct_valid": [], "no_answer": []} for N in budgets}
        for o in range(n_orders):
            order = pool.order(q, o)
            for N in budgets:
                idx = order[:N]
                sel = select_all(a, idx, pool.vlm)
                full = a["T"][idx].sum()
                for m in methods:
                    acc[(N, m)].append(float(a["correct"][idx[sel[m]]]))
                    cost[(N, m)].append(float(a["T"][idx[0]]) if m == "first" else float(full))
                g = a["group"][idx]
                cov[N]["oracle"].append(float(a["correct"][idx].any()))
                cov[N]["distinct"].append(len(set(g.tolist())))
                cov[N]["distinct_valid"].append(len(set(g[g != NO_ANSWER].tolist())))
                cov[N]["no_answer"].append(float((g == NO_ANSWER).mean()))
                if N == 32:
                    sel_rows.append({"sample_id": q, "order": o, **{m: int(idx[sel[m]]) for m in methods}})
        for N in budgets:
            for m in methods:
                acc_rows.append({"sample_id": q, "split": split_of[q], "N": N, "method": m,
                                 "acc": float(np.mean(acc[(N, m)])), "cost": float(np.mean(cost[(N, m)]))})
            cov_rows.append({"sample_id": q, "split": split_of[q], "N": N, **{k: float(np.mean(v)) for k, v in cov[N].items()}})
    res = repo_path(cfg["results"]) / ds
    per_q = pd.DataFrame(acc_rows)
    cov_q = pd.DataFrame(cov_rows)
    per_q.to_csv(res / "offline_per_question.csv", index=False)
    cov_q.to_csv(res / "offline_coverage_per_question.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(res / "offline_selected_N32.csv", index=False)

    rng = np.random.default_rng(int(cfg["seed"]))
    summary = {"dataset": ds, "methods": methods, "required_methods": REQUIRED[ds], "budgets": budgets, "splits": {}}
    for split in ("dev", "eval", "all"):
        pq = per_q if split == "all" else per_q[per_q["split"] == split]
        cq = cov_q if split == "all" else cov_q[cov_q["split"] == split]
        block = {"n_questions": int(pq["sample_id"].nunique()), "by_N": {}}
        for N in budgets:
            c = cq[cq["N"] == N]
            entry = {"oracle": float(c["oracle"].mean()), "distinct_groups": float(c["distinct"].mean()),
                     "distinct_valid_groups": float(c["distinct_valid"].mean()), "no_answer_frac": float(c["no_answer"].mean()),
                     "methods": {}}
            for m in methods:
                x = pq[(pq["N"] == N) & (pq["method"] == m)]
                entry["methods"][m] = {"acc": float(x["acc"].mean()), "cost": float(x["cost"].mean()),
                                       "sampling_failure": float(1 - c["oracle"].mean()),
                                       "selection_failure": float(c["oracle"].mean() - x["acc"].mean())}
            block["by_N"][str(N)] = entry
        summary["splits"][split] = block

    ev = per_q[per_q["split"] == "eval"]
    qs = sorted(ev["sample_id"].unique())
    boot = rng.integers(0, len(qs), size=(int(cfg["bootstrap"]), len(qs)))
    ci = {}
    for N in budgets:
        tab = ev[ev["N"] == N].pivot(index="sample_id", columns="method", values="acc").loc[qs]
        ci[str(N)] = {}
        for ref in ("sc", "first"):
            for m in methods:
                if m == ref:
                    continue
                d = (tab[m] - tab[ref]).to_numpy()
                bs = d[boot].mean(axis=1)
                ci[str(N)][f"{m}-{ref}"] = {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)),
                                            "hi": float(np.percentile(bs, 97.5))}
    summary["bootstrap_eval"] = {"resamples": int(cfg["bootstrap"]), "unit": "question (after averaging over orders)",
                                 "differences": ci}
    summary["seconds"] = t()
    save_json(res / "offline_summary.json", summary)
    e = summary["splits"]["eval"]["by_N"]
    for N in budgets:
        print(ds, N, f"oracle {e[str(N)]['oracle']:.3f}", {m: round(e[str(N)]["methods"][m]["acc"], 3) for m in REQUIRED[ds]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    ap.add_argument("--datasets", default="math,mathvista")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for ds in args.datasets.split(","):
        run(cfg, ds)


if __name__ == "__main__":
    main()

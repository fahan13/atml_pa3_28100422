from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common.utils import load_config, load_json, repo_path, save_json
from task3_ttc.select import Pool, visconf


def quartile_gap(score: np.ndarray, correct: np.ndarray, k: int = 8) -> float:
    s = np.where(np.isnan(score), -np.inf, score)
    order = np.lexsort((np.arange(len(s)), -s))
    return float(correct[order[:k]].mean() - correct[order[-k:]].mean())


def scores_for(a: dict, vlm: bool) -> dict:
    out = {"length": a["T"], "logp": a["L"], "selfcert": a["K"]}
    if vlm:
        C, Ztok, q = visconf(a["kl"], a["v"])
        out.update({"vision": a["V"], "calibrated_conf": C, "visconf": a["V"] * C, "visconf_token": Ztok})
        a["_C32"], a["_q32"], a["_Ztok32"] = C, q, Ztok
    return out


def run(cfg: dict, ds: str) -> None:
    pool = Pool(cfg, ds)
    split = load_json(repo_path(cfg["results"]) / "splits.json")[ds]["eval"]
    rows = []
    for q in split:
        a = pool.arr[q]
        corr = a["correct"].astype(float)
        mixed = 0 < corr.sum() < len(corr)
        for name, s in scores_for(a, pool.vlm).items():
            ok = mixed and np.nanstd(s) > 0
            rho = spearmanr(s, corr, nan_policy="omit").statistic if ok else np.nan
            rho_len = spearmanr(s, a["T"], nan_policy="omit").statistic if np.nanstd(s) > 0 and name != "length" else np.nan
            rows.append({"sample_id": q, "score": name, "mixed": mixed, "spearman_correct": rho,
                         "quartile_gap": quartile_gap(s, corr), "spearman_length": rho_len,
                         "mean_correct": corr.mean()})
    tab = pd.DataFrame(rows)
    res = repo_path(cfg["results"]) / ds
    tab.to_csv(res / "signals_per_question.csv", index=False)
    out = {"split": "eval", "n_questions": len(split), "n_mixed": int(tab.drop_duplicates("sample_id")["mixed"].sum()),
           "quartile": "top 8 vs bottom 8 of 32 rollouts by score, ties by rollout_id", "scores": {}}
    for name, g in tab.groupby("score"):
        m = g[g["mixed"]]
        out["scores"][name] = {
            "mean_spearman_with_correct_mixed": float(m["spearman_correct"].mean()),
            "quartile_gap_all": float(g["quartile_gap"].mean()), "quartile_gap_mixed": float(m["quartile_gap"].mean()),
            "mean_spearman_with_length": float(g["spearman_length"].mean()) if name != "length" else None,
        }

    if pool.vlm:
        rc, ident = [], 0.0
        for q in split:
            a = pool.arr[q]
            C, Ztok, (q05, q95) = a["_C32"], a["_Ztok32"], a["_q32"]
            den = max(q95 - q05, 1e-8)
            for i, (k, v) in enumerate(zip(a["kl"], a["v"])):
                if len(k) < 3:
                    continue
                c = np.clip((k - q05) / den, 0, 1)
                cov = float((c * v).mean() - c.mean() * v.mean())
                ident = max(ident, abs((Ztok[i] - a["V"][i] * C[i]) - cov))
                r = pearsonr(c, v).statistic if c.std() > 0 and v.std() > 0 else np.nan
                rc.append({"sample_id": q, "rollout": i, "corr": r, "cov": cov, "correct": bool(a["correct"][i]), "T": len(k)})
        rc = pd.DataFrame(rc)
        rc.to_csv(res / "conf_vision_corr_per_rollout.csv", index=False)
        perq = rc.groupby("sample_id")["corr"].mean()
        out["confidence_vision"] = {
            "calibration": "token KL pooled over all 32 rollouts of the question, 5th/95th percentiles",
            "per_rollout_corr": {"mean": float(rc["corr"].mean()), "median": float(rc["corr"].median()),
                                 "q25": float(rc["corr"].quantile(0.25)), "q75": float(rc["corr"].quantile(0.75)),
                                 "frac_positive": float((rc["corr"] > 0).mean()), "n_rollouts": int(rc["corr"].notna().sum())},
            "per_question_mean_corr": {"mean": float(perq.mean()), "median": float(perq.median())},
            "corr_by_correctness": {str(k): float(v) for k, v in rc.groupby("correct")["corr"].mean().items()},
            "mean_cov": float(rc["cov"].mean()),
            "identity_max_abs_error": float(ident),
        }
        sel = pd.read_csv(res / "offline_selected_N32.csv", dtype={"sample_id": str})
        sel = sel[sel["order"] == 0].set_index("sample_id")
        picks = []
        for q in split:
            a = pool.arr[q]
            i_vc, i_sc = int(sel.loc[q, "visconf"]), int(sel.loc[q, "sc"])
            if a["correct"][i_vc] != a["correct"][i_sc]:
                picks.append(q)
            if len(picks) == 3:
                break
        traj = []
        roll = pool.roll.set_index(["sample_id", "rollout_id"])
        for q in picks:
            a = pool.arr[q]
            C, (q05, q95) = a["_C32"], a["_q32"]
            den = max(q95 - q05, 1e-8)
            item = {"sample_id": q, "rule": "first 3 eval questions (split order) where VisConf-highest and SC disagree in correctness at N=32, order 0", "rollouts": []}
            for role in ("visconf", "sc"):
                i = int(sel.loc[q, role])
                c = np.clip((a["kl"][i] - q05) / den, 0, 1)
                item["rollouts"].append({"selected_by": role, "rollout_id": i, "answer": None if pd.isna(roll.loc[(q, i), "key"]) else str(roll.loc[(q, i), "key"]),
                                         "correct": bool(a["correct"][i]), "T": int(a["T"][i]), "V": float(a["V"][i]), "C": float(C[i]),
                                         "Z": float(a["V"][i] * C[i]), "c_t": c.round(4).tolist(), "v_t": a["v"][i].round(4).tolist()})
            traj.append(item)
        save_json(res / "trajectories.json", traj)
    save_json(res / "signals.json", out)
    print(ds, {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out["scores"].items()})
    if pool.vlm:
        print(out["confidence_vision"])


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

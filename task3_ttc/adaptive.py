from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from common.utils import load_config, load_json, repo_path, save_json
from task3_ttc.answers import NO_ANSWER
from task3_ttc.select import Pool, rwc, visconf, vote

STEPS = [4, 8, 16, 32]


def quality_scores(pool: Pool, a: dict, idx: np.ndarray) -> np.ndarray:
    if pool.vlm:
        C, _, _ = visconf([a["kl"][i] for i in idx], [a["v"][i] for i in idx])
        return a["V"][idx] * C
    return a["K"][idx]


def precompute(pool: Pool, qids: list[str], n_orders: int) -> dict:
    shape = (len(qids), n_orders, len(STEPS))
    f = {k: np.zeros(shape) for k in ["unique", "noanswer", "frac", "agree", "margin", "sc_correct", "cost"]}
    for qi, q in enumerate(qids):
        a = pool.arr[q]
        for o in range(n_orders):
            order = pool.order(q, o)
            for si, n in enumerate(STEPS):
                idx = order[:n]
                g = a["group"][idx]
                pos, win, tot = vote(g, np.ones(n))
                top = max(tot.values())
                f["unique"][qi, o, si] = sum(v == top for v in tot.values()) == 1
                f["noanswer"][qi, o, si] = win == NO_ANSWER
                f["frac"][qi, o, si] = top / n
                f["sc_correct"][qi, o, si] = a["correct"][idx[pos]]
                f["cost"][qi, o, si] = a["T"][idx].sum()
                _, rwin, rtot = rwc(quality_scores(pool, a, idx), g)
                vals = sorted(rtot.values(), reverse=True)
                total = sum(vals)
                f["agree"][qi, o, si] = rwin == win
                f["margin"][qi, o, si] = (vals[0] - (vals[1] if len(vals) > 1 else 0.0)) / total if total > 0 else 0.0
    return f


def simulate(f: dict, tau: float | None, gamma: float | None) -> dict:
    Q, O, _ = f["cost"].shape
    stop = np.full((Q, O), len(STEPS) - 1)
    if tau is not None:
        done = np.zeros((Q, O), bool)
        for si in range(len(STEPS) - 1):
            cond = (f["unique"][..., si] > 0) & (f["noanswer"][..., si] == 0) & (f["frac"][..., si] >= tau - 1e-12)
            if gamma is not None:
                cond &= (f["agree"][..., si] > 0) & (f["margin"][..., si] >= gamma - 1e-12)
            newly = cond & ~done
            stop[newly] = si
            done |= cond
    take = lambda k: np.take_along_axis(f[k], stop[..., None], axis=2)[..., 0]
    acc, cost = take("sc_correct"), take("cost")
    n = np.array(STEPS)[stop]
    return {"acc_q": acc.mean(axis=1), "cost_q": cost.mean(axis=1), "n_q": n.mean(axis=1), "stop": stop, "correct": acc}


def summarize(sim: dict) -> dict:
    counts = np.bincount(sim["stop"].ravel(), minlength=len(STEPS))
    return {"acc": float(sim["acc_q"].mean()), "mean_cost": float(sim["cost_q"].mean()), "mean_rollouts": float(sim["n_q"].mean()),
            "stop_distribution": {str(STEPS[i]): float(c / counts.sum()) for i, c in enumerate(counts)}}


def choose(grid: list[dict], target: float) -> dict:
    ok = [g for g in grid if g["dev"]["acc"] >= target - 1e-12]
    return sorted(ok, key=lambda g: (g["dev"]["mean_cost"], -g["dev"]["acc"],
                                     g["tau"] if g["tau"] is not None else 9, g["gamma"] if g["gamma"] is not None else 9))[0]


def run(cfg: dict, ds: str) -> None:
    pool = Pool(cfg, ds)
    acfg = cfg["adaptive"]
    assert [int(x) for x in acfg["checks"]] == STEPS[:-1]
    splits = load_json(repo_path(cfg["results"]) / "splits.json")[ds]
    n_orders = int(cfg["n_orders"])
    F = {s: precompute(pool, splits[s], n_orders) for s in ("dev", "eval")}
    fixed = {s: summarize(simulate(F[s], None, None)) for s in ("dev", "eval")}
    target = fixed["dev"]["acc"] - float(acfg["tolerance"])
    families = {"agreement": [(t, None) for t in acfg["tau"]],
                "quality_aware": list(itertools.product(acfg["tau"], acfg["gamma"]))}
    out = {"dataset": ds, "quality_score": "VisConf" if pool.vlm else "self-certainty", "checks": STEPS[:-1],
           "fixed32_sc": fixed, "dev_target_acc": target, "families": {}}
    ev_q = splits["eval"]
    for fam, settings in families.items():
        grid = [{"tau": None, "gamma": None, "fallback_fixed32": True, "dev": fixed["dev"]}]
        for tau, gamma in settings:
            grid.append({"tau": float(tau), "gamma": None if gamma is None else float(gamma), "fallback_fixed32": False,
                         "dev": summarize(simulate(F["dev"], float(tau), None if gamma is None else float(gamma)))})
        best = choose(grid, target)
        sim = simulate(F["eval"], best["tau"], best["gamma"])
        ev = summarize(sim)
        fixed_sim = simulate(F["eval"], None, None)
        early = sim["stop"] < len(STEPS) - 1
        wrong_early = early & (sim["correct"] == 0)
        recover = wrong_early & (fixed_sim["correct"] == 1)
        d = sim["acc_q"] - fixed_sim["acc_q"]
        rng = np.random.default_rng(int(cfg["seed"]))
        bs = d[rng.integers(0, len(d), (int(cfg["bootstrap"]), len(d)))].mean(axis=1)
        pool_frac = np.array([pool.arr[q]["correct"].mean() for q in ev_q])
        bins = [("0", -0.01, 0.0), ("(0,0.25]", 0.0, 0.25), ("(0.25,0.5]", 0.25, 0.5), ("(0.5,0.75]", 0.5, 0.75),
                ("(0.75,1)", 0.75, 0.999), ("1", 0.999, 1.0)]
        by_bin = []
        for label, lo, hi in bins:
            m = (pool_frac > lo) & (pool_frac <= hi)
            if m.any():
                by_bin.append({"pool_correct_frac": label, "n_questions": int(m.sum()),
                               "mean_rollouts": float(sim["n_q"][m].mean()), "adaptive_acc": float(sim["acc_q"][m].mean()),
                               "fixed32_acc": float(fixed_sim["acc_q"][m].mean())})
        examples = []
        for qi, o in zip(*np.nonzero(recover)):
            examples.append({"sample_id": ev_q[qi], "order": int(o), "stopped_at": STEPS[sim["stop"][qi, o]]})
        out["families"][fam] = {
            "grid_dev": grid, "selected": {"tau": best["tau"], "gamma": best["gamma"], "fallback_fixed32": best["fallback_fixed32"]},
            "eval": ev, "eval_fixed32": summarize(fixed_sim),
            "eval_acc_diff_vs_fixed32": {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))},
            "incorrect_early_stops": {"rate_over_question_orders": float(wrong_early.mean()),
                                      "recoverable_by_fixed32_rate": float(recover.mean()),
                                      "share_of_wrong_early_recoverable": float(recover.sum() / max(wrong_early.sum(), 1)),
                                      "examples_first10": examples[:10]},
            "by_pool_correct_fraction": by_bin,
        }
    res = repo_path(cfg["results"]) / ds
    save_json(res / "adaptive.json", out)
    for fam, v in out["families"].items():
        print(ds, fam, v["selected"], {k: v["eval"][k] for k in ("acc", "mean_cost", "mean_rollouts")}, "fixed32", round(v["eval_fixed32"]["acc"], 3))


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

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common.utils import load_json, repo_path


def earliest_max(x: np.ndarray) -> int:
    x = np.where(np.isnan(x), -np.inf, x)
    return int(np.argmax(x))


def earliest_min(x: np.ndarray) -> int:
    x = np.where(np.isnan(x), np.inf, x)
    return int(np.argmin(x))


def vote(groups: np.ndarray, weights: np.ndarray) -> tuple[int, int, dict]:
    totals, first = {}, {}
    for i, (g, w) in enumerate(zip(groups.tolist(), weights.tolist())):
        totals[g] = totals.get(g, 0.0) + w
        first.setdefault(g, i)
    best = max(totals.values())
    winners = [g for g, v in totals.items() if abs(v - best) <= 1e-9]
    g = min(winners, key=lambda k: first[k])
    return first[g], g, totals


def rwc_weights(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n == 1:
        return np.ones(1)
    s = np.where(np.isnan(scores), -np.inf, scores)
    ranks = rankdata(-s, method="average")
    return (n - ranks) / (n - 1)


def rwc(scores: np.ndarray, groups: np.ndarray) -> tuple[int, int, dict]:
    return vote(groups, rwc_weights(scores))


def visconf(kl: list[np.ndarray], v: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    pooled = np.concatenate(kl)
    q05, q95 = np.percentile(pooled, [5, 95])
    den = max(q95 - q05, 1e-8)
    C, Ztok = np.empty(len(kl)), np.empty(len(kl))
    for i, (k, vv) in enumerate(zip(kl, v)):
        if len(k) == 0:
            C[i], Ztok[i] = np.nan, np.nan
            continue
        c = np.clip((k - q05) / den, 0.0, 1.0)
        C[i], Ztok[i] = c.mean(), (c * vv).mean()
    return C, Ztok, (float(q05), float(q95))


class Pool:
    def __init__(self, cfg: dict, ds: str):
        res = repo_path(cfg["results"]) / ds
        self.ds = ds
        self.vlm = cfg["datasets"][ds]["kind"] == "vlm"
        self.roll = pd.read_csv(res / "rollouts.csv", dtype={"sample_id": str}, keep_default_na=False, na_values=[""])
        self.roll = self.roll.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)
        self.qids = sorted(self.roll["sample_id"].unique().tolist())
        self.orders = np.load(res / "orders.npy")
        assert load_json(res / "orders_index.json") == self.qids
        self.qindex = {q: i for i, q in enumerate(self.qids)}
        by = self.roll.groupby("sample_id", sort=True)
        self.arr = {}
        for q, g in by:
            a = {"T": g["T"].to_numpy(float), "group": g["group"].to_numpy(int), "correct": g["correct"].to_numpy(bool),
                 "L": g["L"].to_numpy(float), "K": g["K"].to_numpy(float), "row": g.index.to_numpy()}
            if self.vlm:
                a["V"] = g["V"].to_numpy(float)
            self.arr[q] = a
        if self.vlm:
            tok = np.load(repo_path(cfg["cache"]) / ds / "tokens.npz")
            off, kl, v = tok["offsets"], tok["kl"].astype(np.float64), tok["v"].astype(np.float64)
            for q, a in self.arr.items():
                a["kl"] = [kl[off[r]:off[r + 1]] for r in a["row"]]
                a["v"] = [v[off[r]:off[r + 1]] for r in a["row"]]

    def order(self, q: str, o: int) -> np.ndarray:
        return self.orders[self.qindex[q], o]

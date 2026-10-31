from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import Timer, env_info, gpu_peak, load_config, repo_path, save_json, write_jsonl
from task1_ft import lora as L
from task2_vlm.model import DEVICE, Connector, FeatureStore, Formatter, greedy
from task2_vlm.train import load_data, nll_examples
from task2_vlm.vqa_norm import normalize, soft_accuracy

CONDITIONS = ["correct", "swap", "mismatched", "none"]


def derangement(rng, n: int) -> list[int]:
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p.tolist()


def load_model(cfg, weights_path, device=DEVICE):
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    llm = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    w = torch.load(weights_path, map_location="cpu", weights_only=False)
    conn = Connector().to(device)
    conn.load_state_dict(w["connector"])
    if w.get("lora"):
        wrapped = L.inject(llm, {**cfg["lora"], "protect_k": 0, "dropout": 0.0}, int(cfg["seed"]))
        with torch.no_grad():
            for k, m in wrapped.items():
                m.lora_A.copy_(w["lora"][f"{k}.A"])
                m.lora_B.copy_(w["lora"][f"{k}.B"])
    llm.eval()
    conn.eval()
    return tok, llm, conn


def boot(rng, values: np.ndarray, n: int) -> dict:
    idx = rng.integers(0, len(values), (n, len(values)))
    bs = values[idx].mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2/base.yaml")
    ap.add_argument("--name")
    ap.add_argument("--stage", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.name:
        cfg["name"] = args.name
    ecfg = cfg["eval"]
    res = repo_path(cfg["results"]) / cfg["name"]
    weights = repo_path(cfg["paths"]["runs"]) / cfg["name"] / f"stage{args.stage}.pt"
    data = load_data(cfg)
    fdir = repo_path(cfg["paths"]["features"])
    store = FeatureStore(fdir / "siglip_196.npy", json.loads((fdir / "index.json").read_text()), int(cfg["n_visual"]))
    tok, llm, conn = load_model(cfg, weights)
    fmt = Formatter(tok, cfg)
    rng = np.random.default_rng(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = {"name": cfg["name"], "stage": args.stage, "suites": {}, "mismatch_mapping": {}}
    rows_all = []
    for suite in ("grounding_test", "vqa_test"):
        pairs = data[suite]
        perm = derangement(rng, len(pairs))
        out["mismatch_mapping"][suite] = perm
        ex = [e for p in pairs for e in p]
        image_for = {
            "correct": [e["image_id"] for e in ex],
            "swap": [pairs[i][1 - j]["image_id"] for i in range(len(pairs)) for j in range(2)],
            "mismatched": [pairs[perm[i]][j]["image_id"] for i in range(len(pairs)) for j in range(2)],
            "none": [None] * len(ex),
        }
        partner_canon = [normalize(pairs[i][1 - j]["canonical"]) for i in range(len(pairs)) for j in range(2)]
        per = {}
        for cond in CONDITIONS:
            imgs = image_for[cond]
            preds, secs = [], 0.0
            bs = int(ecfg["batch_size"])
            for i in range(0, len(ex), bs):
                chunk = ex[i:i + bs]
                feats = store.get(imgs[i:i + bs]) if cond != "none" else None
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t = Timer()
                with torch.autocast("cuda", dtype=torch.float16):
                    preds += greedy(llm, conn, fmt, chunk, feats, cond != "none", int(ecfg["max_new_tokens"]))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                secs += t()
            lookup = {id(e): im for e, im in zip(ex, imgs)}
            nll = nll_examples(llm, conn, fmt, store, ex, (lambda e: lookup[id(e)]) if cond != "none" else None, False)
            rows = []
            for k, (e, p) in enumerate(zip(ex, preds)):
                pn = normalize(p["text"])
                score = soft_accuracy(p["text"], e["answers"]) if suite == "vqa_test" else float(pn == normalize(e["canonical"]))
                rows.append({"suite": suite, "condition": cond, "pair_id": e["pair_id"], "position": k % 2, "type": e["kind"],
                             "question": e["question"], "image_id": imgs[k], "reference": e["answer"], "canonical": e["canonical"],
                             "prediction": p["text"], "pred_norm": pn, "score": score, "match_canonical": pn == normalize(e["canonical"]),
                             "follows_partner": pn == partner_canon[k], "empty": pn == "", "nll": nll[k], "n_tokens": p["n_tokens"]})
            per[cond] = rows
            rows_all += rows
            P = len(pairs)
            pair_acc = np.array([rows[2 * i]["match_canonical"] and rows[2 * i + 1]["match_canonical"] for i in range(P)], float)
            ident = np.array([rows[2 * i]["pred_norm"] == rows[2 * i + 1]["pred_norm"] for i in range(P)], float)
            score_pair = np.array([(rows[2 * i]["score"] + rows[2 * i + 1]["score"]) / 2 for i in range(P)])
            nll_pair = np.array([(rows[2 * i]["nll"] + rows[2 * i + 1]["nll"]) / 2 for i in range(P)])
            by_type = {}
            for t in sorted({r["type"] for r in rows}):
                by_type[t] = float(np.mean([r["score"] for r in rows if r["type"] == t]))
            out["suites"].setdefault(suite, {})[cond] = {
                "accuracy": float(np.mean([r["score"] for r in rows])), "by_type": by_type,
                "pair_accuracy": float(pair_acc.mean()), "identical_answer_rate": float(ident.mean()),
                "empty_answer_rate": float(np.mean([r["empty"] for r in rows])), "mean_reference_nll": float(np.mean(nll)),
                "follows_partner_rate": float(np.mean([r["follows_partner"] for r in rows])),
                "seconds_per_answer": secs / len(ex), "n_questions": len(ex), "n_pairs": P,
                "ci": {"accuracy": boot(np.random.default_rng(int(cfg["seed"])), score_pair, int(ecfg["bootstrap"])),
                       "pair_accuracy": boot(np.random.default_rng(int(cfg["seed"])), pair_acc, int(ecfg["bootstrap"])),
                       "nll": boot(np.random.default_rng(int(cfg["seed"])), nll_pair, int(ecfg["bootstrap"]))},
            }
        adv = {}
        c = per["correct"]
        for other in ("swap", "mismatched", "none"):
            o = per[other]
            P = len(pairs)
            d_acc = np.array([(c[2 * i]["score"] + c[2 * i + 1]["score"] - o[2 * i]["score"] - o[2 * i + 1]["score"]) / 2 for i in range(P)])
            d_nll = np.array([(o[2 * i]["nll"] + o[2 * i + 1]["nll"] - c[2 * i]["nll"] - c[2 * i + 1]["nll"]) / 2 for i in range(P)])
            d_pair = np.array([float(c[2 * i]["match_canonical"] and c[2 * i + 1]["match_canonical"])
                               - float(o[2 * i]["match_canonical"] and o[2 * i + 1]["match_canonical"]) for i in range(P)])
            adv[f"correct_vs_{other}"] = {
                "accuracy_advantage": boot(np.random.default_rng(int(cfg["seed"])), d_acc, int(ecfg["bootstrap"])),
                "nll_advantage": boot(np.random.default_rng(int(cfg["seed"])), d_nll, int(ecfg["bootstrap"])),
                "pair_accuracy_advantage": boot(np.random.default_rng(int(cfg["seed"])), d_pair, int(ecfg["bootstrap"]))}
        out["suites"][suite]["advantages"] = adv
    out["bootstrap"] = {"resamples": int(ecfg["bootstrap"]), "unit": "complete pair; same seed for every metric, so resampled pair ids match across conditions"}
    out.update(gpu_peak())
    out["env"] = env_info()
    write_jsonl(res / f"final_eval_stage{args.stage}_examples.jsonl", rows_all)
    save_json(res / f"final_eval_stage{args.stage}.json", out)
    for suite in ("grounding_test", "vqa_test"):
        print(suite, {c: round(out["suites"][suite][c]["accuracy"], 3) for c in CONDITIONS},
              "pair", {c: round(out["suites"][suite][c]["pair_accuracy"], 3) for c in CONDITIONS})


if __name__ == "__main__":
    main()

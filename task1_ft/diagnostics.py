from __future__ import annotations

import argparse
import re
import statistics

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import load_config, repo_path, save_json, write_jsonl
from task1_ft.conditions import adapter_factors, condition, run_dir
from task1_ft.data import chat_prompt

CONDITIONS = ["full_ft_mid", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]


def hellaswag_query(doc: dict) -> str:
    def pre(text: str) -> str:
        text = text.strip().replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        return text.replace("  ", " ")
    return pre(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())


def probe_set(cfg: dict, tok) -> list[dict]:
    from datasets import load_dataset

    path = repo_path(cfg["eval"]["diagnostics"]["results"]) / "probe_ids.json"
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    rng = np.random.default_rng(int(cfg["seed"]))
    gi = sorted(int(x) for x in rng.choice(len(gsm), int(cfg["eval"]["diagnostics"]["n_math"]), replace=False))
    hi = sorted(int(x) for x in rng.choice(len(hs), int(cfg["eval"]["diagnostics"]["n_nonmath"]), replace=False))
    save_json(path, {"seed": int(cfg["seed"]), "gsm8k_test": gi, "hellaswag_validation": hi,
                     "hellaswag_ind": [hs[i]["ind"] for i in hi]})
    rows = [{"domain": "math", "id": i, "text": chat_prompt(tok, f"Q: {gsm[i]['question']}\nA:")} for i in gi]
    rows += [{"domain": "nonmath", "id": i, "text": hellaswag_query(hs[i])} for i in hi]
    return rows


@torch.no_grad()
def cache_inputs(model, tok, rows, layers) -> dict:
    store = {}
    hooks = []
    for i in layers:
        def hook(mod, inp, i=i):
            store[i] = inp[0][0, -1].detach().float().cpu()
        hooks.append(model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook))
    z = {i: [] for i in layers}
    for r in rows:
        enc = tok(r["text"], return_tensors="pt", add_special_tokens=False).to(model.device)
        model(**enc)
        for i in layers:
            z[i].append(store[i])
    for h in hooks:
        h.remove()
    return {i: torch.stack(v) for i, v in z.items()}


def delta(cfg: dict, name: str, layer: int, w0: torch.Tensor) -> torch.Tensor:
    c = condition(cfg, name)
    if c["kind"] == "full":
        d = torch.load(run_dir(cfg, name) / f"diag_u{c['update']}.pt", weights_only=False)
        return d[f"{layer}.down_proj.W"].float() - w0
    factors, scaling, _ = adapter_factors(cfg, name)
    return scaling * (factors[f"{layer}.down_proj.B"].float() @ factors[f"{layer}.down_proj.A"].float())


def summarize(vals: list[float]) -> dict:
    return {"mean": statistics.fmean(vals), "median": statistics.median(vals), "std": statistics.pstdev(vals), "n": len(vals)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1/eval.yaml")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["eval"]["diagnostics"]
    out = repo_path(dcfg["results"])
    layers = [int(x) for x in cfg["diag_layers"]]
    k = int(dcfg["protect_k"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(device).eval()
    rows = probe_set(cfg, tok)
    z = cache_inputs(model, tok, rows, layers)
    w0 = {i: model.model.layers[i].mlp.down_proj.weight.detach().float().cpu() for i in layers}
    del model
    bases = {}
    for i in layers:
        u, _, vh = torch.linalg.svd(w0[i], full_matrices=False)
        bases[i] = (u[:, :k], vh[:k].T)

    weight_rows, act_rows, act_summary = [], [], []
    for name in args.conditions.split(","):
        for i in layers:
            dw = delta(cfg, name, i, w0[i])
            U, V = bases[i]
            fro2 = float(dw.pow(2).sum())
            weight_rows.append({
                "condition": name, "layer": i,
                "rho_k": float((U.T @ dw).pow(2).sum()) / fro2 if fro2 > 0 else None,
                "rho_k_input": float((dw @ V).pow(2).sum()) / fro2 if fro2 > 0 else None,
                "rho_k_isotropic": k / dw.shape[0],
                "eta": fro2 ** 0.5 / float(w0[i].norm()),
                "delta_fro": fro2 ** 0.5, "w0_fro": float(w0[i].norm()),
            })
            num = (z[i] @ dw.T).norm(dim=1)
            den = (z[i] @ w0[i].T).norm(dim=1) + 1e-8
            ratio = (num / den).tolist()
            for r, v in zip(rows, ratio):
                act_rows.append({"condition": name, "layer": i, "domain": r["domain"], "probe_id": r["id"], "R": v})
            for dom in ("math", "nonmath"):
                vals = [v for r, v in zip(rows, ratio) if r["domain"] == dom]
                act_summary.append({"condition": name, "layer": i, "domain": dom, **summarize(vals)})
    save_json(out / "weight_space.json", weight_rows)
    save_json(out / "activation_summary.json", act_summary)
    write_jsonl(out / "activation_per_example.jsonl", act_rows)
    for r in weight_rows:
        print({k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in r.items()})


if __name__ == "__main__":
    main()

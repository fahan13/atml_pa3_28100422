from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import Timer, load_config, repo_path, save_json, size_gb
from task1_ft import lora as L
from task1_ft.conditions import adapter_factors, condition, model_path
from task1_ft.data import chat_prompt

CHECK_TEXTS = [
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA:",
    "Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nA:",
]
PLAIN_TEXTS = [
    "Roof shingle removal: A man is sitting on a roof. He",
    "Question: Which of these would help to prevent infections from spreading?\nAnswer:",
]


def check_batch(tok, device):
    texts = [chat_prompt(tok, t) for t in CHECK_TEXTS] + PLAIN_TEXTS
    tok.padding_side = "right"
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)


@torch.no_grad()
def logits(model, batch):
    return model(**batch).logits.float()


def unmerged_model(cfg, factors, scaling, device):
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float16).to(device).eval()
    lcfg = dict(cfg["lora"])
    lcfg["protect_k"] = 0
    lcfg["dropout"] = 0.0
    lcfg["r"] = factors["0.q_proj.A"].shape[0]
    wrapped = L.inject(model, lcfg, int(cfg["seed"]))
    with torch.no_grad():
        for key, mod in wrapped.items():
            mod.lora_A.copy_(factors[f"{key}.A"].to(device))
            mod.lora_B.copy_(factors[f"{key}.B"].to(device))
            mod.scaling = scaling
    return model


@torch.no_grad()
def oplora_constraints(cfg, name, device) -> dict:
    from safetensors.torch import load_file

    from task1_ft.conditions import run_dir

    c = condition(cfg, name)
    d = run_dir(cfg, name)
    raw = load_file(str(d / f"adapter_raw_u{c['update']}.safetensors"))
    proj = load_file(str(d / f"adapter_projected_u{c['update']}.safetensors"))
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32)
    k = int(cfg["lora"]["protect_k"])
    worst = {"rel_UtB": 0.0, "rel_AV": 0.0, "proj_consistency": 0.0}
    for key in sorted({x.rsplit(".", 1)[0] for x in proj}):
        i, mname = key.split(".")
        layer = base.model.layers[int(i)]
        w = getattr(layer.self_attn if mname in L.ATTN else layer.mlp, mname).weight.to(device)
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        U, V = u[:, :k], vh[:k].T
        a, b = proj[f"{key}.A"].to(device), proj[f"{key}.B"].to(device)
        ra, rb = raw[f"{key}.A"].to(device), raw[f"{key}.B"].to(device)
        worst["rel_UtB"] = max(worst["rel_UtB"], float((U.T @ b).norm() / (b.norm() + 1e-12)))
        worst["rel_AV"] = max(worst["rel_AV"], float((a @ V).norm() / (a.norm() + 1e-12)))
        pa, pb = ra - (ra @ V) @ V.T, rb - U @ (U.T @ rb)
        worst["proj_consistency"] = max(worst["proj_consistency"], float((pa - a).abs().max()), float((pb - b).abs().max()))
    return {"name": name, "protect_k": k, "max_relative_norm": worst,
            "note": "rel_UtB = ||U_k^T B~||_F / ||B~||_F, rel_AV = ||A~ V_k||_F / ||A~||_F, bases recomputed from W0 in fp32"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1/eval.yaml")
    ap.add_argument("--conditions", default="lora4,lora16,oplora16")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    for name in args.conditions.split(","):
        out = repo_path(model_path(cfg, name))
        if (out / "config.json").exists():
            print(f"{name}: merged model exists, skipping")
            continue
        factors, scaling, kind = adapter_factors(cfg, name)
        t = Timer()
        merged = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float16).to(device).eval()
        L.merge_into(merged, factors, scaling)
        merge_s = t()
        batch = check_batch(tok, device)
        lm = logits(merged, batch)
        unm = unmerged_model(cfg, factors, scaling, device)
        lu = logits(unm, batch)
        del unm
        mask = batch["attention_mask"].bool()
        diff = (lm - lu).abs()[mask]
        agree = (lm.argmax(-1) == lu.argmax(-1))[mask].float().mean()
        res = {"name": name, "factors": kind, "scaling": scaling, "positions": int(mask.sum()),
               "max_abs_logit_diff": float(diff.max()), "mean_abs_logit_diff": float(diff.mean()),
               "argmax_agreement": float(agree), "merge_s": merge_s}
        if kind == "projected":
            res["oplora_constraints"] = oplora_constraints(cfg, name, device)
        merged.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        res["merged_gb"] = size_gb(out)
        save_json(f"results/sanity/task1_merge_{name}.json", res)
        print(res)
        del merged
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

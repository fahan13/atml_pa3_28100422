from __future__ import annotations

import argparse

import torch

from common.utils import load_config, read_jsonl, repo_path, save_json


def final_state(cfg: dict, name: str) -> dict:
    ckpt = sorted((repo_path(cfg["out_root"]) / name / "ckpt").glob("update_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
    return torch.load(ckpt, map_location="cpu", weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--straight", required=True)
    ap.add_argument("--resumed", required=True)
    ap.add_argument("--out", default="results/sanity/task1_resume.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = repo_path(cfg["results_root"])
    a = {r["update"]: r for r in read_jsonl(root / args.straight / "train_log.jsonl")}
    b = {r["update"]: r for r in read_jsonl(root / args.resumed / "train_log.jsonl")}
    shared = sorted(set(a) & set(b))
    loss_diff = {u: abs(a[u]["sft_loss"] - b[u]["sft_loss"]) for u in shared}
    sa, sb = final_state(cfg, args.straight), final_state(cfg, args.resumed)
    pdiff = max(float((sa["trainable"][k] - sb["trainable"][k]).abs().max()) for k in sa["trainable"])
    sessions = read_jsonl(root / args.resumed / "sessions.jsonl")
    out = {
        "config": args.config, "updates_compared": shared, "max_loss_abs_diff": max(loss_diff.values()),
        "loss_abs_diff": loss_diff, "final_update": sa["update"], "max_param_abs_diff": pdiff,
        "resumed_sessions": sessions, "pass": max(loss_diff.values()) < 1e-3 and pdiff < 1e-3 and len(sessions) >= 2,
    }
    save_json(args.out, out)
    print(out)


if __name__ == "__main__":
    main()

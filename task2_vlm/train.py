from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import (Timer, append_jsonl, ensure_dir, env_info, gpu_peak, load_config, read_jsonl, repo_path,
                          rng_state, save_json, set_rng_state, set_seed, tensor_hash, write_jsonl)
from task1_ft import lora as L
from task2_vlm.model import DEVICE, Connector, FeatureStore, Formatter, per_example_nll, train_batch


def pair_examples(pairs: list[dict]) -> list[tuple[dict, dict]]:
    out = []
    for p in pairs:
        out.append(tuple({"image_id": it["image_id"], "question": p["question"], "answer": it["answer"], "kind": p["type"],
                          "pair_id": p["pair_id"], "canonical": it["canonical"], "answers": it.get("answers")} for it in p["items"]))
    return out


def load_data(cfg: dict) -> dict:
    built = json.loads(repo_path(cfg["paths"]["built"]).read_text(encoding="utf-8"))
    cap = lambda rows: [{"image_id": r["image_id"], "question": None, "answer": r["caption"], "kind": "caption"} for r in rows]
    return {"caption": cap(built["captions_train"]), "caption_dev": cap(built["captions_dev"]),
            "presence": pair_examples(built["presence_pool"]), "counting": pair_examples(built["counting_pool"]),
            "vqa": pair_examples(built["vqa_train"]), "grounding_dev": pair_examples(built["grounding_dev"]),
            "vqa_dev": pair_examples(built["vqa_dev"]), "grounding_test": pair_examples(built["grounding_test"]),
            "vqa_test": pair_examples(built["vqa_test"])}


class Sampler:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.perm, self.pos = None, 0

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "perm": None if self.perm is None else self.perm.tolist(), "pos": self.pos}

    def load(self, s: dict) -> None:
        self.rng.bit_generator.state = s["rng"]
        self.perm = None if s["perm"] is None else np.array(s["perm"])
        self.pos = s["pos"]

    def sequential(self, n: int, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if self.perm is None or self.pos >= n:
                self.perm, self.pos = self.rng.permutation(n), 0
            out.append(int(self.perm[self.pos]))
            self.pos += 1
        return out

    def draws(self, mix: dict, data: dict, n_draws: int) -> tuple[list[dict], dict]:
        names = list(mix)
        probs = np.array([float(mix[k]) for k in names])
        examples, counts = [], {k: 0 for k in names}
        for _ in range(n_draws):
            c = names[int(self.rng.choice(len(names), p=probs / probs.sum()))]
            counts[c] += 1
            if c == "caption":
                examples += [data["caption"][int(i)] for i in self.rng.integers(0, len(data["caption"]), 2)]
            else:
                examples += list(data[c][int(self.rng.integers(len(data[c])))])
        return examples, counts


def lr_lambda(updates: int, warm_frac: float, final_frac: float):
    warm = max(1, round(warm_frac * updates))

    def f(s):
        if s < warm:
            return s / warm
        prog = (s - warm) / max(1, updates - warm)
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return f


def groups(params: list, lr: float, wd: float) -> list[dict]:
    dec = [p for n, p in params if p.ndim >= 2]
    nodec = [p for n, p in params if p.ndim < 2]
    return [{"params": dec, "lr": lr, "weight_decay": wd}, {"params": nodec, "lr": lr, "weight_decay": 0.0}]


@torch.no_grad()
def nll_examples(llm, conn, fmt, store, examples, image_of, with_eot: bool, bs: int = 16) -> list[float]:
    out = []
    for i in range(0, len(examples), bs):
        chunk = examples[i:i + bs]
        img = image_of is not None
        feats = store.get([image_of(e) for e in chunk]) if img else None
        batch = train_batch(fmt, chunk, image=img, with_eot=with_eot)
        with torch.autocast("cuda", dtype=torch.float16):
            out += per_example_nll(llm, conn, batch, feats).float().tolist()
    return out


def dev_eval(llm, conn, fmt, store, data) -> dict:
    llm.eval()
    res = {"caption_loss": float(np.mean(nll_examples(llm, conn, fmt, store, data["caption_dev"], lambda e: e["image_id"], True)))}
    for suite in ("grounding_dev", "vqa_dev"):
        ex = [e for p in data[suite] for e in p]
        partner = {}
        for a, b in data[suite]:
            partner[id(a)], partner[id(b)] = b["image_id"], a["image_id"]
        c = nll_examples(llm, conn, fmt, store, ex, lambda e: e["image_id"], False)
        s = nll_examples(llm, conn, fmt, store, ex, lambda e: partner[id(e)], False)
        n = nll_examples(llm, conn, fmt, store, ex, None, False)
        res[suite] = {"nll_correct": float(np.mean(c)), "nll_swap": float(np.mean(s)), "nll_none": float(np.mean(n)),
                      "swap_penalty": float(np.mean(np.array(s) - np.array(c)))}
    llm.train()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2/base.yaml")
    ap.add_argument("--name")
    ap.add_argument("--updates", help="comma-separated per-stage updates override (smoke)")
    ap.add_argument("--dev_every", type=int)
    ap.add_argument("--ckpt_every", type=int)
    ap.add_argument("--stop_after", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.name:
        cfg["name"] = args.name
    tc = cfg["train"]
    if args.updates:
        for st, u in zip(tc["stages"], args.updates.split(",")):
            st["updates"] = int(u)
    if args.dev_every:
        tc["dev_every"] = args.dev_every
    if args.ckpt_every:
        tc["checkpoint_every"] = args.ckpt_every
    out_dir = ensure_dir(f"{cfg['paths']['runs']}/{cfg['name']}")
    ckpt_dir = ensure_dir(out_dir / "ckpt")
    res = ensure_dir(f"{cfg['results']}/{cfg['name']}")
    total = sum(int(s["updates"]) for s in tc["stages"])
    if (res / "run_summary.json").exists() and read_jsonl(res / "train_log.jsonl") and read_jsonl(res / "train_log.jsonl")[-1]["update"] >= total:
        print(f"{cfg['name']} complete, skipping")
        return

    set_seed(int(cfg["seed"]))
    device = torch.device(DEVICE)
    data = load_data(cfg)
    fdir = repo_path(cfg["paths"]["features"])
    store = FeatureStore(fdir / "siglip_196.npy", json.loads((fdir / "index.json").read_text()), int(cfg["n_visual"]))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    llm = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float16, attn_implementation="sdpa").to(device)
    for p in llm.parameters():
        p.requires_grad_(False)
    llm.config.use_cache = False
    if tc["grad_checkpointing"]:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train()
    fmt = Formatter(tok, cfg)
    conn = Connector().to(device)
    base_hash_before = tensor_hash(llm.model.layers[0].self_attn.q_proj.weight) + tensor_hash(llm.model.embed_tokens.weight[:1000])
    sampler = Sampler(int(cfg["seed"]))
    scaler = torch.amp.GradScaler("cuda")
    wrapped = {}
    log_path, dev_path = res / "train_log.jsonl", res / "dev_log.jsonl"

    found = sorted(ckpt_dir.glob("update_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    state = torch.load(found[-1], map_location="cpu", weights_only=False) if found else None
    if state is None:
        for f in ("train_log.jsonl", "dev_log.jsonl", "stages.jsonl"):
            (res / f).unlink(missing_ok=True)
        cal = calibrate(cfg, tok, llm, conn, store, data)
        save_json(res / "calibration.json", cal)
        d0 = dev_eval(llm, conn, fmt, store, data)
        append_jsonl(dev_path, {"update": 0, "stage": 0, **d0})
        stage0, step0, u = 0, 0, 0
        elapsed = {}
    else:
        conn.load_state_dict(state["connector"])
        stage0, step0, u, elapsed = state["stage"], state["step"], state["update"], state["elapsed"]
        if stage0 >= 1:
            wrapped = L.inject(llm, {**cfg["lora"], "protect_k": 0}, int(cfg["seed"]))
            with torch.no_grad():
                for k, m in wrapped.items():
                    m.lora_A.copy_(state["lora"][f"{k}.A"])
                    m.lora_B.copy_(state["lora"][f"{k}.B"])
        sampler.load(state["sampler"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        write_jsonl(log_path, [r for r in read_jsonl(log_path) if r["update"] <= u])
        write_jsonl(dev_path, [r for r in read_jsonl(dev_path) if r["update"] <= u])
        print(f"resumed at update {u} (stage {stage0 + 1}, step {step0})")

    sanity = {"template": fmt.check_template(), "n_visual": int(cfg["n_visual"]),
              "frozen_llm_params_requiring_grad": 0, "stages": {}}
    micro, accum = int(tc["micro_batch"]), int(tc["grad_accum"])
    eff = micro * accum
    for si, st in enumerate(tc["stages"]):
        if si < stage0:
            continue
        if si >= 1 and not wrapped:
            wrapped = L.inject(llm, {**cfg["lora"], "protect_k": 0}, int(cfg["seed"]))
        params = [("conn." + n, p) for n, p in conn.named_parameters()]
        lora_params = [(f"{k}.{n}", p) for k, m in wrapped.items() for n, p in (("A", m.lora_A), ("B", m.lora_B))]
        for _, p in lora_params:
            p.requires_grad_(si >= 1)
        pg = groups(params, float(st["lr_connector"]), float(tc["weight_decay"]))
        if si >= 1:
            pg += groups(lora_params, float(st["lr_lora"]), float(tc["weight_decay"]))
        opt = torch.optim.AdamW(pg, betas=(0.9, 0.999), eps=1e-8)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(st["updates"]), float(tc["warmup_frac"]), float(tc["final_lr_frac"])))
        start = 0
        if state is not None and si == stage0 and step0 > 0:
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            start = step0
        trainable = [p for _, p in params] + ([p for _, p in lora_params] if si >= 1 else [])
        n_train = sum(p.numel() for p in trainable)
        sanity["frozen_llm_params_requiring_grad"] = sum(p.numel() for n, p in llm.named_parameters() if p.requires_grad and "lora_" not in n)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_stage = Timer()
        for step in range(start, int(st["updates"])):
            t = Timer()
            if si == 0:
                idx = sampler.sequential(len(data["caption"]), eff)
                examples, counts = [data["caption"][i] for i in idx], {"caption": eff // 2}
            else:
                examples, counts = sampler.draws(st["mix"], data, eff // 2)
            lr_now = [g["lr"] for g in opt.param_groups]
            loss_sum, n_tok = 0.0, 0
            for m in range(accum):
                chunk = examples[m * micro:(m + 1) * micro]
                feats = store.get([e["image_id"] for e in chunk])
                batch = train_batch(fmt, chunk)
                n_tok += int(batch["attention_mask"].sum())
                with torch.autocast("cuda", dtype=torch.float16):
                    nll = per_example_nll(llm, conn, batch, feats)
                    loss = nll.sum() / eff
                scaler.scale(loss).backward()
                loss_sum += float(nll.detach().sum())
            scaler.unscale_(opt)
            if step == start and str(si) not in sanity["stages"]:
                sanity["stages"][str(si)] = {
                    "connector_grad_norm": float(torch.norm(torch.stack([p.grad.norm() for p in conn.parameters() if p.grad is not None]))),
                    "lora_grad_norm": float(torch.norm(torch.stack([p.grad.norm() for _, p in lora_params if p.grad is not None]))) if si >= 1 else None,
                    "trainable_params": n_train}
            gn = float(torch.nn.utils.clip_grad_norm_(trainable, float(tc["max_grad_norm"])))
            sc = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            u += 1
            append_jsonl(log_path, {"update": u, "stage": si + 1, "step": step + 1, "loss": loss_sum / eff, "lr_connector": lr_now[0],
                                    "lr_lora": lr_now[2] if si >= 1 else None, "grad_norm": gn if math.isfinite(gn) else None,
                                    "skipped": scaler.get_scale() < sc, "mix": counts, "tokens": n_tok, "seconds": t()})
            if u % 50 == 0:
                print(f"[{cfg['name']}] stage {si + 1} update {u}/{total} loss {loss_sum / eff:.4f}", flush=True)
            last = step + 1 == int(st["updates"])
            if u % int(tc["dev_every"]) == 0 or last:
                append_jsonl(dev_path, {"update": u, "stage": si + 1, **dev_eval(llm, conn, fmt, store, data)})
            stop_now = args.stop_after is not None and u >= args.stop_after
            if u % int(tc["checkpoint_every"]) == 0 or last or stop_now:
                ck = {"connector": conn.state_dict(), "lora": L.adapter_state(wrapped, projected=False) if wrapped else None,
                      "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                      "sampler": sampler.state(), "stage": si if not last else si + 1, "step": step + 1 if not last else 0,
                      "update": u, "rng": rng_state(), "elapsed": elapsed}
                path = ckpt_dir / f"update_{u}.pt"
                torch.save(ck, path.with_suffix(".tmp"))
                path.with_suffix(".tmp").replace(path)
                for old in sorted(ckpt_dir.glob("update_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[:-1]:
                    old.unlink()
            if stop_now:
                print(f"stopping after update {u}")
                return
        elapsed[str(si + 1)] = elapsed.get(str(si + 1), 0.0) + t_stage()
        torch.save({"connector": conn.state_dict(), "lora": L.adapter_state(wrapped, projected=False) if wrapped else None},
                   out_dir / f"stage{si + 1}.pt")
        append_jsonl(res / "stages.jsonl", {"stage": si + 1, "updates": int(st["updates"]), "seconds_this_session": t_stage(),
                                             "trainable_params": n_train, **gpu_peak()})
        state = None
    sanity["base_weights_unchanged"] = base_hash_before == (tensor_hash(llm.model.layers[0].self_attn.q_proj.base.weight if wrapped else llm.model.layers[0].self_attn.q_proj.weight)
                                                          + tensor_hash(llm.model.embed_tokens.weight[:1000]))
    save_json(f"results/sanity/task2_{cfg['name']}_checks.json", sanity)
    save_json(res / "run_summary.json", {"name": cfg["name"], "n_visual": int(cfg["n_visual"]), "updates": u, "elapsed_s": elapsed,
                                         "stages": read_jsonl(res / "stages.jsonl"), "skipped_updates": [r["update"] for r in read_jsonl(log_path) if r["skipped"]],
                                         "examples_presented": u * eff, "env": env_info(), "config": cfg})
    print(f"done {cfg['name']}: {u} updates")


@torch.no_grad()
def calibrate(cfg, tok, llm, conn, store, data) -> dict:
    ccfg = cfg["calibration"]
    rng = np.random.default_rng(int(cfg["seed"]))
    texts = [e["answer"] for e in data["caption"]] + [e["question"] for p in data["vqa"] for e in p] + [e["answer"] for p in data["vqa"] for e in p]
    pick = [texts[int(i)] for i in rng.choice(len(texts), int(ccfg["n_text"]), replace=False)]
    ids = torch.tensor([t for s in pick for t in tok(s, add_special_tokens=False)["input_ids"]], device=DEVICE)
    text_norm = float(llm.get_input_embeddings()(ids).float().norm(dim=-1).mean())
    imgs = sorted({e["image_id"] for e in data["caption"]})
    sel = [imgs[int(i)] for i in rng.choice(len(imgs), int(ccfg["n_images"]), replace=False)]
    feats = store.get(sel)
    vis_norm = float(conn(feats, scaled=False).float().norm(dim=-1).mean())
    conn.scale.fill_(text_norm / vis_norm)
    return {"rule": "scale = mean L2 norm of Qwen input embeddings of training-text tokens / mean L2 norm of unscaled connector outputs at initialization on training images",
            "text_tokens": int(len(ids)), "n_texts": len(pick), "n_images": len(sel), "text_embedding_norm": text_norm,
            "connector_output_norm_init": vis_norm, "scale": text_norm / vis_norm}


if __name__ == "__main__":
    main()

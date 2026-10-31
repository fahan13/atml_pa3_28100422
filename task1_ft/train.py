from __future__ import annotations

import argparse
import math
import shutil

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common.utils import (
    Timer,
    append_jsonl,
    ensure_dir,
    env_info,
    gpu_peak,
    load_config,
    read_jsonl,
    repo_path,
    rng_state,
    save_json,
    set_rng_state,
    set_seed,
    size_gb,
    tensor_hash,
    write_jsonl,
)
from task1_ft import lora as L
from task1_ft.data import load_cached


class UpdateBatches:
    def __init__(self, data: dict, micro: int, accum: int, pad_id: int):
        self.p, self.t = data["prompt"], data["target"]
        self.micro, self.accum, self.pad = micro, accum, pad_id

    def indices(self, update: int) -> list[list[int]]:
        start = update * self.micro * self.accum
        return [list(range(start + m * self.micro, start + (m + 1) * self.micro)) for m in range(self.accum)]

    def n_target_tokens(self, update: int) -> int:
        return sum(len(self.t[i]) for mb in self.indices(update) for i in mb)

    def collate(self, idx: list[int], device) -> dict:
        seqs = [self.p[i] + self.t[i] for i in idx]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(idx), T), self.pad, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        lab = torch.full((len(idx), T), -100, dtype=torch.long)
        for r, i in enumerate(idx):
            n, np_ = len(seqs[r]), len(self.p[i])
            ids[r, :n] = torch.tensor(seqs[r])
            att[r, :n] = 1
            lab[r, np_:n] = torch.tensor(self.t[i])
        return {"input_ids": ids.to(device), "attention_mask": att.to(device), "labels": lab.to(device)}


def response_loss(model, batch: dict) -> tuple[torch.Tensor, int]:
    h = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
    tgt = batch["labels"][:, 1:]
    mask = tgt != -100
    logits = model.lm_head(h[:, :-1][mask]).float()
    return F.cross_entropy(logits, tgt[mask], reduction="sum"), int(mask.sum())


def param_groups(model, wd: float) -> list[dict]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and not n.endswith(".bias") else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def trainable_named(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def load_reference(cfg: dict, names: list[str], device) -> list[torch.Tensor]:
    ref = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32)
    sd = dict(ref.named_parameters())
    out = [sd[n].detach().to(device) for n in names]
    del ref
    return out


@torch.no_grad()
def l2sp_grad(params: list, refs: list, lam: float) -> float:
    total = 0.0
    for p, r in zip(params, refs):
        d = p.detach() - r.to(p.device, non_blocking=True)
        if p.grad is not None:
            p.grad.add_(d, alpha=lam)
        total += float(d.pow(2).sum())
    return 0.5 * total


@torch.no_grad()
def displacement(params: list, refs: list) -> float:
    total = 0.0
    for p, r in zip(params, refs):
        total += float((p.detach() - r.to(p.device)).pow(2).sum())
    return math.sqrt(total)


def build_model(cfg: dict, device):
    full = cfg["method"] == "full"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.float32 if full else torch.float16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    wrapped, extra = {}, {}
    if not full:
        for p in model.parameters():
            p.requires_grad_(False)
        lcfg = dict(cfg["lora"])
        if cfg["method"] == "lora":
            lcfg["protect_k"] = 0
        wrapped = L.inject(model, lcfg, int(cfg["seed"]))
        extra["init_A_hash"] = tensor_hash(torch.cat([m.lora_A.flatten() for m in wrapped.values()]))
        if lcfg["protect_k"]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = Timer()
            L.compute_bases(wrapped)
            if device.type == "cuda":
                torch.cuda.synchronize()
            extra["basis_preprocess_s"] = t()
            extra["basis_bytes"] = L.basis_bytes(wrapped)
    if cfg["train"]["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, wrapped, extra


def save_recovery(path, model, opt, sched, scaler, update: int, examples: int, elapsed: dict) -> None:
    state = {
        "update": update,
        "examples_seen": examples,
        "data_position": examples,
        "trainable": {n: p.detach().cpu() for n, p in trainable_named(model)},
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng_state(),
        "elapsed": elapsed,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def latest_checkpoint(ckpt_dir):
    found = sorted(ckpt_dir.glob("update_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return found[-1] if found else None


@torch.no_grad()
def export(cfg, model, wrapped, tok, update: int, out_dir, res_dir, refs, ref_names) -> dict:
    t = Timer()
    info = {"update": update}
    layers = [int(x) for x in cfg["diag_layers"]]
    if cfg["method"] == "full":
        d = out_dir / f"export_u{update}"
        sd = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
        model.save_pretrained(d, state_dict=sd, safe_serialization=True)
        tok.save_pretrained(d)
        diag = {f"{i}.down_proj.W": model.model.layers[i].mlp.down_proj.weight.detach().float().cpu() for i in layers}
        torch.save(diag, out_dir / f"diag_u{update}.pt")
        params = [p for _, p in trainable_named(model)]
        if refs is None:
            refs = load_reference(cfg, ref_names, "cpu")
        info["displacement_l2"] = displacement(params, refs)
        info["export_gb"] = size_gb(d)
    else:
        raw = L.adapter_state(wrapped, projected=False)
        save_file(raw, out_dir / f"adapter_raw_u{update}.safetensors")
        info["adapter_raw_gb"] = size_gb(out_dir / f"adapter_raw_u{update}.safetensors")
        if cfg["method"] == "oplora":
            proj = L.adapter_state(wrapped, projected=True)
            save_file(proj, out_dir / f"adapter_projected_u{update}.safetensors")
            keys = [f"{i}.down_proj" for i in layers]
            save_file(L.basis_state(wrapped, keys), out_dir / "bases_diag.safetensors")
            info["adapter_projected_gb"] = size_gb(out_dir / f"adapter_projected_u{update}.safetensors")
            viol = []
            for m in wrapped.values():
                a, b = m.factors()
                viol.append(max(float((m.U.T @ b).abs().max()), float((a @ m.V).abs().max())))
            info["projection_max_violation"] = max(viol)
        save_json(out_dir / "adapter_meta.json", {
            "r": int(cfg["lora"]["r"]), "alpha": float(cfg["lora"]["alpha"]),
            "scaling": float(cfg["lora"]["alpha"]) / int(cfg["lora"]["r"]),
            "targets": cfg["lora"]["targets"], "method": cfg["method"],
            "protect_k": int(cfg["lora"]["protect_k"]) if cfg["method"] == "oplora" else 0,
        })
    info["export_s"] = t()
    append_jsonl(res_dir / "exports.jsonl", info)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name")
    ap.add_argument("--stop_after", type=int)
    ap.add_argument("--checkpoint_updates")
    ap.add_argument("--export_updates")
    ap.add_argument("--resume_from")
    ap.add_argument("--memory_probe", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.name:
        cfg["name"] = args.name
    tcfg = cfg["train"]
    if args.checkpoint_updates is not None:
        tcfg["checkpoint_updates"] = [int(x) for x in args.checkpoint_updates.split(",") if x]
    if args.export_updates is not None:
        tcfg["export_updates"] = [int(x) for x in args.export_updates.split(",") if x]
    total = int(tcfg["updates"])
    stop = min(args.stop_after or total, total)

    out_dir = ensure_dir(f"{cfg['out_root']}/{cfg['name']}")
    res_dir = ensure_dir(f"{cfg['results_root']}/{cfg['name']}")
    ckpt_dir = ensure_dir(out_dir / "ckpt")
    log_path = res_dir / "train_log.jsonl"
    summary_path = res_dir / "run_summary.json"
    if not args.memory_probe and summary_path.exists():
        prev = read_jsonl(log_path)
        if prev and prev[-1]["update"] >= stop:
            print(f"{cfg['name']}: already trained to update {prev[-1]['update']}, skipping")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["seed"]))
    wall = Timer()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    data = load_cached(cfg)
    micro, accum = int(tcfg["micro_batch"]), int(tcfg["grad_accum"])
    assert len(data["target"]) >= total * micro * accum
    batches = UpdateBatches(data, micro, accum, tok.pad_token_id)

    model, wrapped, extra = build_model(cfg, device)
    named = trainable_named(model)
    names, params = [n for n, _ in named], [p for _, p in named]
    n_trainable = sum(p.numel() for p in params)
    lam = float(cfg.get("lambda_l2sp", 0.0))
    refs = None
    if lam > 0:
        t = Timer()
        refs = [p.detach().clone().to(cfg.get("ref_device", "cuda")) for p in params]
        extra["ref_bytes"] = sum(r.numel() * r.element_size() for r in refs)
        extra["ref_copy_s"] = t()
        extra["ref_device"] = cfg.get("ref_device", "cuda")

    opt = torch.optim.AdamW(param_groups(model, float(tcfg["weight_decay"])), lr=float(cfg["lr"]),
                            betas=tuple(tcfg["betas"]), eps=float(tcfg["eps"]))
    sched = get_cosine_schedule_with_warmup(opt, int(tcfg["warmup"]), total)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    if args.memory_probe:
        lengths = [len(p) + len(t) for p, t in zip(data["prompt"], data["target"])]
        longest = sorted(range(len(lengths)), key=lambda i: -lengths[i])[:micro]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = Timer()
        b = batches.collate(longest, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            ls, n = response_loss(model, b)
        scaler.scale(ls / n).backward()
        scaler.unscale_(opt)
        if lam > 0:
            l2sp_grad(params, refs, lam)
        torch.nn.utils.clip_grad_norm_(params, float(tcfg["max_grad_norm"]))
        scaler.step(opt)
        scaler.update()
        out = {"name": cfg["name"], "longest_tokens": [lengths[i] for i in longest], "seconds": t(),
               "trainable_params": n_trainable, **gpu_peak(), **extra}
        save_json(res_dir / "memory_probe.json", out)
        print(out)
        return

    start, elapsed = 0, {"train_s": 0.0, "checkpoint_s": 0.0, "export_s": 0.0}
    resume = repo_path(args.resume_from) if args.resume_from else latest_checkpoint(ckpt_dir)
    if resume is not None:
        st = torch.load(resume, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for n, p in named:
                p.copy_(st["trainable"][n].to(p.device))
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        start, elapsed = int(st["update"]), st["elapsed"]
        write_jsonl(log_path, [r for r in read_jsonl(log_path) if r["update"] <= start])
        print(f"resumed {cfg['name']} from {resume} at update {start}")
        if resume.parent != ckpt_dir:
            shutil.copy2(resume, ckpt_dir / resume.name)
    else:
        for f in ("train_log.jsonl", "exports.jsonl", "checkpoints.jsonl", "sessions.jsonl"):
            (res_dir / f).unlink(missing_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    skipped_total = 0
    for u in range(start, stop):
        t = Timer()
        n_tok = batches.n_target_tokens(u)
        lr = sched.get_last_lr()[0]
        sft_sum, seq_tokens = 0.0, 0
        for idx in batches.indices(u):
            b = batches.collate(idx, device)
            seq_tokens += int(b["attention_mask"].sum())
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                ls, _ = response_loss(model, b)
            scaler.scale(ls / n_tok).backward()
            sft_sum += float(ls.detach())
        scaler.unscale_(opt)
        omega = l2sp_grad(params, refs, lam) if lam > 0 else 0.0
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, float(tcfg["max_grad_norm"])))
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        skipped_total += int(skipped)
        sched.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = t()
        elapsed["train_s"] += dt
        sft = sft_sum / n_tok
        rec = {"update": u + 1, "examples_seen": (u + 1) * micro * accum, "sft_loss": sft, "lr": lr,
               "grad_norm": gnorm if math.isfinite(gnorm) else None, "loss_scale": scale_before,
               "skipped": bool(skipped), "target_tokens": n_tok, "tokens": seq_tokens, "seconds": dt}
        if lam > 0:
            rec.update({"omega": omega, "lambda_omega": lam * omega, "total_loss": sft + lam * omega})
        append_jsonl(log_path, rec)
        if (u + 1) % 10 == 0 or u == start:
            print(f"[{cfg['name']}] update {u + 1}/{total} loss {sft:.4f} lr {lr:.2e} gn {gnorm:.3f} "
                  f"{dt:.1f}s/upd skipped {skipped_total}", flush=True)

        if u + 1 in tcfg["export_updates"]:
            info = export(cfg, model, wrapped, tok, u + 1, out_dir, res_dir, refs, names)
            elapsed["export_s"] += info["export_s"]
        if u + 1 in tcfg["checkpoint_updates"]:
            t = Timer()
            path = ckpt_dir / f"update_{u + 1}.pt"
            save_recovery(path, model, opt, sched, scaler, u + 1, (u + 1) * micro * accum, elapsed)
            append_jsonl(res_dir / "checkpoints.jsonl", {"update": u + 1, "gb": size_gb(path)})
            for old in sorted(ckpt_dir.glob("update_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[: -int(tcfg["keep_recovery"])]:
                old.unlink()
            elapsed["checkpoint_s"] += t()

    log = read_jsonl(log_path)
    summary = {
        "name": cfg["name"], "method": cfg["method"], "lr": float(cfg["lr"]), "lambda_l2sp": lam,
        "updates_done": log[-1]["update"] if log else 0, "updates_total": total,
        "trainable_params": n_trainable, "skipped_updates": [r["update"] for r in log if r["skipped"]],
        "recovery_checkpoints": read_jsonl(res_dir / "checkpoints.jsonl"), "exports": read_jsonl(res_dir / "exports.jsonl"),
        "elapsed": elapsed, "wall_this_session_s": wall(), **gpu_peak(), **extra, "env": env_info(),
        "config": cfg,
    }
    append_jsonl(res_dir / "sessions.jsonl", {"start_update": start, "end_update": summary["updates_done"],
                                               "wall_s": wall(), **gpu_peak()})
    summary["sessions"] = read_jsonl(res_dir / "sessions.jsonl")
    save_json(summary_path, summary)
    print(f"done {cfg['name']}: {summary['updates_done']} updates, peak {summary['peak_allocated_gb']:.2f} GB")


if __name__ == "__main__":
    main()

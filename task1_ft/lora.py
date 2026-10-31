from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, generator: torch.Generator, protect_k: int = 0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.protect_k = protect_k
        dout, din = base.out_features, base.in_features
        bound = 1.0 / math.sqrt(din)
        a = (torch.rand(r, din, generator=generator, dtype=torch.float32) * 2 - 1) * bound
        self.lora_A = nn.Parameter(a.to(base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(dout, r, dtype=torch.float32, device=base.weight.device))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if protect_k:
            self.register_buffer("U", torch.empty(dout, protect_k, device=base.weight.device), persistent=False)
            self.register_buffer("V", torch.empty(din, protect_k, device=base.weight.device), persistent=False)

    @torch.no_grad()
    def set_bases(self) -> None:
        w = self.base.weight.detach().float()
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        self.U.copy_(u[:, : self.protect_k])
        self.V.copy_(vh[: self.protect_k].T)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.lora_A, self.lora_B
        if self.protect_k:
            with torch.autocast(device_type=a.device.type, enabled=False):
                b = b - self.U @ (self.U.T @ b)
                a = a - (a @ self.V) @ self.V.T
        return a, b

    def delta_weight(self) -> torch.Tensor:
        a, b = self.factors()
        with torch.autocast(device_type=a.device.type, enabled=False):
            return self.scaling * (b.float() @ a.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.factors()
        if not torch.is_autocast_enabled(x.device.type):
            a, b = a.to(x.dtype), b.to(x.dtype)
        return self.base(x) + F.linear(F.linear(self.dropout(x), a), b) * self.scaling


def target_modules(model, targets: list[str]):
    for i, layer in enumerate(model.model.layers):
        for name in targets:
            parent = layer.self_attn if name in ATTN else layer.mlp
            yield i, name, parent


def inject(model, lcfg: dict, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    k = int(lcfg.get("protect_k", 0))
    wrapped = {}
    for i, name, parent in target_modules(model, lcfg["targets"]):
        mod = LoRALinear(getattr(parent, name), int(lcfg["r"]), float(lcfg["alpha"]), float(lcfg["dropout"]), gen, k)
        setattr(parent, name, mod)
        wrapped[f"{i}.{name}"] = mod
    return wrapped


def compute_bases(wrapped: dict) -> None:
    for mod in wrapped.values():
        mod.set_bases()


def adapter_state(wrapped: dict, projected: bool) -> dict:
    out = {}
    for key, mod in wrapped.items():
        a, b = mod.factors() if projected else (mod.lora_A, mod.lora_B)
        out[f"{key}.A"] = a.detach().float().cpu().contiguous()
        out[f"{key}.B"] = b.detach().float().cpu().contiguous()
    return out


def basis_state(wrapped: dict, keys: list[str] | None = None) -> dict:
    out = {}
    for key, mod in wrapped.items():
        if mod.protect_k and (keys is None or key in keys):
            out[f"{key}.U"] = mod.U.detach().cpu().contiguous()
            out[f"{key}.V"] = mod.V.detach().cpu().contiguous()
    return out


def basis_bytes(wrapped: dict) -> int:
    return sum(m.U.numel() * m.U.element_size() + m.V.numel() * m.V.element_size() for m in wrapped.values() if m.protect_k)


@torch.no_grad()
def merge_into(model, factors: dict, scaling: float) -> None:
    for key in {k.rsplit(".", 1)[0] for k in factors}:
        i, name = key.split(".")
        layer = model.model.layers[int(i)]
        lin = getattr(layer.self_attn if name in ATTN else layer.mlp, name)
        a = factors[f"{key}.A"].to(lin.weight.device, torch.float32)
        b = factors[f"{key}.B"].to(lin.weight.device, torch.float32)
        lin.weight.copy_((lin.weight.float() + scaling * (b @ a)).to(lin.weight.dtype))

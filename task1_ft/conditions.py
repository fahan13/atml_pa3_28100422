from __future__ import annotations

from safetensors.torch import load_file

from common.utils import load_json, repo_path


def condition(cfg: dict, name: str) -> dict:
    return cfg["eval"]["conditions"][name]


def run_dir(cfg: dict, name: str):
    return repo_path(cfg["out_root"]) / condition(cfg, name)["run"]


def model_path(cfg: dict, name: str) -> str:
    c = condition(cfg, name)
    if c["kind"] == "base":
        return cfg["model"]
    if c["kind"] == "full":
        return str(run_dir(cfg, name) / f"export_u{c['update']}")
    return str(repo_path(cfg["eval"]["merged_root"]) / name)


def adapter_factors(cfg: dict, name: str) -> tuple[dict, float, str]:
    c = condition(cfg, name)
    d = run_dir(cfg, name)
    meta = load_json(d / "adapter_meta.json")
    kind = "projected" if meta["method"] == "oplora" else "raw"
    return load_file(str(d / f"adapter_{kind}_u{c['update']}.safetensors")), float(meta["scaling"]), kind

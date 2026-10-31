from __future__ import annotations

import argparse
import importlib.metadata as md
import shutil
import subprocess

import torch

from common.utils import env_info, save_json

PACKAGES = ["transformers", "accelerate", "datasets", "safetensors", "numpy", "pandas", "lm_eval", "math-verify", "pycocotools"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/env/environment.json")
    args = ap.parse_args()
    info = env_info()
    info["packages"] = {}
    for p in PACKAGES:
        try:
            info["packages"][p] = md.version(p)
        except md.PackageNotFoundError:
            info["packages"][p] = None
    info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    info["gpu_mem_gb"] = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(torch.cuda.device_count())]
    du = shutil.disk_usage(".")
    info["disk_free_gb"] = du.free / 1e9
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        info["nvidia_smi"] = None
    save_json(args.out, info)
    print(info)


if __name__ == "__main__":
    main()

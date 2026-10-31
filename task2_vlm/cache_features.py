from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common.utils import Timer, env_info, gpu_peak, load_config, repo_path, save_json

ID_RE = re.compile(r"(?:COCO_(train|val)2014_)?(\d{12})\.jpg$")


def index_images(roots: list[Path]) -> dict:
    found = {}
    for r in roots:
        for p in r.rglob("*.jpg"):
            m = ID_RE.search(p.name)
            if not m:
                continue
            split, img = m.group(1), int(m.group(2))
            key = (split, img)
            if split is None:
                key = (None, img)
            found.setdefault(key, p)
    return found


def resolve(files: dict, img: int, split: str):
    return files.get((split, img)) or files.get((None, img))


class Images(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.proc = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            px = self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return px


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2/base.yaml")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    from transformers import AutoImageProcessor, SiglipVisionModel

    out_dir = repo_path(cfg["paths"]["features"])
    feat_path, index_path = out_dir / "siglip_196.npy", out_dir / "index.json"
    if index_path.exists():
        print("features exist, skipping")
        return
    built = json.loads(repo_path(cfg["paths"]["built"]).read_text(encoding="utf-8"))
    wanted = sorted((int(k), v[0]) for k, v in built["image_files"].items())
    t_index = Timer()
    files = index_images([Path(r) for r in args.roots])
    index_s = t_index()
    paths, missing = [], []
    for img, split in wanted:
        p = resolve(files, img, split)
        (paths.append(p) if p else missing.append(img))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[:5]}")

    proc = AutoImageProcessor.from_pretrained(cfg["vision"])
    model = SiglipVisionModel.from_pretrained(cfg["vision"], dtype=torch.float16).to("cuda").eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float16, shape=(len(paths), 196, 768))
    loader = DataLoader(Images(paths, proc), batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    torch.cuda.reset_peak_memory_stats()
    t = Timer()
    row, ln_check = 0, None
    with torch.no_grad():
        for px in loader:
            px = px.to("cuda", torch.float16)
            out = model(pixel_values=px, output_hidden_states=ln_check is None)
            h = out.last_hidden_state
            if ln_check is None:
                pre = out.hidden_states[-1]
                once = model.vision_model.post_layernorm(pre)
                twice = model.vision_model.post_layernorm(once)
                ln_check = {"max_abs_diff_last_vs_LN_once": float((h - once).abs().max()),
                            "max_abs_diff_last_vs_LN_twice": float((h - twice).abs().max()),
                            "shape": list(h.shape)}
            assert h.shape[1:] == (196, 768)
            feats[row:row + len(h)] = h.cpu().numpy().astype(np.float16)
            row += len(h)
    feats.flush()
    seconds = t()
    index = {str(img): i for i, (img, _) in enumerate(wanted)}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    proc_cfg = {k: getattr(proc, k, None) for k in ("size", "resample", "image_mean", "image_std", "do_resize", "do_normalize")}
    save_json(repo_path(cfg["results"]) / "feature_cache.json", {
        "n_images": len(paths), "shape": [len(paths), 196, 768], "dtype": "float16", "file_gb": feat_path.stat().st_size / 1e9,
        "encode_seconds": seconds, "images_per_second": len(paths) / seconds, "file_index_seconds": index_s,
        "layernorm_check": ln_check, "processor": {k: str(v) for k, v in proc_cfg.items()},
        "feature": "SiglipVisionModel.last_hidden_state (post_layernorm applied once inside the model), 14x14 row-major patch order",
        **gpu_peak(), "env": env_info(),
    })
    print(f"cached {len(paths)} images in {seconds:.0f}s, LN check {ln_check}")


if __name__ == "__main__":
    main()

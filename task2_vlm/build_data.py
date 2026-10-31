from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common.utils import load_config, repo_path, save_json
from task2_vlm.vqa_norm import normalize

PLURAL = {"person": "people", "sheep": "sheep", "skis": "skis", "scissors": "scissors", "mouse": "mice", "knife": "knives"}
NO_ARTICLE = {"skis", "scissors"}
IMAGE_DIRS = {"train2014", "val2014", "test2014", "train2017", "val2017", "test2017", "images"}


def plural(name: str) -> str:
    if name in PLURAL:
        return PLURAL[name]
    if name.endswith(("s", "sh", "ch", "x")):
        return name + "es"
    if name.endswith("y") and name[-2:-1] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def with_article(name: str) -> str:
    if name in NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def find_file(name: str, roots: list[Path]) -> Path | None:
    for r in roots:
        if not r.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = sorted(d for d in dirnames if d not in IMAGE_DIRS)
            if name in filenames:
                return Path(dirpath) / name
    return None


def fetch(cfg: dict, names: list[str], url: str, roots: list[Path]) -> dict[str, Path]:
    dest = repo_path(cfg["paths"]["downloads"])
    found = {n: find_file(n, [dest] + roots) for n in names}
    if all(found.values()):
        return found
    dest.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(url, timeout=600).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            if Path(member).name in names:
                (dest / Path(member).name).write_bytes(z.read(member))
    found = {n: find_file(n, [dest] + roots) for n in names}
    missing = [n for n, p in found.items() if p is None]
    if missing:
        raise FileNotFoundError(f"missing after download from {url}: {missing}")
    return found


def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def coco_split(files: dict, split: str) -> dict:
    caps = load_json(files[f"captions_{split}2014.json"])
    inst = load_json(files[f"instances_{split}2014.json"])
    images = {im["id"]: im["file_name"] for im in caps["images"]}
    captions = defaultdict(list)
    for a in caps["annotations"]:
        captions[a["image_id"]].append((a["id"], a["caption"].strip()))
    objects = defaultdict(list)
    for a in inst["annotations"]:
        objects[a["image_id"]].append((a["category_id"], float(a["area"]), int(a["iscrowd"])))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    return {"images": images, "captions": {k: sorted(v) for k, v in captions.items()}, "objects": objects, "cats": cats}


def vqa_split(files: dict, split: str) -> dict:
    qs = load_json(files[f"v2_OpenEnded_mscoco_{split}2014_questions.json"])["questions"]
    anns = load_json(files[f"v2_mscoco_{split}2014_annotations.json"])["annotations"]
    pairs = load_json(files[f"v2_mscoco_{split}2014_complementary_pairs.json"])
    q = {x["question_id"]: x for x in qs}
    a = {x["question_id"]: x for x in anns}
    return {"questions": q, "annotations": a, "pairs": pairs}


def consensus(answers: list[dict]) -> tuple[str, int, str]:
    raw = [x["answer"] for x in answers]
    norm = [normalize(r) for r in raw]
    canon, n = Counter(norm).most_common(1)[0]
    target = Counter(r for r, m in zip(raw, norm) if m == canon).most_common(1)[0][0]
    return canon, n, target


def vqa_candidates(v: dict, min_consensus: int) -> dict:
    out = defaultdict(list)
    for q1, q2 in v["pairs"]:
        if q1 not in v["annotations"] or q2 not in v["annotations"]:
            continue
        a1, a2 = v["annotations"][q1], v["annotations"][q2]
        x1, x2 = v["questions"][q1], v["questions"][q2]
        if x1["image_id"] == x2["image_id"] or a1["answer_type"] != a2["answer_type"]:
            continue
        c1, n1, t1 = consensus(a1["answers"])
        c2, n2, t2 = consensus(a2["answers"])
        if n1 < min_consensus or n2 < min_consensus or c1 == c2:
            continue
        items = []
        for x, a, c, t in ((x1, a1, c1, t1), (x2, a2, c2, t2)):
            items.append({"qid": int(x["question_id"]), "image_id": int(x["image_id"]), "answer": t, "canonical": c,
                          "answers": [y["answer"] for y in a["answers"]]})
        out[a1["answer_type"]].append({"type": a1["answer_type"], "question": x1["question"], "items": items})
    return out


def take(rng, cands: list, n: int, used: set | None) -> list:
    out = []
    for i in rng.permutation(len(cands)):
        p = cands[int(i)]
        imgs = {it["image_id"] for it in p["items"]}
        if used is not None and (imgs & used or len(imgs) < 2):
            continue
        out.append(p)
        if used is not None:
            used |= imgs
        if len(out) == n:
            return out
    raise RuntimeError(f"only {len(out)} of {n} candidates available")


def object_index(coco: dict, image_ids: list[int], min_area: float, count_range: tuple[int, int]):
    pos, neg, counts = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for img in image_ids:
        objs = coco["objects"].get(img, [])
        present = defaultdict(list)
        for c, area, crowd in objs:
            present[c].append((area, crowd))
        for c in coco["cats"]:
            if c not in present:
                neg[c].append(img)
                continue
            inst = present[c]
            if any(a >= min_area and not cr for a, cr in inst):
                pos[c].append(img)
            if all(a >= min_area and not cr for a, cr in inst) and count_range[0] <= len(inst) <= count_range[1]:
                counts[c][len(inst)].append(img)
    return pos, neg, counts


def grounding_pairs(rng, coco: dict, image_ids: list[int], n_presence: int, n_counting: int, dcfg: dict,
                    used: set | None, suite: str) -> list:
    pos, neg, counts = object_index(coco, sorted(image_ids), float(dcfg["min_object_area"]), tuple(dcfg["count_range"]))
    out, seen = [], set()
    pres_cats = sorted(c for c in coco["cats"] if pos[c] and neg[c])
    count_cats = sorted(c for c in coco["cats"] if len(counts[c]) >= 2)

    def ok(imgs):
        return used is None or not (set(imgs) & used)

    tries = 0
    while sum(p["type"] == "presence" for p in out) < n_presence:
        tries += 1
        if tries > 200000:
            raise RuntimeError("cannot build presence pairs")
        c = pres_cats[int(rng.integers(len(pres_cats)))]
        p = pos[c][int(rng.integers(len(pos[c])))]
        q = neg[c][int(rng.integers(len(neg[c])))]
        if (c, p, q) in seen or not ok([p, q]) or p == q:
            continue
        seen.add((c, p, q))
        if used is not None:
            used |= {p, q}
        name = coco["cats"][c]
        out.append({"type": "presence", "category": name, "question": f"Does this image contain {with_article(name)}?",
                    "items": [{"image_id": p, "answer": "yes", "canonical": "yes"}, {"image_id": q, "answer": "no", "canonical": "no"}]})
    tries = 0
    while sum(p["type"] == "counting" for p in out) < n_counting:
        tries += 1
        if tries > 200000:
            raise RuntimeError("cannot build counting pairs")
        c = count_cats[int(rng.integers(len(count_cats)))]
        k1, k2 = rng.choice(sorted(counts[c]), size=2, replace=False)
        p = counts[c][k1][int(rng.integers(len(counts[c][k1])))]
        q = counts[c][k2][int(rng.integers(len(counts[c][k2])))]
        if (c, p, q) in seen or not ok([p, q]) or p == q:
            continue
        seen.add((c, p, q))
        if used is not None:
            used |= {p, q}
        name = coco["cats"][c]
        out.append({"type": "counting", "category": name, "question": f"How many {plural(name)} can you see?",
                    "items": [{"image_id": p, "answer": str(int(k1)), "canonical": str(int(k1))},
                              {"image_id": q, "answer": str(int(k2)), "canonical": str(int(k2))}]})
    for i, p in enumerate(out):
        p["pair_id"] = f"{suite}-{i}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2/base.yaml")
    ap.add_argument("--roots", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_path = repo_path(cfg["paths"]["built"])
    if out_path.exists():
        print("built data exists, skipping")
        return
    roots = [Path(r) for r in args.roots] + [repo_path(r) for r in cfg["paths"].get("input_roots", [])]
    d = cfg["data"]
    urls = cfg["urls"]
    coco_names = [f"{k}_{s}2014.json" for k in ("captions", "instances") for s in ("train", "val")]
    files = fetch(cfg, coco_names, urls["coco_annotations"], roots)
    for s, cap in (("train", "Train"), ("val", "Val")):
        files.update(fetch(cfg, [f"v2_OpenEnded_mscoco_{s}2014_questions.json"], urls[f"vqa_questions_{s}"], roots))
        files.update(fetch(cfg, [f"v2_mscoco_{s}2014_annotations.json"], urls[f"vqa_annotations_{s}"], roots))
        files.update(fetch(cfg, [f"v2_mscoco_{s}2014_complementary_pairs.json"], urls[f"vqa_pairs_{s}"], roots))
    sha = {n: hashlib.sha1(p.read_bytes()).hexdigest() for n, p in files.items()}

    rng = np.random.default_rng(int(cfg["seed"]))
    coco_tr, coco_va = coco_split(files, "train"), coco_split(files, "val")
    vqa_tr, vqa_va = vqa_split(files, "train"), vqa_split(files, "val")

    train_imgs = sorted(i for i, c in coco_tr["captions"].items() if len(c) >= int(d["captions_per_train_image"]))
    cap_imgs = [train_imgs[int(i)] for i in rng.permutation(len(train_imgs))[: int(d["caption_train_images"])]]
    captions_train = [{"image_id": i, "caption_id": cid, "caption": cap}
                      for i in cap_imgs for cid, cap in coco_tr["captions"][i][: int(d["captions_per_train_image"])]]

    tr_cands = vqa_candidates(vqa_tr, int(d["min_consensus"]))
    va_cands = vqa_candidates(vqa_va, int(d["min_consensus"]))
    vqa_train = []
    for t, n in d["vqa_train"].items():
        vqa_train += take(rng, tr_cands[t], int(n), None)
    for i, p in enumerate(vqa_train):
        p["pair_id"] = f"vqa_train-{i}"

    pool_imgs = sorted(set(cap_imgs) | {it["image_id"] for p in vqa_train for it in p["items"]})
    pools = grounding_pairs(rng, coco_tr, pool_imgs, int(d["presence_pool_pairs"]), int(d["counting_pool_pairs"]), d, None, "pool")

    used: set[int] = set()
    vqa_test, vqa_dev = [], []
    for t, n in d["vqa_test"].items():
        vqa_test += take(rng, va_cands[t], int(n), used)
    for t, n in d["vqa_dev"].items():
        vqa_dev += take(rng, va_cands[t], int(n), used)
    for name, lst in (("vqa_test", vqa_test), ("vqa_dev", vqa_dev)):
        for i, p in enumerate(lst):
            p["pair_id"] = f"{name}-{i}"
    val_imgs = sorted(coco_va["images"])
    g_test = grounding_pairs(rng, coco_va, val_imgs, int(d["grounding_test"]["presence"]), int(d["grounding_test"]["counting"]), d, used, "grounding_test")
    g_dev = grounding_pairs(rng, coco_va, val_imgs, int(d["grounding_dev"]["presence"]), int(d["grounding_dev"]["counting"]), d, used, "grounding_dev")
    free = [i for i in (val_imgs[int(k)] for k in rng.permutation(len(val_imgs))) if i not in used and coco_va["captions"].get(i)]
    cap_dev_imgs = free[: int(d["caption_dev_images"])]
    used |= set(cap_dev_imgs)
    captions_dev = [{"image_id": i, "caption_id": coco_va["captions"][i][0][0], "caption": coco_va["captions"][i][0][1]} for i in cap_dev_imgs]

    def imgs(pairs):
        return {it["image_id"] for p in pairs for it in p["items"]}

    train_set = set(cap_imgs) | imgs(vqa_train) | imgs(pools)
    dev_set = imgs(g_dev) | imgs(vqa_dev) | set(cap_dev_imgs)
    test_set = imgs(g_test) | imgs(vqa_test)
    files_map = {**{i: ("train", coco_tr["images"][i]) for i in train_set}, **{i: ("val", coco_va["images"][i]) for i in dev_set | test_set}}
    built = {
        "captions_train": captions_train, "captions_dev": captions_dev,
        "presence_pool": [p for p in pools if p["type"] == "presence"], "counting_pool": [p for p in pools if p["type"] == "counting"],
        "vqa_train": vqa_train, "vqa_dev": vqa_dev, "vqa_test": vqa_test, "grounding_dev": g_dev, "grounding_test": g_test,
        "image_files": {str(k): v for k, v in sorted(files_map.items())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(built), encoding="utf-8")

    checks = {
        "train_dev_overlap": len(train_set & dev_set), "train_test_overlap": len(train_set & test_set),
        "dev_test_overlap": len(dev_set & test_set),
        "eval_images_unique_within_split": {"dev": len(dev_set) == 2 * (len(g_dev) + len(vqa_dev)) + len(cap_dev_imgs),
                                            "test": len(test_set) == 2 * (len(g_test) + len(vqa_test))},
        "train_images_all_train2014": all(files_map[i][0] == "train" for i in train_set),
        "eval_images_all_val2014": all(files_map[i][0] == "val" for i in dev_set | test_set),
    }
    summary = {
        "counts": {k: len(v) for k, v in built.items() if isinstance(v, list)},
        "vqa_types": {k: dict(Counter(p["type"] for p in built[k])) for k in ("vqa_train", "vqa_dev", "vqa_test")},
        "grounding_types": {k: dict(Counter(p["type"] for p in built[k])) for k in ("presence_pool", "counting_pool", "grounding_dev", "grounding_test")},
        "vqa_candidates": {"train": {k: len(v) for k, v in tr_cands.items()}, "val": {k: len(v) for k, v in va_cands.items()}},
        "images": {"train": len(train_set), "dev": len(dev_set), "test": len(test_set), "caption_train": len(cap_imgs)},
        "count_answer_distribution": {k: dict(Counter(it["answer"] for p in built[k] if p["type"] == "counting" for it in p["items"]))
                                      for k in ("counting_pool", "grounding_dev", "grounding_test")},
        "pool_categories": dict(Counter(p["category"] for p in pools).most_common()),
        "checks": checks, "source_sha1": sha,
        "rules": {"presence_positive": f"at least one non-crowd instance with area >= {d['min_object_area']} px",
                  "presence_negative": "no annotation of the category at all",
                  "counting": f"all instances of the category non-crowd with area >= {d['min_object_area']} px; count in {d['count_range']}",
                  "vqa": f"complementary pair, both questions >= {d['min_consensus']}/10 annotators agree after normalization, different consensus answers, same answer_type"},
    }
    res = repo_path(cfg["results"]) / "data"
    save_json(res / "data_summary.json", summary)
    save_json(res / "selection_ids.json", {
        "caption_train_images": cap_imgs, "caption_train_caption_ids": [c["caption_id"] for c in captions_train],
        "caption_dev": [[c["image_id"], c["caption_id"]] for c in captions_dev],
        **{k: [[p["pair_id"], p["question"], [it["image_id"] for it in p["items"]], [it.get("qid") for it in p["items"]]] for p in built[k]]
           for k in ("vqa_train", "vqa_dev", "vqa_test", "grounding_dev", "grounding_test", "presence_pool", "counting_pool")},
    })
    print(json.dumps({k: summary[k] for k in ("counts", "images", "checks")}, indent=1))


if __name__ == "__main__":
    main()

"""Extract the qualitative examples for Tasks 2 and 3 with fixed selection rules (CPU only).

    python report/make_examples.py

Writes report/examples/t2_pairs.tex and report/examples/t3_examples.md. Every example is chosen by the rule
printed next to it, never by looking at which examples are interesting.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task3_ttc.select import vote  # noqa: E402

R = ROOT / "results"
OUT = ROOT / "report" / "examples"
NO_ANSWER = -1


def jl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def tex(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def t2_pairs():
    """Rule: the first pair (lowest pair index in the saved test order) of every question type in each final-test suite."""
    rows = jl(R / "task2" / "vlm196" / "final_eval_stage3_examples.jsonl")
    by = {}
    for r in rows:
        by.setdefault((r["suite"], r["pair_id"]), {}).setdefault(r["condition"], {})[r["position"]] = r
    chosen = []
    for suite in ("grounding_test", "vqa_test"):
        seen = set()
        pids = sorted({p for s, p in by if s == suite}, key=lambda p: int(p.rsplit("-", 1)[1]))
        for p in pids:
            t = by[(suite, p)]["correct"][0]["type"]
            if t not in seen:
                seen.add(t)
                chosen.append((suite, p, t))
    out = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l l l l llll@{}}", r"\toprule",
           r"Type & Question & Image & Ref. & Correct & Swapped & Mismatched & No image \\", r"\midrule"]
    for suite, p, t in chosen:
        e = by[(suite, p)]
        q = e["correct"][0]["question"]
        for pos in (0, 1):
            c = e["correct"][pos]
            preds = [e[k][pos]["prediction"] or "(empty)" for k in ("correct", "swap", "mismatched", "none")]
            mark = lambda k: (r"$^\checkmark$" if e[k][pos]["score"] >= 0.999 else "") + (r"$^\dagger$" if k == "swap" and e[k][pos]["follows_partner"] else "")
            cells = " & ".join(f"{tex(pr)}{mark(k)}" for pr, k in zip(preds, ("correct", "swap", "mismatched", "none")))
            out.append(f"{tex(t) if pos == 0 else ''} & {tex(q) if pos == 0 else ''} & {c['image_id']} & {tex(c['canonical'])} & {cells} \\\\")
        out.append(r"\addlinespace[2pt]")
    out[-1:] = [r"\midrule", r"\multicolumn{8}{@{}l}{$^\checkmark$ full credit against the original reference; "
                r"$^\dagger$ swapped-image answer matches the partner image's reference.} \\", r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t2_pairs.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    js_out = [{"suite": s, "pair_id": p, "type": t, "image_ids": [by[(s, p)]["correct"][i]["image_id"] for i in (0, 1)]}
              for s, p, t in chosen]
    (OUT / "t2_pairs.json").write_text(json.dumps(js_out, indent=1), encoding="utf-8")
    print("wrote report/examples/t2_pairs.tex", len(chosen), "pairs")


def load_pool(ds):
    rows = list(csv.DictReader(open(R / "task3" / ds / "rollouts.csv", encoding="utf-8")))
    pool = {}
    for r in rows:
        pool.setdefault(r["sample_id"], []).append(r)
    for q in pool:
        pool[q].sort(key=lambda r: int(r["rollout_id"]))
    gold = {r["sample_id"]: r["gold"] for r in csv.DictReader(open(R / "task3" / ds / "questions.csv", encoding="utf-8"))}
    orders = np.load(R / "task3" / ds / "orders.npy")
    qindex = {q: i for i, q in enumerate(js(R / "task3" / ds / "orders_index.json"))}
    return pool, gold, orders, qindex


def key_of(pool, q, g):
    if g == NO_ANSWER:
        return "NO_ANSWER"
    return next(r["key"] for r in pool[q] if int(r["group"]) == g)


def correct_of(pool, q, g):
    return g != NO_ANSWER and next(r["correct"] == "True" for r in pool[q] if int(r["group"]) == g)


def dist(pool, q, idx):
    groups = np.array([int(pool[q][i]["group"]) for i in idx])
    _, win, tot = vote(groups, np.ones(len(groups)))
    parts = sorted(tot.items(), key=lambda kv: -kv[1])
    return win, ", ".join(f"`{key_of(pool, q, g)}`{'*' if correct_of(pool, q, g) else ''} ×{int(v)}" for g, v in parts)


def t3_examples():
    md = ["# Task 3 qualitative examples", "",
          "Answers marked * are correct. Counts are answer-group sizes among the revealed rollouts.", ""]
    splits = js(R / "task3" / "splits.json")
    pools = {ds: load_pool(ds) for ds in ("math", "mathvista")}

    md += ["## Selection disagreement (MATH)", "",
           "Rule: the first 3 eval questions (split order) where the rollouts selected by self-certainty + RWC and by self-consistency differ in "
           "correctness at N = 32, order 0.", ""]
    pool, gold, orders, qindex = pools["math"]
    sel = {r["sample_id"]: r for r in csv.DictReader(open(R / "task3" / "math" / "offline_selected_N32.csv", encoding="utf-8"))
           if r["order"] == "0"}
    n = 0
    for q in splits["math"]["eval"]:
        s = sel.get(q)
        if s is None:
            continue
        gs, gr = (int(pool[q][int(s[m])]["group"]) for m in ("sc", "selfcert_rwc"))
        if correct_of(pool, q, gs) != correct_of(pool, q, gr):
            _, d = dist(pool, q, orders[qindex[q], 0][:32])
            md += [f"- `{q}` gold `{gold[q]}`: SC → `{key_of(pool, q, gs)}` ({'correct' if correct_of(pool, q, gs) else 'wrong'}); "
                   f"self-cert. + RWC → `{key_of(pool, q, gr)}` ({'correct' if correct_of(pool, q, gr) else 'wrong'}). Pool: {d}"]
            n += 1
            if n == 3:
                break

    md += ["", "## Selection disagreement (MathVista)", ""]
    for i, t in enumerate(js(R / "task3" / "mathvista" / "trajectories.json")):
        md += [f"Rule: {t['rule']}.", ""] if i == 0 else []
        picks = "; ".join(f"{r['selected_by']} → `{r['answer']}` ({'correct' if r['correct'] else 'wrong'}, T={r['T']}, "
                          f"V={r['V']:.3f}, C={r['C']:.3f}, Z={r['Z']:.3f})" for r in t["rollouts"])
        md += [f"- `{t['sample_id']}`: {picks}"]

    md += ["", "## Revision (live, 50 MATH questions)", "",
           "Rule: every question where the revised answer differs from the draft answer.", ""]
    for r in csv.DictReader(open(R / "task3" / "live" / "live_per_question.csv", encoding="utf-8")):
        if r["answer_changed"] == "True":
            draft = r["initial_answer"] or "none"
            note = " (draft truncated at 1,024 tokens)" if r["initial_trunc"] == "True" else ""
            md += [f"- `{r['sample_id']}` gold `{r['gold']}`: draft `{draft}`{note} "
                   f"({'correct' if r['initial_correct'] == 'True' else 'wrong'}) → revised `{r['refine_answer'] or 'none'}` "
                   f"({'correct' if r['refine_correct'] == 'True' else 'wrong'})"]

    for ds, name in (("math", "MATH"), ("mathvista", "MathVista")):
        pool, gold, orders, qindex = pools[ds]
        ad = js(R / "task3" / ds / "adaptive.json")["families"]["agreement"]
        tau = ad["selected"]["tau"]
        md += ["", f"## Premature stopping ({name}, agreement rule, τ = {tau:g})", "",
               "Rule: the first 3 distinct questions in `adaptive.json` → `incorrect_early_stops.examples_first10` (eval split "
               "order, first listed order of each), i.e. early stops with a wrong answer that fixed N = 32 self-consistency gets right.", ""]
        picked, seen = [], set()
        for ex in ad["incorrect_early_stops"]["examples_first10"]:
            if ex["sample_id"] not in seen and len(picked) < 3:
                seen.add(ex["sample_id"])
                picked.append(ex)
        for ex in picked:
            q, o, k = ex["sample_id"], ex["order"], ex["stopped_at"]
            order = orders[qindex[q], o]
            w_n, d_n = dist(pool, q, order[:k])
            w_32, d_32 = dist(pool, q, order[:32])
            md += [f"- `{q}` order {o}, gold `{gold[q]}`: stopped at n = {k} with `{key_of(pool, q, w_n)}` [{d_n}]; "
                   f"at N = 32 SC gives `{key_of(pool, q, w_32)}` [{d_32}]"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t3_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote report/examples/t3_examples.md")


if __name__ == "__main__":
    t2_pairs()
    t3_examples()

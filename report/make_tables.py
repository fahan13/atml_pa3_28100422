"""Regenerate every report table from saved result files as LaTeX (booktabs) tabulars.

    python report/make_tables.py             # all tables
    python report/make_tables.py t1_benchmarks
    python report/make_tables.py --list

Each table is one function; each reads only files under results/ and writes
report/tables/<name>.tex (a tabular only; caption and float go in the report).
Needs \\usepackage{booktabs} in the preamble.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "tables"

T1 = [("baseline", "Base model"), ("full_ft_mid", "Full FT (u625)"), ("full_ft", "Full FT"),
      ("l2sp", "Full FT + L2-SP"), ("lora4", "LoRA-4"), ("lora16", "LoRA-16"), ("oplora16", "OPLoRA-16")]
RUNS = T1[2:]


def jl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", (OUT / f"{name}.tex").relative_to(ROOT))


def pct(v, nd=1):
    return f"{100 * v:.{nd}f}"


def delta(v, b):
    d = 100 * (v - b)
    return f"\\textsubscript{{{'+' if d >= 0 else '$-$'}{abs(d):.1f}}}"


def t1_benchmarks():
    """Seven conditions x four benchmarks, primary score with pp change from baseline; GSM8K lenient as a check."""
    ev = {c: js(R / "task1" / "eval" / c / "scores.json") for c, _ in T1}
    cols = [("gsm8k", "flexible_em"), ("gsm8k", "lenient_em_supplementary"), ("hellaswag", "acc_norm"),
            ("boolq", "acc"), ("arc_easy", "acc_norm")]
    b = ev["baseline"]
    out = [r"\begin{tabular}{@{}l rr rrr@{}}", r"\toprule",
           r" & \multicolumn{2}{c}{GSM8K (EM)} & HellaSwag & BoolQ & ARC-E \\ \cmidrule(lr){2-3}",
           r"Condition & flexible & lenient$^\dagger$ & acc\_norm & acc & acc\_norm \\", r"\midrule"]
    for c, lab in T1:
        cells = []
        for t, m in cols:
            v = ev[c][t][m]
            cells.append(pct(v) if c == "baseline" else f"{pct(v)}{delta(v, b[t][m])}")
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if c in ("full_ft_mid",):
            out.append(r"\addlinespace[1pt]")
    n = {t: ev["baseline"][t]["n"] for t in ("gsm8k", "hellaswag", "boolq", "arc_easy")}
    sek = {"gsm8k": "flexible_em_stderr", "hellaswag": "acc_norm_stderr", "boolq": "acc_stderr", "arc_easy": "acc_norm_stderr"}
    se = {t: max(100 * ev[c][t][k] for c, _ in T1) for t, k in sek.items()}
    out += [r"\midrule",
            f"$n$ / max s.e. & {n['gsm8k']} / {se['gsm8k']:.1f} & & {n['hellaswag']} / {se['hellaswag']:.1f} & "
            f"{n['boolq']} / {se['boolq']:.1f} & {n['arc_easy']} / {se['arc_easy']:.1f} \\\\",
            r"\bottomrule", r"\end{tabular}"]
    write("t1_benchmarks", out)


def t1_resources():
    """Trainable parameters, training time, peak GPU memory, checkpoint and export sizes, method overheads."""
    out = [r"\begin{tabular}{@{}l r rr r rr l@{}}", r"\toprule",
           r" & Trainable & \multicolumn{2}{c}{Training} & Peak mem. & \multicolumn{2}{c}{Size (GB)} & \\ \cmidrule(lr){3-4} \cmidrule(lr){6-7}",
           r"Run & params (M) & h & s/upd. & (GB) & ckpt & final & Extra cost \\", r"\midrule"]
    for c, lab in RUNS:
        d = R / "task1" / c
        s = js(d / "run_summary.json")
        log = jl(d / "train_log.jsonl")
        spu = sum(r["seconds"] for r in log) / len(log)
        ck = jl(d / "checkpoints.jsonl")[-1]["gb"]
        ex = jl(d / "exports.jsonl")[-1]
        final = ex.get("export_gb", ex.get("adapter_projected_gb", ex.get("adapter_raw_gb")))
        extra = ""
        if c == "l2sp":
            ls = sum(r["l2sp_s"] for r in log) / len(log)
            extra = f"$\\theta_0$ {s['ref_bytes'] / 1e9:.2f} GB (CPU), {ls:.2f} s/upd."
        if c == "oplora16":
            extra = f"bases {s['basis_bytes'] / 1e9:.2f} GB, SVD {s['basis_preprocess_s']:.0f} s"
        out.append(f"{lab} & {s['trainable_params'] / 1e6:.1f} & {s['elapsed']['train_s'] / 3600:.2f} & {spu:.2f} & "
                   f"{s['peak_allocated_gb']:.2f} & {ck:.3f} & {final:.3f} & {extra} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("t1_resources", out)


def t1_annotation():
    """Counts from the 42-response annotation (results/task1/qualitative/annotations.json)."""
    p = R / "task1" / "qualitative" / "annotations.json"
    if not p.exists():
        print("skip t1_annotation: no", p.relative_to(ROOT))
        return
    ann = js(p)
    rows = {}
    for a in ann:
        rows.setdefault(a["condition"], []).append(a)
    resp = {}
    for c, _ in T1:
        for r in jl(R / "task1" / "qualitative" / f"{c}.jsonl"):
            resp[(c, r["id"])] = r
    out = [r"\begin{tabular}{@{}l c c c c c c@{}}", r"\toprule",
           r"Condition & Format fails$^\ast$ & Incoherent & Repetition & Math framing & Inappropriate & Truncated \\", r"\midrule"]
    for c, lab in T1:
        a = rows.get(c, [])

        def n(k, lo=1):
            return sum(1 for x in a if x["labels"].get(k) is not None and x["labels"][k] >= lo)

        fmt_n = sum(1 for x in a if x["format"] != "plain")
        tr = sum(1 for x in a if resp[(c, x["question"] if x["format"] == "plain" else f"{x['question']}_{x['format']}")]["truncated"])
        out.append(f"{lab} & {n('format', 2)}/{fmt_n} & {n('coherence', 2)}/{len(a)} & {n('repetition')}/{len(a)} & "
                   f"{n('math')}/{len(a)} & {n('inappropriate')}/{len(a)} & {tr}/{len(a)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("t1_annotation", out)


def ci(c, scale=100, nd=1):
    return f"[{scale * c['lo']:.{nd}f}, {scale * c['hi']:.{nd}f}]"


def t2_results():
    """Both suites x four image conditions for the 196-token model, with question types, pair diagnostics, NLL, advantages."""
    d = js(R / "task2" / "vlm196" / "final_eval_stage3.json")
    names = {"correct": "Correct", "swap": "Swapped", "mismatched": "Mismatched", "none": "No image"}
    types = {"grounding_test": [("presence", "Pres."), ("counting", "Count."), (None, "")],
             "vqa_test": [("yes/no", "Y/N"), ("number", "Num."), ("other", "Other")]}
    out = [r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}l rl rrr rrr r rl@{}}", r"\toprule",
           r" & \multicolumn{2}{c}{Accuracy} & \multicolumn{3}{c}{By type} & Pair & Ident. & Follows & Ref. & "
           r"\multicolumn{2}{c}{Correct-image adv.} \\ \cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){11-12}",
           r"Image & \% & 95\% CI & & & & acc. & ans. & partner & NLL & pp & 95\% CI \\"]
    for suite, sname in (("grounding_test", "Grounding"), ("vqa_test", "VQAv2")):
        s = d["suites"][suite]
        n = s["correct"]["n_questions"]
        tlabels = " & ".join(t[1] for t in types[suite])
        out += [r"\midrule", f"\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{sname}}} ({n} q., {n // 2} pairs)}} & {tlabels} & & & & & & \\\\"]
        for c in ("correct", "swap", "mismatched", "none"):
            x = s[c]
            tv = " & ".join(pct(x["by_type"][t]) if t else "" for t, _ in types[suite])
            if c == "correct":
                adv = " & "
            else:
                a = s["advantages"][f"correct_vs_{c}"]["accuracy_advantage"]
                adv = f"{100 * a['mean']:+.1f} & {ci(a)}"
            out.append(f"{names[c]} & {pct(x['accuracy'])} & {ci(x['ci']['accuracy'])} & {tv} & {pct(x['pair_accuracy'])} & "
                       f"{pct(x['identical_answer_rate'])} & {pct(x['follows_partner_rate'])} & {x['mean_reference_nll']:.2f} & {adv} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("t2_results", out)


def t2_resources():
    """196 vs 49 visual tokens: accuracy, grounding advantage, training time per stage, memory, decoding time, parameters."""
    fc = js(R / "task2" / "feature_cache.json")
    out = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l rr rr rrr rr rr@{}}", r"\toprule",
           r" & \multicolumn{2}{c}{Accuracy (\%)} & \multicolumn{2}{c}{Adv. vs swap (pp)} & \multicolumn{3}{c}{Train time (min)} & "
           r"\multicolumn{2}{c}{Peak mem. (GB)} & \multicolumn{2}{c}{ms / answer} \\ "
           r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12}",
           r"Visual tokens & Grd. & VQA & Grd. & VQA & S1 & S2 & S3 & train & eval & image & none \\", r"\midrule"]
    params = None
    for name, lab in (("vlm196", "196"), ("vlm49", "49 (2$\\times$2 pool)")):
        rs = js(R / "task2" / name / "run_summary.json")
        ev = js(R / "task2" / name / "final_eval_stage3.json")
        g, v = ev["suites"]["grounding_test"], ev["suites"]["vqa_test"]
        st = rs["stages"]
        params = [s["trainable_params"] for s in st]
        peak = max(s["peak_allocated_gb"] for s in st)
        ms_img = 1000 * (g["correct"]["seconds_per_answer"] * g["correct"]["n_questions"] + v["correct"]["seconds_per_answer"] * v["correct"]["n_questions"]) / (g["correct"]["n_questions"] + v["correct"]["n_questions"])
        ms_none = 1000 * (g["none"]["seconds_per_answer"] * g["none"]["n_questions"] + v["none"]["seconds_per_answer"] * v["none"]["n_questions"]) / (g["none"]["n_questions"] + v["none"]["n_questions"])
        ag = g["advantages"]["correct_vs_swap"]["accuracy_advantage"]["mean"]
        av = v["advantages"]["correct_vs_swap"]["accuracy_advantage"]["mean"]
        times = " & ".join(f"{s['seconds_this_session'] / 60:.0f}" for s in st)
        out.append(f"{lab} & {pct(g['correct']['accuracy'])} & {pct(v['correct']['accuracy'])} & {100 * ag:.1f} & {100 * av:.1f} & "
                   f"{times} & {peak:.2f} & {ev['peak_allocated_gb']:.2f} & {ms_img:.0f} & {ms_none:.0f} \\\\")
    out += [r"\midrule",
            f"\\multicolumn{{12}}{{@{{}}l}}{{Trainable parameters: S1 connector {params[0] / 1e6:.2f}M; S2--S3 connector + LoRA {params[1] / 1e6:.2f}M.}} \\\\",
            f"\\multicolumn{{12}}{{@{{}}l}}{{SigLIP feature cache (shared): {fc['n_images']:,} images, {fc['file_gb']:.2f} GB, encoded once in {fc['encode_seconds']:.0f} s.}} \\\\",
            r"\bottomrule", r"\end{tabular}"]
    write("t2_resources", out)


TABLES = {f.__name__: f for f in (t1_benchmarks, t1_resources, t1_annotation, t2_results, t2_resources)}


def main(argv):
    if "--list" in argv:
        print("\n".join(TABLES))
        return
    names = [a for a in argv if not a.startswith("-")] or list(TABLES)
    for n in names:
        TABLES[n]()


if __name__ == "__main__":
    main(sys.argv[1:])

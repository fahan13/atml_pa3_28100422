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
    """Trainable parameters, training time, peak GPU memory, checkpoint/export sizes, displacement, method overheads."""
    out = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l r rr r rr r rr@{}}", r"\toprule",
           r" & Trainable & \multicolumn{2}{c}{Training} & Peak & \multicolumn{2}{c}{Size (GB)} & & \multicolumn{2}{c}{Method overhead} \\ "
           r"\cmidrule(lr){3-4} \cmidrule(lr){6-7} \cmidrule(lr){9-10}",
           r"Run & params (M) & h & s/upd. & GB & ckpt & final & $\|\theta-\theta_0\|_2$ & state (GB) & time \\", r"\midrule"]
    for c, lab in RUNS:
        d = R / "task1" / c
        s = js(d / "run_summary.json")
        log = jl(d / "train_log.jsonl")
        spu = sum(r["seconds"] for r in log) / len(log)
        ck = jl(d / "checkpoints.jsonl")[-1]["gb"]
        ex = jl(d / "exports.jsonl")[-1]
        final = ex.get("export_gb", ex.get("adapter_projected_gb", ex.get("adapter_raw_gb")))
        disp = f"{ex['displacement_l2']:.2f}" if "displacement_l2" in ex else "--"
        state, extra_t = "--", "--"
        if c == "l2sp":
            state = f"{s['ref_bytes'] / 1e9:.2f} (CPU)"
            extra_t = f"{sum(r['l2sp_s'] for r in log) / len(log):.2f} s/upd."
        if c == "oplora16":
            state = f"{s['basis_bytes'] / 1e9:.2f}"
            extra_t = f"{s['basis_preprocess_s']:.0f} s once"
        out.append(f"{lab} & {s['trainable_params'] / 1e6:.1f} & {s['elapsed']['train_s'] / 3600:.2f} & {spu:.2f} & "
                   f"{s['peak_allocated_gb']:.2f} & {ck:.3f} & {final:.3f} & {disp} & {state} & {extra_t} \\\\")
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
    out = [r"\begin{tabular}{@{}l cccccc@{}}", r"\toprule",
           r" & Format$^\ast$ & Coherence & Repetition & Math framing & Inappropriate & Truncated \\",
           r" & C/P/F & C/P/O & N/S/L & N/S/H & N/B/Y & (512 tok.) \\", r"\midrule"]
    for c, lab in T1:
        a = rows.get(c, [])

        def tri(k):
            v = [x["labels"].get(k) for x in a if x["labels"].get(k) is not None]
            return "/".join(str(sum(1 for y in v if y == i)) for i in range(3))

        tr = sum(1 for x in a if resp[(c, x["question"] if x["format"] == "plain" else f"{x['question']}_{x['format']}")]["truncated"])
        out.append(f"{lab} & {tri('format')} & {tri('coherence')} & {tri('repetition')} & {tri('math')} & {tri('inappropriate')} & {tr}/{len(a)} \\\\")
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


T3_METHODS = [("first", "First answer"), ("sc", "Self-consistency (SC)"), ("shortest", "Shortest rollout"),
              ("logp", "Max mean log-prob"), ("logp_rwc", "Log-prob + RWC"), ("selfcert", "Max self-certainty"),
              ("selfcert_rwc", "Self-certainty + RWC"), ("vision", "Max vision share"), ("vision_rwc", "Vision + RWC"),
              ("visconf", "Max VisConf"), ("visconf_rwc", "VisConf + RWC"), ("visconf_token", "Max token VisConf"),
              ("visconf_token_rwc", "Token VisConf + RWC")]


def _sgn(v, nd=1):
    return f"{'+' if v >= 0 else '$-$'}{abs(v):.{nd}f}"


def t3_methods():
    """Eval-split accuracy at N = 4 and 32, difference from SC at N = 32 with bootstrap CI, and selection failure."""
    o = {ds: js(R / "task3" / ds / "offline_summary.json") for ds in ("math", "mathvista")}
    out = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l rrrr rrrr@{}}", r"\toprule",
           r" & \multicolumn{4}{c}{MATH (800 q.)} & \multicolumn{4}{c}{MathVista (800 q.)} \\ \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
           r"Method & $N{=}4$ & $N{=}32$ & $\Delta_{\mathrm{SC}}$ [95\% CI] & sel.\ fail & "
           r"$N{=}4$ & $N{=}32$ & $\Delta_{\mathrm{SC}}$ [95\% CI] & sel.\ fail \\", r"\midrule"]

    def cells(ds, m):
        e = o[ds]["splits"]["eval"]["by_N"]
        if m not in e["32"]["methods"]:
            return "& & & "
        a4, a32 = e["4"]["methods"][m]["acc"], e["32"]["methods"][m]
        if m == "sc":
            dlt = "--"
        else:
            b = o[ds]["bootstrap_eval"]["differences"]["32"][f"{m}-sc"]
            dlt = f"{_sgn(100 * b['diff'])} [{_sgn(100 * b['lo'])}, {_sgn(100 * b['hi'])}]"
        return f"{pct(a4)} & {pct(a32['acc'])} & {dlt} & {pct(a32['selection_failure'])}"

    for m, lab in T3_METHODS:
        out.append(f"{lab} & {cells('math', m)} & {cells('mathvista', m)} \\\\")
        if m in ("shortest", "selfcert_rwc"):
            out.append(r"\addlinespace[2pt]")
    orc = " & ".join(f"{pct(o[ds]['splits']['eval']['by_N']['4']['oracle'])} & {pct(o[ds]['splits']['eval']['by_N']['32']['oracle'])} & "
                     f"\\multicolumn{{2}}{{l}}{{sampling fail {pct(1 - o[ds]['splits']['eval']['by_N']['32']['oracle'])}}}"
                     for ds in ("math", "mathvista"))
    out += [r"\midrule", f"Oracle (any correct) & {orc} \\\\", r"\bottomrule", r"\end{tabular}"]
    write("t3_methods", out)


def t3_signals():
    """Within-question association of each score with correctness and with length (eval split, N = 32)."""
    sg = {ds: js(R / "task3" / ds / "signals.json") for ds in ("math", "mathvista")}
    rows = [("logp", "Mean log-prob"), ("selfcert", "Self-certainty"), ("length", "Length (tokens)"), ("vision", "Vision share"),
            ("calibrated_conf", "Calibrated confidence"), ("visconf", "VisConf"), ("visconf_token", "Token VisConf")]
    out = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l rrr rrr@{}}", r"\toprule",
           f" & \\multicolumn{{3}}{{c}}{{MATH ({sg['math']['n_mixed']} mixed q.)}} & \\multicolumn{{3}}{{c}}{{MathVista ({sg['mathvista']['n_mixed']} mixed q.)}} \\\\ "
           r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
           r"Score & $\rho_{\mathrm{corr}}$ & top--bottom & $\rho_{\mathrm{len}}$ & $\rho_{\mathrm{corr}}$ & top--bottom & $\rho_{\mathrm{len}}$ \\", r"\midrule"]

    def c(ds, k):
        x = sg[ds]["scores"].get(k)
        if x is None:
            return " & & "
        rl = "--" if x["mean_spearman_with_length"] is None else _sgn(x["mean_spearman_with_length"], 2)
        return f"{_sgn(x['mean_spearman_with_correct_mixed'], 3)} & {_sgn(100 * x['quartile_gap_mixed'])} & {rl}"

    for k, lab in rows:
        out.append(f"{lab} & {c('math', k)} & {c('mathvista', k)} \\\\")
    cv = sg["mathvista"]["confidence_vision"]["per_rollout_corr"]
    out += [r"\midrule",
            f"\\multicolumn{{7}}{{@{{}}l}}{{MathVista corr$(c_t, v_t)$ within rollouts: mean {_sgn(cv['mean'], 2)}, "
            f"IQR [{_sgn(cv['q25'], 2)}, {_sgn(cv['q75'], 2)}], {100 * cv['frac_positive']:.0f}\\% $>0$.}} \\\\",
            r"\bottomrule", r"\end{tabular}"]
    write("t3_signals", out)


def t3_adaptive():
    """Dev-selected stopping rules on the eval split against fixed N = 32 SC."""
    out = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}ll cc r l rr rrrr@{}}", r"\toprule",
           r" & & & & & & & & \multicolumn{4}{c}{Stops at $n$ (\%)} \\ \cmidrule(lr){9-12}",
           r"Data & Rule & $\tau$ & $\gamma$ & Acc. & $\Delta$ vs.\ $N{=}32$ [95\% CI] & Rollouts & Tokens & 4 & 8 & 16 & 32 \\"]
    for ds, name in (("math", "MATH"), ("mathvista", "MathVista")):
        a = js(R / "task3" / ds / "adaptive.json")
        f = a["fixed32_sc"]["eval"]
        out.append(r"\midrule")
        out.append(f"{name} & fixed $N{{=}}32$ SC & & & {pct(f['acc'])} & & {f['mean_rollouts']:.1f} & {f['mean_cost']:,.0f} & & & & \\\\")
        for fam, lab in (("agreement", "agreement"), ("quality_aware", f"+ {a['quality_score']}")):
            x = a["families"][fam]
            sel, e, dlt = x["selected"], x["eval"], x["eval_acc_diff_vs_fixed32"]
            g = "--" if sel["gamma"] is None else f"{sel['gamma']:g}"
            sd = e["stop_distribution"]
            out.append(f" & {lab} & {sel['tau']:g} & {g} & {pct(e['acc'])} & {_sgn(100 * dlt['diff'], 2)} [{_sgn(100 * dlt['lo'], 2)}, {_sgn(100 * dlt['hi'], 2)}] & "
                       f"{e['mean_rollouts']:.1f} & {e['mean_cost']:,.0f} & " + " & ".join(f"{100 * sd[k]:.0f}" for k in ("4", "8", "16", "32")) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("t3_adaptive", out)


def t3_live():
    """Live Qwen2.5-3B-Instruct strategies on 50 MATH questions under a 3,072-token output allowance."""
    L = js(R / "task3" / "live" / "live_summary.json")
    st = L["strategies"]
    out = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l l r rr r rr@{}}", r"\toprule",
           r"Strategy & Budget (tokens) & Acc. & In & Out & s / q. & Trunc. & Extract. \\", r"\midrule"]
    rows = [("single", "Single answer", "$1\\times3072$", st["single"]["truncation_rate"], st["single"]["extraction_rate"]),
            ("parallel", "Parallel + SC", "$3\\times1024$", st["parallel"]["truncation_rate_per_call"], st["parallel"]["extraction_rate_per_call"]),
            ("refine", "Draft + revision", "$1024 + 2048$", None, st["refine"]["revision_extraction_rate"])]
    for k, lab, bud, tr, ex in rows:
        x = st[k]
        trc = pct(tr, 1) if tr is not None else f"{pct(x['initial_truncation_rate'], 0)} / {pct(x['revision_truncation_rate'], 0)}"
        out.append(f"{lab} & {bud} & {pct(x['accuracy'], 0)} & {x['mean_input_tokens']:.0f} & {x['mean_output_tokens']:.0f} & "
                   f"{x['mean_seconds']:.1f} & {trc} & {pct(ex, 1)} \\\\")
    t = L["transitions_initial_to_revised"]
    pb = L["paired_bootstrap"]
    par, ref = st["parallel"], st["refine"]

    def bs(k):
        return f"{_sgn(100 * pb[k]['diff'], 0)} [{_sgn(100 * pb[k]['lo'], 0)}, {_sgn(100 * pb[k]['hi'], 0)}]"

    notes = [f"Parallel: any-of-3 correct {pct(par['oracle_any_of_3'], 0)}\\%; SC picks no answer {pct(par['winner_no_answer_rate'], 0)}\\%.",
             f"Revision: draft acc.\\ {pct(ref['initial_accuracy'], 0)}\\%; answer changed {pct(ref['answer_changed_rate'], 0)}\\%.",
             f"Draft$\\to$revision: W$\\to$C {t['wrong_to_correct']}, C$\\to$W {t['correct_to_wrong']}, C$\\to$C {t['correct_to_correct']}, W$\\to$W {t['wrong_to_wrong']}.",
             f"Paired bootstrap, parallel$-$single: {bs('parallel-single')} pp.",
             f"Paired bootstrap, revision$-$single: {bs('refine-single')} pp."]
    out.append(r"\midrule")
    out += [f"\\multicolumn{{8}}{{@{{}}l}}{{{n}}} \\\\" for n in notes]
    out += [r"\bottomrule", r"\end{tabular}"]
    write("t3_live", out)


TABLES = {f.__name__: f for f in (t1_benchmarks, t1_resources, t1_annotation, t2_results, t2_resources,
                                  t3_methods, t3_signals, t3_adaptive, t3_live)}


def main(argv):
    if "--list" in argv:
        print("\n".join(TABLES))
        return
    names = [a for a in argv if not a.startswith("-")] or list(TABLES)
    for n in names:
        TABLES[n]()


if __name__ == "__main__":
    main(sys.argv[1:])

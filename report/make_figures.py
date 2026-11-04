"""Regenerate every report figure from saved result files (CPU only, no models).

    python report/make_figures.py            # all figures
    python report/make_figures.py t1_training
    python report/make_figures.py --list

Each figure is one function; each reads only files under results/ and writes
report/figures/<name>.pdf (for LaTeX) and <name>.png (preview). Sized for the
5.5 in NeurIPS text width.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "figures"
W = 5.5

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4", "#008300", "#e34948"]
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"

T1 = [("baseline", "Base", GREY), ("full_ft_mid", "Full FT (mid)", C[0]), ("full_ft", "Full FT", C[0]),
      ("l2sp", "Full FT+L2-SP", C[1]), ("lora4", "LoRA-4", C[2]), ("lora16", "LoRA-16", C[3]),
      ("oplora16", "OPLoRA-16", C[4])]
RUNS = [r for r in T1 if r[0] not in ("baseline", "full_ft_mid")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.linewidth": 0.6, "axes.labelpad": 2, "axes.titlepad": 4, "axes.grid": False,
    "lines.linewidth": 1.3, "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.1, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def jl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def smooth(y, k):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    head = c[1:k] / np.arange(1, k)
    return np.concatenate([head, (c[k:] - c[:-k]) / k])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", (OUT / f"{name}.pdf").relative_to(ROOT))


def sub(ax, s):
    ax.set_title(s, color=INK)


def top_legend(fig, handles, ncol, y=1.02):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


def t1_training():
    """(a) SFT loss of the five runs; (b) GSM8K gain against mean retention change."""
    from matplotlib.lines import Line2D
    d = R / "task1"
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.95), gridspec_kw={"wspace": 0.3})
    handles = []
    for run, lab, col in RUNS:
        log = jl(d / run / "train_log.jsonl")
        x = np.array([r["update"] for r in log])
        y = smooth([r["sft_loss"] for r in log], 50)
        ls = (0, (3, 1.5)) if run == "oplora16" else "-"
        ax[0].plot(x[49:], y[49:], color=col, ls=ls, lw=1.1)
        handles.append(Line2D([], [], color=col, ls=ls, label=lab))
    ax[0].set_xlim(50, 1250)
    ax[0].set_ylim(0.17, 0.30)
    ax[0].set_xlabel("optimizer update")
    ax[0].set_ylabel("SFT loss")
    sub(ax[0], "(a) Training loss (50-update mean)")

    ev = {c: js(d / "eval" / c / "scores.json") for c, _, _ in T1}

    def ret(s):
        return np.mean([s["hellaswag"]["acc_norm"], s["boolq"]["acc"], s["arc_easy"]["acc_norm"]])

    b = ev["baseline"]
    ax[1].axhline(0, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
    place = {"full_ft_mid": (5, -7, "left"), "full_ft": (-6, 0, "right"), "l2sp": (-6, 0, "right"),
             "lora4": (6, 0, "left"), "lora16": (6, 0, "left"), "oplora16": (-6, 0, "right")}
    for c, lab, col in T1[1:]:
        gx = 100 * (ev[c]["gsm8k"]["flexible_em"] - b["gsm8k"]["flexible_em"])
        ry = 100 * (ret(ev[c]) - ret(b))
        ax[1].plot([gx], [ry], "o", ms=5, color=col, mfc="white" if c == "full_ft_mid" else col, mew=1.2, zorder=3)
        dx, dy, ha = place[c]
        ax[1].annotate(lab if c != "full_ft_mid" else "Full FT, update 625", (gx, ry), xytext=(dx, dy),
                       textcoords="offset points", fontsize=7, ha=ha, va="center", color=INK)
    ax[1].set_xlim(11, 25)
    ax[1].set_ylim(-1.5, 6.5)
    ax[1].set_xlabel("GSM8K change from base (pp)")
    ax[1].set_ylabel("retention change (pp)")
    sub(ax[1], "(b) Math gain vs. retention")
    top_legend(fig, handles, 5)
    save(fig, "t1_training")


def t1_diagnostics():
    """down_proj at layers 0/13/27: rho_k, eta, and R(z) on math vs non-math probes."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = R / "task1" / "diagnostics"
    ws = {(r["condition"], r["layer"]): r for r in js(d / "weight_space.json")}
    act = {(r["condition"], r["layer"], r["domain"]): r for r in js(d / "activation_summary.json")}
    conds = T1[1:]
    layers = [0, 13, 27]
    n = len(conds)
    bw = 0.8 / n
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.8), gridspec_kw={"wspace": 0.38})
    for j, (c, lab, col) in enumerate(conds):
        xs = np.arange(3) + (j - (n - 1) / 2) * bw
        hollow = c == "full_ft_mid"
        kw = dict(width=bw * 0.82, color="white" if hollow else col, edgecolor=col, linewidth=0.8)
        ax[0].bar(xs, [ws[(c, L)]["rho_k"] for L in layers], **kw)
        ax[1].bar(xs, [100 * ws[(c, L)]["eta"] for L in layers], **kw)
        rm = [act[(c, L, "math")]["mean"] for L in layers]
        rn = [act[(c, L, "nonmath")]["mean"] for L in layers]
        ax[2].vlines(xs, rn, rm, color=col, lw=0.9, zorder=2)
        ax[2].plot(xs, rm, "o", ms=3, color=col, mfc="white" if hollow else col, mew=0.9, zorder=3)
        ax[2].plot(xs, rn, "o", ms=3, color=col, mfc="white", mew=0.9, zorder=3, alpha=0.9)
    iso = js(d / "weight_space.json")[0]["rho_k_isotropic"]
    ax[0].axhline(iso, color=GREY, lw=0.7, ls=(0, (2, 2)))
    ax[0].set_ylim(0, 0.16)
    ax[0].set_yticks([0, 0.05, 0.10, 0.15])
    ax[0].set_ylabel(r"$\rho_k$")
    sub(ax[0], r"(a) Subspace interference $\rho_{128}$")
    ax[1].set_ylabel(r"$\eta$ (%)")
    sub(ax[1], r"(b) Relative update size $\eta$")
    ax[2].set_ylim(0, 0.105)
    ax[2].set_ylabel(r"mean $R(z)$")
    sub(ax[2], r"(c) $R(z)$: math (top) vs. non-math")
    for a in ax:
        a.set_xticks(range(3), [f"layer {L}" for L in layers])
        a.tick_params(axis="x", length=0)
    handles = [Patch(facecolor="white" if c == "full_ft_mid" else col, edgecolor=col, label=lab if c != "full_ft_mid" else "Full FT (u625)")
               for c, lab, col in conds]
    handles.append(Line2D([], [], color=GREY, ls=(0, (2, 2)), label=r"$k/d_\mathrm{out}$"))
    top_legend(fig, handles, 7)
    save(fig, "t1_diagnostics")


def t2_curves():
    """196-token VLM across the three stages: training and dev caption loss, and dev answer NLL per image condition."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = R / "task2" / "vlm196"
    st = jl(d / "stages.jsonl")
    bounds = [0] + [int(v) for v in np.cumsum([s["updates"] for s in st])]
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.8), gridspec_kw={"wspace": 0.3})
    log = [r for r in jl(d / "train_log.jsonl") if r["loss"] is not None]
    x = np.array([r["update"] for r in log])
    y = np.array([r["loss"] for r in log])
    for s in range(1, len(bounds)):
        m = (x > bounds[s - 1]) & (x <= bounds[s])
        ax[0].plot(x[m], smooth(y[m], 50), color=C[0], lw=1.0)
    dev = jl(d / "dev_log.jsonl")
    u = np.array([r["update"] for r in dev])
    ax[0].plot(u, [r["caption_loss"] for r in dev], "o", ms=3, color=C[1], zorder=3)
    ax[0].set_ylim(0, 4.2)
    ax[0].set_yticks([0, 1, 2, 3, 4])
    ax[0].set_ylabel("loss")
    sub(ax[0], "(a) Training and dev caption loss")
    for a, suite, name in ((ax[1], "grounding_dev", "(b) Grounding dev NLL"), (ax[2], "vqa_dev", "(c) VQA dev NLL")):
        cor = np.array([r[suite]["nll_correct"] for r in dev])
        swp = np.array([r[suite]["nll_swap"] for r in dev])
        non = np.array([r[suite]["nll_none"] for r in dev])
        a.fill_between(u, cor, swp, where=swp >= cor, color=C[2], alpha=0.15, lw=0, interpolate=True)
        a.plot(u, cor, color=C[2], marker="o", ms=2.5)
        a.plot(u, swp, color=C[4], marker="o", ms=2.5)
        a.plot(u, non, color=GREY, ls=(0, (2, 1.5)))
        a.set_ylim(0, 7)
        a.set_yticks([0, 2, 4, 6])
        a.set_ylabel("answer NLL")
        sub(a, name)
    for a in ax:
        a.set_xlim(0, bounds[-1])
        for b in bounds[1:-1]:
            a.axvline(b, color=GREY, lw=0.6, zorder=0)
        a.set_xticks(bounds, [str(b) for b in bounds])
        a.set_xlabel("update")
        for i in range(len(bounds) - 1):
            a.text((bounds[i] + bounds[i + 1]) / 2, 0.97, f"S{i + 1}", transform=a.get_xaxis_transform(),
                   ha="center", va="top", fontsize=7, color=GREY)
    for a in ax[1:]:
        a.set_xticks(bounds[:-1] + [bounds[-1]], ["0", str(bounds[1]), str(bounds[2]), ""])
    ax[0].set_xticks(bounds[:-1] + [bounds[-1]], ["0", str(bounds[1]), str(bounds[2]), ""])
    handles = [Line2D([], [], color=C[0], label="train loss"), Line2D([], [], color=C[1], marker="o", ls="", ms=3, label="dev caption loss"),
               Line2D([], [], color=C[2], marker="o", ms=2.5, label="correct image"), Line2D([], [], color=C[4], marker="o", ms=2.5, label="swapped image"),
               Line2D([], [], color=GREY, ls=(0, (2, 1.5)), label="no image"), Patch(color=C[2], alpha=0.15, label="swap penalty")]
    top_legend(fig, handles, 6)
    save(fig, "t2_curves")


def t3_scaling():
    """Accuracy vs. rollouts N on the eval split for MATH and MathVista, with oracle coverage and the adaptive rules."""
    from matplotlib.lines import Line2D
    d = R / "task3"
    series = [("oracle", "oracle", GREY, (0, (2, 1.5)), None), ("sc", "SC", C[0], "-", "o"),
              ("logp", "max log-prob", C[7], "-", "v"), ("selfcert", "max self-cert.", C[3], "-", "s"),
              ("selfcert_rwc", "self-cert. + RWC", C[3], (0, (3, 1.5)), "s"),
              ("visconf", "max VisConf", C[4], "-", "D"), ("visconf_rwc", "VisConf + RWC", C[4], (0, (3, 1.5)), "D")]
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.95), gridspec_kw={"wspace": 0.25})
    for a, ds, name in ((ax[0], "math", "(a) MATH"), (ax[1], "mathvista", "(b) MathVista")):
        o = js(d / ds / "offline_summary.json")
        ns = [int(n) for n in o["budgets"]]
        byn = o["splits"]["eval"]["by_N"]
        for key, lab, col, ls, mk in series:
            if key == "oracle":
                y = [byn[str(n)]["oracle"] for n in ns]
            elif key in byn[str(ns[0])]["methods"]:
                y = [byn[str(n)]["methods"][key]["acc"] for n in ns]
            else:
                continue
            a.plot(ns, [100 * v for v in y], color=col, ls=ls, marker=mk, ms=3, mfc="white" if ls != "-" else col, lw=1.1)
        ad = js(d / ds / "adaptive.json")
        for fam, mk in (("agreement", "*"), ("quality_aware", "P")):
            e = ad["families"][fam]["eval"]
            a.plot([e["mean_rollouts"]], [100 * e["acc"]], mk, ms=7 if mk == "*" else 5.5, color=INK, zorder=4)
        first = 100 * byn[str(ns[0])]["methods"]["first"]["acc"]
        a.axhline(first, color=GREY, lw=0.6, zorder=0)
        a.set_xscale("log", base=2)
        a.set_xticks(ns, [str(n) for n in ns])
        a.minorticks_off()
        a.set_xlabel("rollouts per question $N$")
        sub(a, name)
    ax[0].set_ylabel("accuracy (%)")
    handles = [Line2D([], [], color=col, ls=ls, marker=mk, ms=3, mfc="white" if ls != "-" else col, label=lab)
               for key, lab, col, ls, mk in series]
    handles += [Line2D([], [], color=INK, marker="*", ls="", ms=7, label="adaptive (agree.)"),
                Line2D([], [], color=INK, marker="P", ls="", ms=5.5, label="adaptive (quality)")]
    top_legend(fig, handles, 5)
    save(fig, "t3_scaling")


FIGS = {f.__name__: f for f in (t1_training, t1_diagnostics, t2_curves, t3_scaling)}


def main(argv):
    if "--list" in argv:
        print("\n".join(FIGS))
        return
    names = [a for a in argv if not a.startswith("-")] or list(FIGS)
    for n in names:
        FIGS[n]()


if __name__ == "__main__":
    main(sys.argv[1:])

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
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.labelpad": 1.5, "axes.titlepad": 3, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.1, "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.2, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
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


def title(ax, s):
    ax.set_title(s, loc="left", color=INK)


def t1_training():
    """(a) SFT loss of the five runs; (b) parameter displacement, Full FT vs L2-SP; (c) math gain vs retention."""
    d = R / "task1"
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.65), gridspec_kw={"width_ratios": [1.15, 1, 1.05], "wspace": 0.36})
    for run, lab, col in RUNS:
        log = jl(d / run / "train_log.jsonl")
        x = [r["update"] for r in log]
        ls = "--" if run == "oplora16" else "-"
        ax[0].plot(x, smooth([r["sft_loss"] for r in log], 25), color=col, ls=ls, label=lab, lw=1.0)
    ax[0].set_ylim(0.16, 0.42)
    ax[0].set_xlim(0, 1250)
    ax[0].set_xlabel("optimizer update")
    ax[0].set_ylabel("SFT loss (25-update mean)")
    ax[0].legend(loc="upper right", ncol=1, handlelength=1.3)
    title(ax[0], "(a) Training loss")

    log = jl(d / "l2sp" / "train_log.jsonl")
    x = np.array([r["update"] for r in log])
    disp = np.sqrt(2 * np.array([r["omega"] for r in log]))
    ax[1].plot(x, disp, color=C[1], label=r"Full FT+L2-SP, $\sqrt{2\Omega}$")
    fe = jl(d / "full_ft" / "exports.jsonl")
    ax[1].plot([e["update"] for e in fe], [e["displacement_l2"] for e in fe], "o", ms=3.5, color=C[0], label="Full FT, exports")
    le = jl(d / "l2sp" / "exports.jsonl")[-1]
    ax[1].plot([le["update"]], [le["displacement_l2"]], "s", ms=3, color=C[1], mfc="white", mew=1)
    for e in fe:
        ax[1].annotate(f"{e['displacement_l2']:.2f}", (e["update"], e["displacement_l2"]), xytext=(0, 3.5),
                       textcoords="offset points", ha="center", va="bottom", fontsize=6, color=INK)
    ax[1].annotate(f"{le['displacement_l2']:.2f}", (le["update"], le["displacement_l2"]), xytext=(0, -4),
                   textcoords="offset points", ha="center", va="top", fontsize=6, color=INK)
    lo = log[-1]
    share = lo["lambda_omega"] / lo["total_loss"]
    ax[1].text(0.97, 0.56, f"final $\\lambda\\Omega$ = {lo['lambda_omega']:.3f}\n({100 * share:.0f}% of objective)",
               transform=ax[1].transAxes, ha="right", va="bottom", fontsize=6, color=MUTED)
    ax[1].set_xlim(0, 1320)
    ax[1].set_ylim(0, 13)
    ax[1].set_xlabel("optimizer update")
    ax[1].set_ylabel(r"$\|\theta-\theta_0\|_2$")
    ax[1].legend(loc="lower right", handlelength=1.2, borderaxespad=0.1)
    title(ax[1], "(b) Distance from start")

    ev = {c: js(d / "eval" / c / "scores.json") for c, _, _ in T1}

    def ret(s):
        return np.mean([s["hellaswag"]["acc_norm"], s["boolq"]["acc"], s["arc_easy"]["acc_norm"]])

    b = ev["baseline"]
    ax[2].axhline(0, color=MUTED, lw=0.6, zorder=1)
    offs = {"full_ft_mid": (5, 0), "full_ft": (-5, 0), "l2sp": (-5, 0), "lora4": (5, 0), "lora16": (5, 0), "oplora16": (-5, 0)}
    has = {"full_ft_mid": "left", "full_ft": "right", "l2sp": "right", "lora4": "left", "lora16": "left", "oplora16": "right"}
    for c, lab, col in T1[1:]:
        gx = 100 * (ev[c]["gsm8k"]["flexible_em"] - b["gsm8k"]["flexible_em"])
        ry = 100 * (ret(ev[c]) - ret(b))
        hollow = c == "full_ft_mid"
        ax[2].plot([gx], [ry], "o", ms=4.5, color=col, mfc="white" if hollow else col, mew=1.1, zorder=3)
        ax[2].annotate(lab, (gx, ry), xytext=offs[c], textcoords="offset points", fontsize=6, ha=has[c], va="center", color=INK)
    ax[2].set_xlim(11, 25)
    ax[2].set_ylim(-1.5, 6.5)
    ax[2].set_xlabel("GSM8K gain (pp)")
    ax[2].set_ylabel("mean retention change (pp)")
    title(ax[2], "(c) Gain vs. retention")
    save(fig, "t1_training")


def t1_diagnostics():
    """down_proj diagnostics at layers 0/13/27: rho_k, eta, and R(z) on math vs non-math probes."""
    d = R / "task1" / "diagnostics"
    ws = {(r["condition"], r["layer"]): r for r in js(d / "weight_space.json")}
    act = {(r["condition"], r["layer"], r["domain"]): r for r in js(d / "activation_summary.json")}
    conds = T1[1:]
    layers = [0, 13, 27]
    n = len(conds)
    bw = 0.8 / n
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.55), gridspec_kw={"wspace": 0.32})
    for j, (c, lab, col) in enumerate(conds):
        xs = np.arange(3) + (j - (n - 1) / 2) * bw
        hollow = c == "full_ft_mid"
        rho = [ws[(c, L)]["rho_k"] for L in layers]
        eta = [ws[(c, L)]["eta"] for L in layers]
        kw = dict(width=bw * 0.86, color="white" if hollow else col, edgecolor=col, linewidth=0.7)
        ax[0].bar(xs, rho, label=lab, **kw)
        ax[1].bar(xs, [100 * e for e in eta], **kw)
        rm = [act[(c, L, "math")]["mean"] for L in layers]
        rn = [act[(c, L, "nonmath")]["mean"] for L in layers]
        for x0, a, b2 in zip(xs, rm, rn):
            ax[2].plot([x0, x0], [b2, a], color=col, lw=0.8, zorder=2)
        ax[2].plot(xs, rm, "o", ms=2.8, color=col, mfc="white" if hollow else col, mew=0.8, zorder=3)
        ax[2].plot(xs, rn, "D", ms=2.4, color=col, mfc="white", mew=0.8, zorder=3)
    iso = js(d / "weight_space.json")[0]["rho_k_isotropic"]
    ax[0].axhline(iso, color=MUTED, lw=0.7, ls=":")
    ax[0].text(2.45, iso, "  $k/d_{out}$", va="center", ha="left", fontsize=6, color=MUTED)
    for i in range(3):
        ax[0].text(i + 2.5 * bw, 0.004, "0", ha="center", va="bottom", fontsize=5.5, color=C[4])
    ax[0].set_ylim(0, 0.17)
    ax[0].set_ylabel(r"$\rho_k$ ($k=128$)")
    title(ax[0], r"(a) Subspace interference $\rho_k$")
    ax[1].set_ylabel(r"$\eta$ (%)")
    title(ax[1], r"(b) Relative update size $\eta$")
    ax[2].set_ylabel(r"$R(z)$ (mean over 32 probes)")
    title(ax[2], r"(c) Activation correction $R(z)$")
    from matplotlib.lines import Line2D
    ax[2].legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, ms=3, label="GSM8K (math)"),
                          Line2D([], [], marker="D", ls="", color=MUTED, mfc="white", ms=2.6, label="HellaSwag")],
                 loc="upper right", ncol=2, handletextpad=0.2, columnspacing=0.8)
    ax[2].set_ylim(0, 0.125)
    for a in ax:
        a.set_xticks(range(3), [f"layer {L}" for L in layers])
        a.grid(axis="x", visible=False)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.1),
               handlelength=1.0, handleheight=0.7)
    save(fig, "t1_diagnostics")


def t2_curves():
    """Learning curves for the 196-token VLM: train loss, dev caption loss, dev answer NLL by image condition, swap penalty."""
    from matplotlib.lines import Line2D
    d = R / "task2" / "vlm196"
    st = jl(d / "stages.jsonl")
    bounds = [0] + [int(v) for v in np.cumsum([s["updates"] for s in st])]
    fig, ax = plt.subplots(1, 4, figsize=(W, 1.55), gridspec_kw={"wspace": 0.42})
    log = [r for r in jl(d / "train_log.jsonl") if r["loss"] is not None]
    x = np.array([r["update"] for r in log])
    y = np.array([r["loss"] for r in log])
    for s in range(1, len(bounds)):
        m = (x > bounds[s - 1]) & (x <= bounds[s])
        ax[0].plot(x[m], smooth(y[m], 50), color=C[0], lw=0.9)
    dev = jl(d / "dev_log.jsonl")
    u = [r["update"] for r in dev]
    ax[1].plot(u, [r["caption_loss"] for r in dev], color=C[0], marker="o", ms=1.8)
    suites = (("grounding_dev", C[2]), ("vqa_dev", C[4]))
    for suite, col in suites:
        ax[2].plot(u, [r[suite]["nll_correct"] for r in dev], color=col, marker="o", ms=1.8)
        ax[2].plot(u, [r[suite]["nll_swap"] for r in dev], color=col, ls="--", lw=0.9)
        ax[2].plot(u, [r[suite]["nll_none"] for r in dev], color=col, ls=":", lw=0.9)
        ax[3].plot(u, [r[suite]["swap_penalty"] for r in dev], color=col, marker="o", ms=1.8)
    ax[0].set_ylim(0, 3.6)
    ax[0].set_ylabel("train loss (50-update mean)")
    title(ax[0], "(a) Training loss")
    ax[1].set_ylabel("dev caption loss")
    ax[1].set_ylim(2.0, 3.2)
    title(ax[1], "(b) Caption loss")
    ax[2].set_ylim(0, 6.2)
    ax[2].set_ylabel("dev answer NLL")
    title(ax[2], "(c) Answer NLL")
    ax[3].axhline(0, color=MUTED, lw=0.6)
    ax[3].set_ylabel("NLL(swapped) $-$ NLL(correct)")
    title(ax[3], "(d) Swap penalty")
    mids = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)]
    for a in ax:
        a.set_xlim(0, bounds[-1])
        for b in bounds[1:-1]:
            a.axvline(b, color=MUTED, lw=0.6, ls=":", zorder=1)
        a.set_xticks(bounds, [""] * len(bounds))
        a.set_xticks(mids, [f"S{i + 1}" for i in range(len(mids))], minor=True)
        a.tick_params(axis="x", which="minor", length=0, pad=2)
        a.grid(axis="x", visible=False)
    fig.supxlabel(f"training stage (boundaries at updates {bounds[1]} and {bounds[2]}; {bounds[-1]} total)", fontsize=6.5, y=-0.04)
    fig.legend(handles=[Line2D([], [], color=C[2], label="grounding"), Line2D([], [], color=C[4], label="VQA"),
                        Line2D([], [], color=MUTED, label="correct image"), Line2D([], [], color=MUTED, ls="--", label="swapped image"),
                        Line2D([], [], color=MUTED, ls=":", label="no image")],
               loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1), handlelength=1.8)
    save(fig, "t2_curves")


FIGS = {f.__name__: f for f in (t1_training, t1_diagnostics, t2_curves)}


def main(argv):
    if "--list" in argv:
        print("\n".join(FIGS))
        return
    names = [a for a in argv if not a.startswith("-")] or list(FIGS)
    for n in names:
        FIGS[n]()


if __name__ == "__main__":
    main(sys.argv[1:])

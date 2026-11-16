"""Figures for the README: architecture, training curve, DET, per-dialect EER,
saliency (where in frequency the CNN looks) and an embedding map that shows
the English / Arabic domain shift."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spec_lcnn import data  # noqa: E402
from spec_lcnn.model import LCNN  # noqa: E402

RES, FIG, CACHE = ROOT / "results", ROOT / "assets" / "figures", data.DATA_ROOT.parent / "cache"
FIG.mkdir(parents=True, exist_ok=True)
REAL, FAKE, GREY, GREEN = "#2563eb", "#dc2626", "#6b7280", "#16a34a"
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})


def save(fig, name):
    fig.savefig(FIG / name, facecolor="white"); plt.close(fig); print("  ", name)


def load_cache(name):
    z = np.load(CACHE / f"{name}.npz")
    return z["wav"], pd.DataFrame(json.loads(str(z["meta"])))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 2.2)); ax.axis("off")
    blocks = [("4 s audio\n(silence removed)", GREY), ("log-power\nspectrogram\n257 x 401", "#0ea5e9"),
              ("Conv 5x5\n+ MFM", "#9333ea"), ("4 x [Conv1x1-MFM\nConv3x3-MFM\nBN, MaxPool]", "#9333ea"),
              ("max over freq\nmean over time", "#16a34a"), ("FC -> logit\n(fake score)", FAKE)]
    for i, (t, c) in enumerate(blocks):
        ax.add_patch(FancyBboxPatch((i * 1.85, 0), 1.55, 1, boxstyle="round,pad=0.05", fc=c, ec="none"))
        ax.text(i * 1.85 + 0.775, 0.5, t, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        if i < len(blocks) - 1:
            ax.annotate("", (i * 1.85 + 1.8, 0.5), (i * 1.85 + 1.6, 0.5), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.set_xlim(-0.1, len(blocks) * 1.85); ax.set_ylim(-0.1, 1.1)
    save(fig, "architecture.png")


def fig_training(m):
    h = pd.DataFrame(m["history"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2)); ax2 = ax.twinx()
    ax.plot(h.epoch, h.loss, "-o", color=GREY, ms=3, label="train loss"); ax.set_ylabel("loss", color=GREY)
    ax2.plot(h.epoch, h.val_eer * 100, "-o", color=GREEN, ms=3); ax2.set_ylabel("validation EER (%)", color=GREEN)
    ax2.spines["right"].set_visible(True); ax.set_xlabel("epoch")
    ax.set_title("Training on English (ASVspoof 2019 LA)", loc="left")
    save(fig, "training_curve.png")


def det(ax, y, s, label, color, ls="-"):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; k = (fpr > 0) & (fnr > 0)
    ax.plot(norm.ppf(fpr[k]), norm.ppf(fnr[k]), color=color, ls=ls, lw=2, label=label)


def fig_det(m_en, m_ar):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ticks = [0.001, 0.01, 0.05, 0.2, 0.5]
    for f in (ax.set_xticks, ax.set_yticks):
        f(norm.ppf(ticks))
    ax.set_xticklabels([f"{t*100:g}" for t in ticks]); ax.set_yticklabels([f"{t*100:g}" for t in ticks])
    ax.set_xlim(norm.ppf(0.0005), norm.ppf(0.6)); ax.set_ylim(norm.ppf(0.0005), norm.ppf(0.6))
    for tag, mm, col in (("en", m_en, REAL), ("ar", m_ar, GREEN)):
        if mm is None:
            continue
        who = "trained on ENGLISH" if tag == "en" else "trained on ARABIC"
        s = pd.read_csv(RES / f"scores_{tag}_on_english.csv")
        det(ax, s.is_fake, s.score, f"{who} -> English test  {mm['english_test']['eer']*100:.1f}%", col, "-")
        s = pd.read_csv(RES / f"scores_{tag}_on_arabic.csv")
        det(ax, s.is_fake, s.score, f"{who} -> Arabic test  {mm['arabic_test']['eer']*100:.1f}%", col, "--")
    ax.plot([-5, 5], [-5, 5], ":", color=GREY, lw=0.8)
    ax.set_xlabel("False alarm rate (%)"); ax.set_ylabel("Miss rate (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right"); ax.set_title("DET curves: the language gap", loc="left")
    save(fig, "det_curves.png")


def fig_gap(m_en, m_ar):
    rows = [("English test", m_en["english_test"]["eer"], m_ar["english_test"]["eer"] if m_ar else np.nan),
            ("Arabic test", m_en["arabic_test"]["eer"], m_ar["arabic_test"]["eer"] if m_ar else np.nan)]
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.8))
    b1 = ax.bar(x - w / 2, [r[1] * 100 for r in rows], w, color=REAL, label="LCNN trained on English")
    b2 = ax.bar(x + w / 2, [r[2] * 100 for r in rows], w, color=GREEN, label="LCNN trained on Arabic")
    for bars in (b1, b2):
        for r in bars:
            if not np.isnan(r.get_height()):
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{r.get_height():.1f}%", ha="center", fontweight="bold")
    ax.axhline(50, ls=":", color=GREY); ax.text(1.45, 51, "coin flip", color=GREY, fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel("EER (%)"); ax.set_ylim(0, 60)
    ax.legend(frameon=False, loc="upper left"); ax.set_title("Same architecture, same recipe, different language", loc="left")
    save(fig, "language_gap.png")


def fig_dialect(m_en, m_ar):
    if "arabic_per_dialect_eer" not in m_en:
        return
    d = sorted(m_en["arabic_per_dialect_eer"])
    x = np.arange(len(d)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(x - w / 2, [m_en["arabic_per_dialect_eer"][k] * 100 for k in d], w, color=REAL, label="trained on English")
    if m_ar:
        ax.bar(x + w / 2, [m_ar["arabic_per_dialect_eer"][k] * 100 for k in d], w, color=GREEN, label="trained on Arabic")
    ax.set_xticks(x); ax.set_xticklabels({"EGY": "Egyptian", "Gulf": "Gulf", "MSA": "MSA"}.get(k, k) for k in d)
    ax.set_ylabel("EER (%)"); ax.legend(frameon=False); ax.set_title("Per-dialect EER on ArA-DF-2026", loc="left")
    save(fig, "per_dialect.png")


def fig_saliency(en_w, en_m, ar_w, ar_m):
    model = LCNN(); model.load_state_dict(torch.load(RES / "models" / "lcnn_en.pt", map_location="cpu")); model.eval()
    freqs = np.linspace(0, 8000, 257)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.22})
    rng = np.random.default_rng(0)
    example = None
    for name, W, M, col in (("English fakes", en_w, en_m, REAL), ("Arabic fakes", ar_w, ar_m, FAKE)):
        idx = rng.choice(np.where(M["is_fake"].values == 1)[0], 96, replace=False)
        sal = 0
        for k in range(0, len(idx), 16):                       # small chunks: full-res backward is memory hungry
            x = torch.from_numpy(W[idx[k:k + 16]].astype(np.float32) / 32767)
            spec = model.front(x).detach().requires_grad_(True)
            model.forward_spec(spec).sum().backward()
            sal = sal + (spec.grad * spec).abs().squeeze(1).mean(dim=2).sum(0).detach().numpy()
        a.plot(freqs, sal / sal.max(), color=col, lw=2, label=name)
        if example is None:
            example = (spec[0, 0].detach().numpy(), (spec.grad[0, 0] * spec[0, 0]).abs().detach().numpy())
    a.axvspan(4000, 8000, color="#f59e0b", alpha=0.1)
    a.set_xlabel("Frequency (Hz)"); a.set_ylabel("relative saliency"); a.legend(frameon=False)
    a.set_title("Where the English-trained CNN looks", loc="left")
    S, G = example
    b.imshow(S, origin="lower", aspect="auto", cmap="gray", extent=[0, 4, 0, 8000])
    b.imshow(np.clip(G / np.percentile(G, 99.5), 0, 1), origin="lower", aspect="auto", cmap="hot", alpha=0.55, extent=[0, 4, 0, 8000])
    b.set_xlabel("time (s)"); b.set_ylabel("Hz"); b.set_title("Gradient x input, one English fake", loc="left")
    save(fig, "saliency.png")


def fig_embedding(en_w, en_m, ar_w, ar_m):
    model = LCNN(); model.load_state_dict(torch.load(RES / "models" / "lcnn_en.pt", map_location="cpu")); model.eval()
    rng = np.random.default_rng(1)
    pts, groups = [], []
    for corpus, W, M in (("English", en_w, en_m), ("Arabic", ar_w, ar_m)):
        for lab, name in ((0, "real"), (1, "fake")):
            idx = rng.choice(np.where(M["is_fake"].values == lab)[0], 300, replace=False)
            with torch.no_grad():
                h = torch.cat([model.body(model.front(torch.from_numpy(W[idx[k:k + 25]].astype(np.float32) / 32767))).amax(2).mean(2)
                               for k in range(0, len(idx), 25)])
            pts.append(h.numpy()); groups += [f"{corpus} {name}"] * len(idx)
    Z = TSNE(perplexity=35, random_state=0, init="pca").fit_transform(np.concatenate(pts))
    groups = np.array(groups)
    sty = {"English real": (REAL, "o"), "English fake": (FAKE, "o"), "Arabic real": (REAL, "^"), "Arabic fake": (FAKE, "^")}
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for g, (c, mk) in sty.items():
        s = groups == g
        ax.scatter(Z[s, 0], Z[s, 1], s=10, c=c, marker=mk, alpha=0.55 if "English" in g else 0.9,
                   edgecolors="none" if "English" in g else "k", linewidths=0.2, label=g)
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, markerscale=2)
    ax.set_title("t-SNE of the English-trained CNN's embedding", loc="left")
    save(fig, "embedding_tsne.png")


def main():
    m_en = json.load(open(RES / "metrics_en.json"))
    m_ar = json.load(open(RES / "metrics_ar.json")) if (RES / "metrics_ar.json").exists() else None
    fig_architecture(); fig_training(m_en); fig_det(m_en, m_ar); fig_gap(m_en, m_ar); fig_dialect(m_en, m_ar)
    en_w, en_m = load_cache("asvspoof"); ar_w, ar_m = load_cache("aradf_dev")
    fig_saliency(en_w, en_m, ar_w, ar_m); fig_embedding(en_w, en_m, ar_w, ar_m)


if __name__ == "__main__":
    main()

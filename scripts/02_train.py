"""Train the LCNN and evaluate on English (held-out speakers) and Arabic.

    python scripts/02_train.py --train asvspoof     # main experiment: English -> Arabic
    python scripts/02_train.py --train aradf_train  # reference: trained on Arabic
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spec_lcnn import data  # noqa: E402
from spec_lcnn.metrics import summary  # noqa: E402
from spec_lcnn.model import LCNN, n_params  # noqa: E402

CACHE = data.DATA_ROOT.parent / "cache"
RES = ROOT / "results"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
SEED = 0


def load(name):
    z = np.load(CACHE / f"{name}.npz")
    return z["wav"], pd.DataFrame(json.loads(str(z["meta"])))


def english_split(meta):
    """Same speaker-disjoint 70/30 split as Model 1 (same seed, same speakers)."""
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=SEED).split(meta, groups=meta["speaker"]))
    return tr, te


@torch.no_grad()
def predict(model, wav, bs=32):
    model.eval()
    out = []
    for i in range(0, len(wav), bs):
        x = torch.from_numpy(wav[i:i + bs].astype(np.float32) / 32767).to(DEV)
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def augment(x: torch.Tensor) -> torch.Tensor:
    x = torch.roll(x, int(np.random.randint(x.shape[1])), dims=1)          # random circular shift
    x = x * torch.empty(x.shape[0], 1, device=x.device).uniform_(0.3, 1.0)  # random gain
    return x + 10 ** np.random.uniform(-4.5, -2.5) * torch.randn_like(x)    # faint noise floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="asvspoof", choices=["asvspoof", "aradf_train"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)

    en_w, en_m = load("asvspoof")
    ar_w, ar_m = load("aradf_dev")
    tr_i, te_i = english_split(en_m)
    if a.train == "asvspoof":
        X, y = en_w[tr_i], en_m["is_fake"].values[tr_i]
    else:
        tw, tm = load("aradf_train")
        X, y = tw, tm["is_fake"].values
    # 10% of training rows held out for model selection
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X)); nv = len(X) // 10
    va, tr = perm[:nv], perm[nv:]

    model = LCNN().to(DEV)
    print(f"LCNN params: {n_params(model)/1e3:.0f}k  device={DEV}  train={len(tr)} val={nv}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    pos_weight = torch.tensor(float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)), dtype=torch.float32, device=DEV)

    tag = "en" if a.train == "asvspoof" else "ar"
    best, hist = 1.0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); losses = []
        order = rng.permutation(tr)
        for i in range(0, len(order) - a.bs + 1, a.bs):
            b = order[i:i + a.bs]
            x = augment(torch.from_numpy(X[b].astype(np.float32) / 32767).to(DEV))
            t = torch.from_numpy(y[b].astype(np.float32)).to(DEV)
            loss = F.binary_cross_entropy_with_logits(model(x), t, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        v = summary(y[va], predict(model, X[va]))["eer"]
        if DEV.type == "mps":
            torch.mps.empty_cache()
        hist.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_eer": v})
        print(f"ep {ep+1:2d} loss {np.mean(losses):.4f} val EER {v*100:.2f}%  ({time.time()-t0:.0f}s)")
        if v <= best:
            best = v
            (RES / "models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), RES / "models" / f"lcnn_{tag}.pt")

    model.load_state_dict(torch.load(RES / "models" / f"lcnn_{tag}.pt", map_location=DEV))
    s_en = predict(model, en_w[te_i]); s_ar = predict(model, ar_w)
    m = {"train_corpus": a.train, "params": n_params(model), "history": hist,
         "english_test": summary(en_m["is_fake"].values[te_i], s_en),
         "arabic_test": summary(ar_m["is_fake"].values, s_ar)}
    pa, te_m = {}, en_m.iloc[te_i]
    for atk in sorted(set(te_m["attack"]) - {"bonafide"}):
        sel = ((te_m["attack"] == atk) | (te_m["attack"] == "bonafide")).values
        pa[atk] = summary(te_m["is_fake"].values[sel], s_en[sel])["eer"]
    m["per_attack_eer"] = pa
    if "dialect" in ar_m:
        m["arabic_per_dialect_eer"] = {d: summary(g["is_fake"].values, s_ar[g.index])["eer"]
                                       for d, g in ar_m.groupby("dialect") if g["is_fake"].nunique() == 2}
    json.dump(m, open(RES / f"metrics_{tag}.json", "w"), indent=2)
    pd.DataFrame({"id": en_m["id"].values[te_i], "is_fake": en_m["is_fake"].values[te_i], "score": s_en}
                 ).to_csv(RES / f"scores_{tag}_on_english.csv", index=False)
    pd.DataFrame({"id": ar_m["id"], "dialect": ar_m.get("dialect"), "is_fake": ar_m["is_fake"], "score": s_ar}
                 ).to_csv(RES / f"scores_{tag}_on_arabic.csv", index=False)
    print(f"[{tag}] EN EER {m['english_test']['eer']*100:.2f}%   AR EER {m['arabic_test']['eer']*100:.2f}%")


if __name__ == "__main__":
    main()

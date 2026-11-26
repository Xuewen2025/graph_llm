"""Train the spatio-temporal graph predictor (Sec. III-A-2). Also pre-trains f_graph (Eq. 14).

    python scripts/train_predictor.py --epochs 30
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import PredictorDataset, collate_predictor, load_scenes, predictor_loss  # noqa: E402
from sage.models import STGraphPredictor, to_dev  # noqa: E402


def evaluate(model, dl, dev):
    model.eval()
    tot = dict(ade=0.0, fde=0.0, n=0)
    with torch.no_grad():
        for b in dl:
            b = to_dev(b, dev)
            out, _, _ = model(b)
            _, m = predictor_loss(out, b)
            tot["ade"] += m["ade"]; tot["fde"] += m["fde"]; tot["n"] += 1
    model.train()
    return {k: tot[k] / max(tot["n"], 1) for k in ("ade", "fde")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="outputs/predictor.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--drop_etypes", nargs="*", default=[], help="ablation, e.g. agent-agent")
    a = ap.parse_args()
    dev = "cuda"
    tr = PredictorDataset(load_scenes(f"{a.cache}/train_scenes.pkl"))
    va = PredictorDataset(load_scenes(f"{a.cache}/val_scenes.pkl"))
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate_predictor,
                    drop_last=True, persistent_workers=True)
    dv = DataLoader(va, a.bs, num_workers=a.workers, collate_fn=collate_predictor)
    model = STGraphPredictor(a.d, a.layers, drop_etypes=a.drop_etypes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl), pct_start=0.05)
    best = 1e9
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ep in range(a.epochs):
        t = time.time()
        for it, b in enumerate(dl):
            b = to_dev(b, dev)
            out, _, _ = model(b)
            loss, m = predictor_loss(out, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if it % 100 == 0:
                print(f"ep {ep} it {it} loss {loss.item():.3f} ade {m['ade']:.3f} fde {m['fde']:.3f}", flush=True)
        r = evaluate(model, dv, dev)
        print(f"== ep {ep} val ADE@3s {r['ade']:.3f} FDE@3s {r['fde']:.3f}  ({time.time() - t:.0f}s)", flush=True)
        if r["ade"] < best:
            best = r["ade"]
            torch.save(dict(model=model.state_dict(), cfg=dict(d=a.d, layers=a.layers, drop_etypes=a.drop_etypes), val=r), a.out)
    print("best val ADE", best)


if __name__ == "__main__":
    main()

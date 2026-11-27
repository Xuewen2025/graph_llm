"""P1: train + evaluate the pure-GNN ego planner (no LLM).

    python scripts/train_gnn_planner.py --out outputs/gnn_plan --init outputs/predictor.pt
Writes {out}/best.pt, train_log.jsonl, eval_val.json, eval_val_pred.jsonl (same format as the LLM runs,
key `text_traj`, so scripts/stratify.py / significance.py can read it).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import load_scenes  # noqa: E402
from sage.gnn_planner import EgoGNNPlanner, PlannerDataset, collate_planner, planner_loss  # noqa: E402
from sage.metrics import summarize  # noqa: E402
from sage.models import to_dev  # noqa: E402
from sage.scene import ego_future  # noqa: E402
from sage.serialize import ego_kinematics  # noqa: E402


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    P, S = [], []
    for b in dl:
        sid = b.pop("sid")
        plan, _ = model(to_dev(b, dev))
        P.append(plan.float().cpu().numpy()); S += sid
    model.train()
    return np.concatenate(P), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="outputs/gnn_plan")
    ap.add_argument("--init", default="outputs/predictor.pt", help="predictor checkpoint or 'none'")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_ego", type=float, default=1.0)
    ap.add_argument("--w_aux", type=float, default=1.0)
    ap.add_argument("--no_nav", action="store_true")
    ap.add_argument("--no_fut", action="store_true")
    ap.add_argument("--drop_etypes", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--max_train", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)

    trs = load_scenes(f"{a.cache}/train_scenes.pkl")
    if a.max_train:
        trs = trs[:a.max_train]
    vas = load_scenes(f"{a.cache}/val_scenes.pkl")
    tr = PlannerDataset(trs, f"{a.cache}/train_records.jsonl")
    va = PlannerDataset(vas, f"{a.cache}/val_records.jsonl")
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate_planner, drop_last=True,
                    persistent_workers=True, prefetch_factor=4)
    dv = DataLoader(va, 128, num_workers=a.workers, collate_fn=collate_planner)

    model = EgoGNNPlanner(drop_etypes=a.drop_etypes, use_nav=not a.no_nav, use_fut=not a.no_fut).to(dev)
    if a.init != "none":
        ck = torch.load(a.init, map_location=dev)
        assert list(ck["cfg"].get("drop_etypes", [])) == list(a.drop_etypes), "init / drop_etypes mismatch"
        model.pred.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl), pct_start=0.05)
    log = open(f"{a.out}/train_log.jsonl", "w")
    gts = {s["sid"]: ego_future(s) for s in vas}
    best = 1e9
    for ep in range(a.epochs):
        t = time.time()
        for it, b in enumerate(dl):
            b.pop("sid")
            b = to_dev(b, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                plan, out = model(b)
            loss, m = planner_loss(plan.float(), out.float(), b, a.w_ego, a.w_aux)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if it % 100 == 0:
                print(f"ep {ep} it {it} loss {loss.item():.3f} " + " ".join(f"{k} {v:.3f}" for k, v in m.items()),
                      flush=True)
        P, S = predict(model, dv, dev)
        ade = float(np.mean([np.linalg.norm(p - gts[s], axis=1).mean() for p, s in zip(P, S)]))
        print(f"== ep {ep} val ego ADE@6s {ade:.3f} ({time.time() - t:.0f}s)", flush=True)
        log.write(json.dumps(dict(ep=ep, val_ade6=ade)) + "\n"); log.flush()
        if ade < best:
            best = ade
            torch.save(dict(model=model.state_dict(), args=vars(a), val_ade6=ade), f"{a.out}/best.pt")

    # final evaluation with the best checkpoint, same metrics as scripts/evaluate.py
    model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev)["model"])
    P, S = predict(model, dv, dev)
    th = json.load(open(f"{a.cache}/fvr_thresholds.json"))
    sc = {s["sid"]: s for s in vas}
    v0 = [ego_kinematics(sc[s])[0] for s in S]
    gt = [gts[s] for s in S]
    res = dict(variant="gnn_planner", out=a.out, n=len(S), thresholds=th,
               text=summarize(list(P), gt, v0, th, [sc[s] for s in S]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{a.out}/eval_val.json", "w"), indent=1)
    with open(f"{a.out}/eval_val_pred.jsonl", "w") as f:
        for p, s in zip(P, S):
            f.write(json.dumps(dict(sid=s, text_traj=p.round(3).tolist())) + "\n")


if __name__ == "__main__":
    main()

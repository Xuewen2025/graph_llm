"""P2: decision-conditioned GNN decoder (+ GNN decision heads). Train, then evaluate with decisions from
GT labels (oracle), the GNN's own heads, and (optionally) an LLM.

    python scripts/train_dec_planner.py --out outputs/dec_plan
    python scripts/train_dec_planner.py --out outputs/dec_plan --eval_only --llm_dec outputs/llm_dec/dec_val.jsonl
Writes {out}/eval_{source}.json and eval_{source}_pred.jsonl (key text_traj), source in gt / own / llm.
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
from sage.gnn_planner import DecisionDataset, DecisionGNNPlanner, collate_decision, decision_loss, planner_loss  # noqa: E402
from sage.metrics import summarize  # noqa: E402
from sage.models import to_dev  # noqa: E402
from sage.scene import ego_future  # noqa: E402
from sage.serialize import ego_kinematics  # noqa: E402

KEYS = ("lane", "lead", "v3", "v6")


@torch.no_grad()
def predict(model, dl, dev, source):
    model.eval()
    P, S, D, G = [], [], [], []
    for b in dl:
        sid = b.pop("sid")
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            plan, _, lg, own = model(b, source)
        used = {"gt": b["dec"], "own": own}.get(source)
        if used is None:
            used = torch.where(b["dec_ovr"] == -2, own, b["dec_ovr"])
        P.append(plan.float().cpu().numpy()); S += sid
        D.append(used.cpu().numpy()); G.append(b["dec"].cpu().numpy())
    model.train()
    return np.concatenate(P), S, np.concatenate(D), np.concatenate(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="outputs/dec_plan")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_dec", type=float, default=0.5)
    ap.add_argument("--w_aux", type=float, default=1.0)
    ap.add_argument("--p_own", type=float, default=0.0, help="prob. of decoding from own decisions in training")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--instr", action="store_true", help="D: condition on instruction embeddings")
    ap.add_argument("--instr_set", default="seen", choices=["seen", "unseen"], help="eval instruction set")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--llm_dec", default=None, help="jsonl {sid, dec} of LLM decisions for source 'llm'")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    C = a.cache
    vas = load_scenes(f"{C}/val_scenes.pkl")
    va = DecisionDataset(vas, f"{C}/val_records.jsonl", f"{C}/val_dec.jsonl", f"{C}/val_nodemap.jsonl")
    dv = lambda: DataLoader(va, 128, num_workers=a.workers, collate_fn=collate_decision)  # noqa: E731
    gts = {s["sid"]: ego_future(s) for s in vas}
    import pickle
    if a.instr:
        va.instr = pickle.load(open(f"{C}/val_instr_emb.pkl", "rb"))
        va.instr_set = a.instr_set
    model = DecisionGNNPlanner(instr_dim=1024 if a.instr else 0).to(dev)

    if not a.eval_only:
        json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
        trs = load_scenes(f"{C}/train_scenes.pkl")
        if a.max_train:
            trs = trs[:a.max_train]
        tr = DecisionDataset(trs, f"{C}/train_records.jsonl", f"{C}/train_dec.jsonl", f"{C}/train_nodemap.jsonl")
        if a.instr:
            tr.instr, tr.train = pickle.load(open(f"{C}/train_instr_emb.pkl", "rb")), True
        dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate_decision, drop_last=True,
                        persistent_workers=True, prefetch_factor=4)
        opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl), pct_start=0.05)
        log = open(f"{a.out}/train_log.jsonl", "w")
        best = 1e9
        for ep in range(a.epochs):
            t = time.time()
            for it, b in enumerate(dl):
                b.pop("sid")
                b = to_dev(b, dev)
                src = "own" if np.random.rand() < a.p_own else "gt"
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    plan, out, lg, _ = model(b, src)
                loss, m = planner_loss(plan.float(), out.float(), b, 1.0, a.w_aux)
                ld, acc = decision_loss(lg, b["dec"])
                loss = loss + a.w_dec * ld
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                if it % 100 == 0:
                    m.update(acc)
                    print(f"ep {ep} it {it} loss {loss.item():.3f} " + " ".join(f"{k} {v:.3f}" for k, v in m.items()),
                          flush=True)
            r = {}
            for src in ("gt", "own"):
                P, S, _, _ = predict(model, dv(), dev, src)
                r[src] = float(np.mean([np.linalg.norm(p - gts[s], axis=1).mean() for p, s in zip(P, S)]))
            print(f"== ep {ep} val ADE@6s gt-dec {r['gt']:.3f} own-dec {r['own']:.3f} ({time.time() - t:.0f}s)", flush=True)
            log.write(json.dumps(dict(ep=ep, **r)) + "\n"); log.flush()
            if r["own"] < best:  # select on the deployable setting (own decisions), not the oracle
                best = r["own"]
                torch.save(dict(model=model.state_dict(), args=vars(a), val=r), f"{a.out}/best.pt")

    model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev)["model"])
    th = json.load(open(f"{C}/fvr_thresholds.json"))
    sc = {s["sid"]: s for s in vas}
    sources = ["gt", "own"]
    if a.llm_dec:
        va.override = {(r := json.loads(l))["sid"]: r["dec"] for l in open(a.llm_dec)}
        sources.append("llm")
    for src in sources:
        P, S, D, G = predict(model, dv(), dev, src)
        v0 = [ego_kinematics(sc[s])[0] for s in S]
        res = dict(source=src, n=len(S), text=summarize(list(P), [gts[s] for s in S], v0, th, [sc[s] for s in S]),
                   dec_acc={k: float((D[:, j] == G[:, j]).mean()) for j, k in enumerate(KEYS)})
        if src == "llm":
            res["llm_missing"] = int(sum(s not in va.override or va.override[s] is None for s in S))
        print(src, json.dumps(res["text"]), res["dec_acc"], flush=True)
        tag = f"{src}_{a.instr_set}" if a.instr else src
        json.dump(res, open(f"{a.out}/eval_{tag}.json", "w"), indent=1)
        with open(f"{a.out}/eval_{tag}_pred.jsonl", "w") as f:
            for p, s, d in zip(P, S, D):
                f.write(json.dumps(dict(sid=s, text_traj=p.round(3).tolist(), dec=d.tolist())) + "\n")


if __name__ == "__main__":
    main()

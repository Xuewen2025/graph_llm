"""Experiment 1: train a VLM planner variant (raw | hist | img | imghist), then evaluate on val.

    python scripts/train_vlm.py --variant img --out outputs/vlm_img
Writes {out}/final.pt (LoRA + merger), eval_val.json, eval_val_pred.jsonl (key text_traj).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage import config as C  # noqa: E402
from sage.data import load_scenes  # noqa: E402
from sage.metrics import summarize  # noqa: E402
from sage.serialize import ego_kinematics, parse_traj  # noqa: E402
from sage.vlm import VL_VARIANTS, VLCollator, VLMPlanner  # noqa: E402


def recs(cache, split, n=0):
    r = [json.loads(l) for l in open(f"{cache}/{split}_records.jsonl")]
    return r[:n] if n else r


def evaluate(model, a, dl_val, tag="eval"):
    model.eval()
    scenes = {s["sid"]: s for s in load_scenes(f"{a.cache}/val_scenes.pkl")}
    th = json.load(open(f"{a.cache}/fvr_thresholds.json"))
    P, G, V, S, fail = [], [], [], [], 0
    with open(f"{a.out}/{tag}_val_pred.jsonl", "w") as f:
        for b in tqdm(dl_val, mininterval=30):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                texts = model.plan(b)
            for i, sid in enumerate(b["sid"]):
                v0 = ego_kinematics(scenes[sid])[0]
                p = parse_traj("Trajectory:" + texts[i].split("Trajectory:")[-1])
                if p is None:
                    fail += 1
                    p = np.stack([v0 * C.DT * np.arange(1, C.N_PLAN + 1), np.zeros(C.N_PLAN)], 1)
                P.append(p); G.append(b["gt"][i].numpy()); V.append(v0); S.append(sid)
                f.write(json.dumps(dict(sid=sid, output=texts[i], text_traj=p.round(2).tolist())) + "\n")
    res = dict(variant=a.variant, n=len(S), parse_fail_rate=fail / max(1, len(S)),
               text=summarize(P, G, V, th, [scenes[s] for s in S]))
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(f"{a.out}/{tag}_val.json", "w"), indent=1)
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VL_VARIANTS))
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_merger", type=float, default=2e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_val", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    a.out = a.out or f"outputs/vlm_{a.variant}"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    model = VLMPlanner(a.vlm, a.variant, a.lora_r, a.grad_ckpt).cuda()
    mk = lambda split, n, bs, sh: DataLoader(  # noqa: E731
        recs(a.cache, split, n), bs, shuffle=sh, drop_last=sh, num_workers=a.workers,
        collate_fn=VLCollator(model.proc, a.variant, f"{a.cache}/bev/{split}"), persistent_workers=sh)
    dl_val = mk("val", a.max_val, a.eval_bs, False)
    if a.eval_only:
        model.load_state_dict(torch.load(f"{a.out}/final.pt", map_location="cuda"), strict=False)
        evaluate(model, a, dl_val)
        return
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    dl = mk("train", a.max_train, a.bs, True)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    merger = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    print(f"trainable: lora {sum(p.numel() for p in lora)/1e6:.1f}M merger {sum(p.numel() for p in merger)/1e6:.1f}M"
          f" train {len(dl.dataset)}", flush=True)
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": merger, "lr": a.lr_merger}], weight_decay=0.0)
    total = int(a.epochs * len(dl) / a.accum)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log = open(f"{a.out}/train_log.jsonl", "a")
    step, t0, hist = 0, time.time(), []
    model.train()
    while step < total:
        for it, b in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(b)["loss"]
            (loss / a.accum).backward()
            hist.append(loss.item())
            if (it + 1) % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(lora + merger, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                m = dict(loss=sum(hist) / len(hist), step=step, total=total, lr=sched.get_last_lr()[0],
                         min=(time.time() - t0) / 60, mem=torch.cuda.max_memory_allocated() / 2**30)
                hist = []
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
                log.write(json.dumps(m) + "\n"); log.flush()
            if step >= total:
                break
    torch.save(model.trainable_state_dict(), f"{a.out}/final.pt")
    print("saved", flush=True)
    evaluate(model, a, dl_val)


if __name__ == "__main__":
    main()

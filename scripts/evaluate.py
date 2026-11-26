"""Open-loop evaluation (ADE/FDE @3s/6s, FVR, collision proxy) of a trained variant.

    python scripts/evaluate.py --variant full --ckpt outputs/llm_full/final.pt
    python scripts/evaluate.py --variant raw --ckpt none          # zero-shot raw LLM
Writes {out}/eval_{split}.json and predictions jsonl.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage import config as C  # noqa: E402
from sage.data import load_scenes  # noqa: E402
from sage.llm import VARIANTS, SageLLM  # noqa: E402
from sage.metrics import fvr_thresholds, summarize  # noqa: E402
from sage.serialize import ego_kinematics, parse_traj  # noqa: E402
from train_llm import build_loader  # noqa: E402


def get_thresholds(cache):
    p = f"{cache}/fvr_thresholds.json"
    if os.path.exists(p):
        return json.load(open(p))
    sc = load_scenes(f"{cache}/train_scenes.pkl")
    gts = [s["agent_pos"][0, C.T0 + 1:C.T0 + 1 + C.N_PLAN] for s in sc]
    v0s = [ego_kinematics(s)[0] for s in sc]
    th = fvr_thresholds(gts, v0s)
    json.dump(th, open(p, "w"), indent=1)
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--q_tokens", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    a = ap.parse_args()
    a.out = a.out or f"outputs/llm_{a.variant}"
    os.makedirs(a.out, exist_ok=True)
    tag = "zeroshot" if a.ckpt == "none" else "eval"

    th = get_thresholds(a.cache)
    model = SageLLM(a.llm, a.variant, lora_r=a.lora_r, q_tokens=a.q_tokens).cuda().eval()
    if a.ckpt and a.ckpt != "none":
        missing = model.load_state_dict(torch.load(a.ckpt, map_location="cuda"), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    dl = build_loader(a, a.split, model, a.bs, False, a.max)
    scenes = {s["sid"]: s for s in load_scenes(f"{a.cache}/{a.split}_scenes.pkl")}

    res = dict(sid=[], gt=[], text=[], head=[], v0=[], parse_fail=0)
    fout = open(f"{a.out}/{tag}_{a.split}_pred.jsonl", "w")
    for b in tqdm(dl):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            texts, head = model.plan(b)
        for i, sid in enumerate(b["sid"]):
            v0 = ego_kinematics(scenes[sid])[0]
            p = parse_traj(texts[i])
            if p is None:  # unparsable output -> constant-velocity fallback (counted as failure)
                res["parse_fail"] += 1
                p = np.stack([v0 * C.DT * np.arange(1, C.N_PLAN + 1), np.zeros(C.N_PLAN)], 1)
            res["sid"].append(sid); res["gt"].append(b["gt"][i].numpy()); res["text"].append(p)
            res["v0"].append(v0)
            if head is not None:
                res["head"].append(head[i])
            fout.write(json.dumps(dict(sid=sid, output=texts[i], text_traj=p.round(2).tolist(),
                                       head_traj=None if head is None else head[i].round(2).tolist())) + "\n")
    sc = [scenes[s] for s in res["sid"]]
    metrics = dict(variant=a.variant, ckpt=a.ckpt, n=len(res["sid"]), thresholds=th,
                   parse_fail_rate=res["parse_fail"] / max(1, len(res["sid"])),
                   text=summarize(res["text"], res["gt"], res["v0"], th, sc))
    if res["head"]:
        metrics["head"] = summarize(res["head"], res["gt"], res["v0"], th, sc)
    # simple physics baseline for reference
    cv = [np.stack([v * C.DT * np.arange(1, C.N_PLAN + 1), np.zeros(C.N_PLAN)], 1) for v in res["v0"]]
    metrics["const_vel_ref"] = summarize(cv, res["gt"], res["v0"], th, sc)
    print(json.dumps(metrics, indent=1))
    json.dump(metrics, open(f"{a.out}/{tag}_{a.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

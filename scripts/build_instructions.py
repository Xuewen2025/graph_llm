"""D: instructions per scene -> data/cache/{split}_instr.jsonl
   train: {sid, seen: [K variants]};  val: {sid, seen: [1], unseen: [1]}

    python scripts/build_instructions.py --split train
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import load_scenes  # noqa: E402
from sage.graph import history_graphs  # noqa: E402
from sage.instruct import make_instruction  # noqa: E402

AUX = None


def _init(cache, split):
    global AUX
    AUX = ({(r := json.loads(l))["sid"]: r["nodes"] for l in open(f"{cache}/{split}_nodemap.jsonl")},
           {(r := json.loads(l))["sid"]: r["dec"] for l in open(f"{cache}/{split}_dec.jsonl")}, split)


def _one(s):
    nm, dec, split = AUX
    g0 = history_graphs(s)[-1]
    rng = np.random.default_rng(int(s["sid"].replace("-", "")[:8], 16))
    out = dict(sid=s["sid"])
    if split == "train":
        out["seen"] = [make_instruction(s, g0, nm[s["sid"]], dec[s["sid"]], rng, "seen") for _ in range(4)]
    else:
        seed = int(s["sid"].replace("-", "")[8:16], 16)   # same mentioned components in both versions
        out["seen"] = [make_instruction(s, g0, nm[s["sid"]], dec[s["sid"]], rng, "seen", np.random.default_rng(seed))]
        out["unseen"] = [make_instruction(s, g0, nm[s["sid"]], dec[s["sid"]], rng, "unseen", np.random.default_rng(seed))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    scenes = load_scenes(f"{a.cache}/{a.split}_scenes.pkl")
    with Pool(4, initializer=_init, initargs=(a.cache, a.split)) as p:
        out = p.map(_one, scenes, chunksize=64)
    with open(f"{a.cache}/{a.split}_instr.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(a.split, len(out))
    for r in out[:3]:
        print(r)


if __name__ == "__main__":
    main()

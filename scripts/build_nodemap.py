"""Prompt node name -> graph node index, for node-level graph tokens (variant `node`).

    python scripts/build_nodemap.py --split train      -> data/cache/train_nodemap.jsonl
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import load_scenes  # noqa: E402
from sage.graph import history_graphs  # noqa: E402
from sage.serialize import node_map  # noqa: E402


def _one(s):
    return dict(sid=s["sid"], nodes=node_map(s, history_graphs(s)[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    scenes = load_scenes(f"{a.cache}/{a.split}_scenes.pkl")
    with Pool(16) as p:
        out = p.map(_one, scenes, chunksize=64)
    with open(f"{a.cache}/{a.split}_nodemap.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(a.split, len(out))


if __name__ == "__main__":
    main()

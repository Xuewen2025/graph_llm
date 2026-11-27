"""P2: derive decision labels from GT -> data/cache/{split}_dec.jsonl  {sid, dec, text}

    python scripts/build_decisions.py --split train
"""
import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import load_scenes  # noqa: E402
from sage.decision import derive, to_text  # noqa: E402

NM = None


def _init(p):
    global NM
    NM = {(r := json.loads(l))["sid"]: r["nodes"] for l in open(p)}


def _one(s):
    d = derive(s, NM[s["sid"]])
    return dict(sid=s["sid"], dec=d, text=to_text(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    scenes = load_scenes(f"{a.cache}/{a.split}_scenes.pkl")
    with Pool(8, initializer=_init, initargs=(f"{a.cache}/{a.split}_nodemap.jsonl",)) as p:
        out = p.map(_one, scenes, chunksize=64)
    with open(f"{a.cache}/{a.split}_dec.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n = len(out)
    print(a.split, n, "| lane none %.1f%%" % (100 * sum(r["dec"]["lane"] == "none" for r in out) / n),
          "| lead none %.1f%%" % (100 * sum(r["dec"]["lead"] == "none" for r in out) / n))
    print("lane", collections.Counter(r["dec"]["lane"] for r in out).most_common(6))
    print("lead", collections.Counter(r["dec"]["lead"] for r in out).most_common(6))
    print("v6", sorted(collections.Counter(r["dec"]["v6"] for r in out).items()))
    print("example:", out[0]["text"])


if __name__ == "__main__":
    main()

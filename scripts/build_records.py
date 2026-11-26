"""Build text records (prompts for every variant + GT) for a split.

    python scripts/build_records.py --split train
Output: data/cache/{split}_records.jsonl, one line per scenario:
  {sid, gt, nav, raw, hist, fut}   (scene texts; instructions are added at training time)
"""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import load_scenes  # noqa: E402
from sage.graph import future_graphs, history_graphs  # noqa: E402
from sage.scene import ego_future  # noqa: E402
from sage.serialize import graph_prompt, nav_command, raw_prompt  # noqa: E402

PREDS = None


def _init(p):
    global PREDS
    with open(p, "rb") as f:
        PREDS = pickle.load(f)


def _rec(s):
    gt = ego_future(s)
    nav = nav_command(gt)
    h = history_graphs(s)
    f = future_graphs(s, PREDS[s["sid"]])
    return dict(sid=s["sid"], gt=gt.round(3).tolist(), nav=nav, raw=raw_prompt(s, nav),
                hist=graph_prompt(s, h, None, nav), fut=graph_prompt(s, h, f, nav))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    scenes = load_scenes(f"{a.cache}/{a.split}_scenes.pkl")
    with Pool(32, initializer=_init, initargs=(f"{a.cache}/{a.split}_pred.pkl",)) as p:
        recs = list(tqdm(p.imap(_rec, scenes, chunksize=16), total=len(scenes)))
    with open(f"{a.cache}/{a.split}_records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(len(recs), "records")


if __name__ == "__main__":
    main()

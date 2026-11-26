"""AV2 raw scenarios -> list of ego-centric scene dicts (pickle).

    python scripts/preprocess.py --split train --root data/av2 --out data/cache
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.scene import load_scene  # noqa: E402


def _load(d):
    try:
        return load_scene(d)
    except Exception as e:  # corrupted / incomplete downloads
        print("skip", d, e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--root", default="data/av2")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    dirs = sorted(os.path.join(a.root, a.split, d) for d in os.listdir(os.path.join(a.root, a.split)))
    if a.max:
        dirs = dirs[: a.max]
    with Pool(a.workers) as p:
        scenes = [s for s in tqdm(p.imap(_load, dirs, chunksize=16), total=len(dirs)) if s is not None]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.split}_scenes.pkl"), "wb") as f:
        pickle.dump(scenes, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{a.split}: {len(scenes)} scenes saved")


if __name__ == "__main__":
    main()

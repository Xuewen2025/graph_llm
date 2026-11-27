"""Exp 1: render BEV rasters for every scene -> {cache}/bev/{split}/{sid}.png

    python scripts/render_bev.py --split train
"""
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.bev import render  # noqa: E402
from sage.data import load_scenes  # noqa: E402

OUT = None


def _one(s):
    render(s).save(f"{OUT}/{s['sid']}.png", optimize=False, compress_level=3)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    OUT = f"{a.cache}/bev/{a.split}"
    os.makedirs(OUT, exist_ok=True)
    scenes = load_scenes(f"{a.cache}/{a.split}_scenes.pkl")
    with Pool(7) as p:
        p.map(_one, scenes, chunksize=64)
    print(a.split, len(scenes))


if __name__ == "__main__":
    main()

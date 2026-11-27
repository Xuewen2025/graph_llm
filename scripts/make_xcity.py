"""Cross-city split: train on all cities except --hold, test on --hold (train+val scenes of that city)
plus the in-domain val scenes of the other cities.  -> data/cache_xcity/{train,val}_*   (+ val_city.json)

    python scripts/make_xcity.py --hold pittsburgh
Then rebuild predictor-dependent files with a predictor trained on the source cities only (scripts/xcity_jobs.sh).
"""
import argparse
import json
import os
import pickle
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import load_scenes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", default="pittsburgh")
    ap.add_argument("--src", default="data/cache")
    ap.add_argument("--dst", default="data/cache_xcity")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    city = {**json.load(open(f"{a.src}/train_city.json")), **json.load(open(f"{a.src}/val_city.json"))}
    tr, va = load_scenes(f"{a.src}/train_scenes.pkl"), load_scenes(f"{a.src}/val_scenes.pkl")
    new_tr = [s for s in tr if city[s["sid"]] != a.hold]
    new_va = [s for s in tr if city[s["sid"]] == a.hold] + va           # OOD city (train+val) + all val
    for name, sc in (("train", new_tr), ("val", new_va)):
        with open(f"{a.dst}/{name}_scenes.pkl", "wb") as f:
            pickle.dump(sc, f)
        ids = {s["sid"] for s in sc}
        for kind in ("nodemap", "dec"):   # predictor-independent per-scene files
            with open(f"{a.dst}/{name}_{kind}.jsonl", "w") as f:
                for sp in ("train", "val"):
                    for l in open(f"{a.src}/{sp}_{kind}.jsonl"):
                        if json.loads(l)["sid"] in ids:
                            f.write(l)
        json.dump({s["sid"]: city[s["sid"]] for s in sc}, open(f"{a.dst}/{name}_city.json", "w"))
        print(name, len(sc))
    shutil.copy(f"{a.src}/fvr_thresholds.json", a.dst)


if __name__ == "__main__":
    main()

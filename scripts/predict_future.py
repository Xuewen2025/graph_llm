"""Run the trained graph predictor -> predicted dynamic states for future graph reconstruction.

    python scripts/predict_future.py --split train
Output: data/cache/{split}_pred.pkl  {sid: [Na, TF, 5] (px,py,vx,vy,heading)}
"""
import argparse
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import PredictorDataset, collate_predictor, load_scenes  # noqa: E402
from sage.models import STGraphPredictor, to_dev  # noqa: E402


def load_predictor(path, dev="cuda"):
    ck = torch.load(path, map_location="cpu")
    m = STGraphPredictor(**ck["cfg"])
    m.load_state_dict(ck["model"])
    return m.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--ckpt", default="outputs/predictor.pt")
    a = ap.parse_args()
    scenes = load_scenes(f"{a.cache}/{a.split}_scenes.pkl")
    model = load_predictor(a.ckpt)
    dl = DataLoader(PredictorDataset(scenes), 128, num_workers=16, collate_fn=collate_predictor)
    preds, k = {}, 0
    with torch.no_grad():
        for b in tqdm(dl):
            nag = b["n_agents"].tolist()
            out, _, _ = model(to_dev(b, "cuda"))
            st = STGraphPredictor.denorm(out).cpu().numpy()
            for i, n in enumerate(nag):
                preds[scenes[k]["sid"]] = st[i, :n].astype("float32")
                k += 1
    with open(f"{a.cache}/{a.split}_pred.pkl", "wb") as f:
        pickle.dump(preds, f)
    print(len(preds), "saved")


if __name__ == "__main__":
    main()

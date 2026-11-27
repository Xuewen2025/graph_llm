"""Experiment 1 comparison (VLM: image vs graph) -> outputs/vlm_results.md

    python scripts/compare_vlm.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_p1 import main  # noqa: E402

RUNS = {"vlm_ego": "outputs/vlm_ego", "vlm_raw": "outputs/vlm_raw", "vlm_img": "outputs/vlm_img", "vlm_hist": "outputs/vlm_hist",
        "vlm_imghist": "outputs/vlm_imghist", "llm_raw(0.6B)": "outputs/llm_raw", "llm_hist(0.6B)": "outputs/llm_hist",
        "gnn_plan": "outputs/gnn_plan"}
PAIRS = [("vlm_img", "vlm_ego"), ("vlm_hist", "vlm_ego"), ("vlm_imghist", "vlm_ego"), ("vlm_raw", "vlm_ego"), ("vlm_hist", "vlm_img"), ("vlm_imghist", "vlm_img"), ("vlm_hist", "vlm_raw"), ("vlm_img", "vlm_raw"),
         ("vlm_imghist", "vlm_hist")]


def turn_groups(sc):
    nav = {(r := json.loads(l))["sid"]: r["nav"] for l in open("data/cache/val_records.jsonl")}
    return {"turn (nav left/right)": lambda s: nav[s] in ("turn left", "turn right"),
            "intersection (ego lane)": lambda s: bool(sc[s]["lane_inter"].any()) and nav[s] != "stop / stay"}


if __name__ == "__main__":
    main(RUNS, PAIRS, "outputs/vlm_results.md", turn_groups)

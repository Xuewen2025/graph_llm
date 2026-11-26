"""Load an AV2 motion-forecasting scenario into a compact ego-centric scene dict.

All coordinates are in the ego frame at t0 (x forward, y left), sampled at 0.5 s.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import config as C


def _resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    if len(pts) == 1:
        return np.repeat(pts, n, 0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(pts[:1], n, 0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)


def _pt_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """distance from points p [..., 2] to segments a,b [S,2] -> [..., S]"""
    ab = b - a
    ap = p[..., None, :] - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p[..., None, :] - proj, axis=-1)


def load_scene(scn_dir: str):
    pq = glob.glob(os.path.join(scn_dir, "scenario_*.parquet"))[0]
    mp = glob.glob(os.path.join(scn_dir, "log_map_archive_*.json"))[0]
    df = pd.read_parquet(pq)
    with open(mp) as f:
        amap = json.load(f)

    sid = str(df.scenario_id.iloc[0])
    raw = np.array(C.RAW_STEPS)
    S = len(raw)
    tracks = {}
    for tid, g in df.groupby("track_id", sort=False):
        ts = g.timestep.values
        arr = np.full((110, 5), np.nan, np.float64)
        arr[ts] = g[["position_x", "position_y", "velocity_x", "velocity_y", "heading"]].values
        tracks[tid] = (arr[raw], str(g.object_type.iloc[0]))
    if "AV" not in tracks:
        return None
    ego_arr = tracks["AV"][0]
    if np.isnan(ego_arr).any():
        return None

    # ego frame at t0
    o = ego_arr[C.T0, :2].copy()
    yaw = ego_arr[C.T0, 4]
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])

    def tf_pos(p):
        return (p - o) @ R.T

    def tf_vec(v):
        return v @ R.T

    def tf_head(h):
        return (h - yaw + np.pi) % (2 * np.pi) - np.pi

    # ---- dynamic agents ----
    cand = []
    for tid, (arr, otype) in tracks.items():
        if tid == "AV":
            continue
        if np.isnan(arr[C.T0]).any():
            continue
        d = np.linalg.norm(arr[C.T0, :2] - o)
        if d <= C.AGENT_RADIUS:
            cand.append((d, tid))
    cand.sort()
    ids = ["AV"] + [t for _, t in cand[: C.MAX_AGENTS]]
    Na = len(ids)
    pos = np.zeros((Na, S, 2), np.float32)
    vel = np.zeros((Na, S, 2), np.float32)
    head = np.zeros((Na, S), np.float32)
    valid = np.zeros((Na, S), bool)
    cls = np.zeros(Na, np.int64)
    for k, tid in enumerate(ids):
        arr, otype = tracks[tid]
        v = ~np.isnan(arr).any(1)
        valid[k] = v
        a = np.nan_to_num(arr)
        pos[k] = np.where(v[:, None], tf_pos(a[:, :2]), 0)
        vel[k] = np.where(v[:, None], tf_vec(a[:, 2:4]), 0)
        head[k] = np.where(v, tf_head(a[:, 4]), 0)
        cls[k] = 0 if tid == "AV" else C.agent_class(otype)

    # ---- lanes ----
    lanes = []
    for lid, ls in amap["lane_segments"].items():
        cl = np.array([[p["x"], p["y"]] for p in ls["centerline"]])
        lb = np.array([[p["x"], p["y"]] for p in ls["left_lane_boundary"]])
        rb = np.array([[p["x"], p["y"]] for p in ls["right_lane_boundary"]])
        cl = tf_pos(_resample_polyline(cl, C.LANE_PTS))
        d = np.linalg.norm(cl, axis=1).min()
        if d > C.LANE_RADIUS:
            continue
        width = np.linalg.norm(_resample_polyline(lb, 5) - _resample_polyline(rb, 5), axis=1).mean()
        lanes.append((d, int(ls["id"]), cl, width, ls))
    lanes.sort(key=lambda x: x[0])
    lanes = lanes[: C.MAX_LANES]
    lid2idx = {l[1]: i for i, l in enumerate(lanes)}
    M = len(lanes)
    lane_pts = np.zeros((M, C.LANE_PTS, 2), np.float32)
    lane_width = np.zeros(M, np.float32)
    lane_inter = np.zeros(M, bool)
    lane_type = np.zeros(M, np.int64)
    lane_left = -np.ones(M, np.int64)
    lane_right = -np.ones(M, np.int64)
    lane_has_left = np.zeros(M, bool)
    lane_has_right = np.zeros(M, bool)
    lane_succ = []
    lane_ids = np.zeros(M, np.int64)
    for i, (_, lid, cl, w, ls) in enumerate(lanes):
        lane_ids[i] = lid
        lane_pts[i] = cl
        lane_width[i] = w
        lane_inter[i] = bool(ls["is_intersection"])
        lane_type[i] = C.LANE_TYPES.index(ls["lane_type"]) if ls["lane_type"] in C.LANE_TYPES else 0
        ln, rn = ls.get("left_neighbor_id"), ls.get("right_neighbor_id")
        lane_has_left[i] = ln is not None
        lane_has_right[i] = rn is not None
        lane_left[i] = lid2idx.get(ln, -1) if ln is not None else -1
        lane_right[i] = lid2idx.get(rn, -1) if rn is not None else -1
        for sc in ls["successors"]:
            if sc in lid2idx:
                lane_succ.append((i, lid2idx[sc]))
    lane_succ = np.array(lane_succ, np.int64).reshape(-1, 2)

    # ---- pedestrian crossings (regulatory nodes; AV2 has no traffic-light states) ----
    xws = []
    for xid, xw in amap["pedestrian_crossings"].items():
        e1 = tf_pos(np.array([[p["x"], p["y"]] for p in xw["edge1"]]))
        e2 = tf_pos(np.array([[p["x"], p["y"]] for p in xw["edge2"]]))
        seg = np.stack([(e1[0] + e2[0]) / 2, (e1[-1] + e2[-1]) / 2])  # crosswalk axis
        width = np.linalg.norm(e1[0] - e2[0])
        d = _pt_seg_dist(np.zeros(2), seg[:1], seg[1:])[0]
        if d <= C.XW_RADIUS:
            xws.append((d, seg, width))
    xws.sort(key=lambda x: x[0])
    xws = xws[: C.MAX_XW]
    xw_seg = np.array([x[1] for x in xws], np.float32).reshape(-1, 2, 2)
    xw_width = np.array([x[2] for x in xws], np.float32)

    return dict(
        sid=sid, city=str(df.city.iloc[0]),
        agent_ids=np.array(ids), agent_pos=pos, agent_vel=vel, agent_head=head,
        agent_valid=valid, agent_cls=cls,
        lane_ids=lane_ids, lane_pts=lane_pts, lane_width=lane_width, lane_inter=lane_inter,
        lane_type=lane_type, lane_left=lane_left, lane_right=lane_right,
        lane_has_left=lane_has_left, lane_has_right=lane_has_right, lane_succ=lane_succ,
        xw_seg=xw_seg, xw_width=xw_width,
    )


def ego_future(scene) -> np.ndarray:
    """GT ego positions at t0 + k*dt, k = 1..N  -> [N, 2] (ego frame)"""
    return scene["agent_pos"][0, C.T0 + 1: C.T0 + 1 + C.N_PLAN].copy()

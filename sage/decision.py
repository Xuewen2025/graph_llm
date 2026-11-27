"""P2: discrete driving decisions, derived automatically from the GT ego future (no teacher LLM).

A decision points at nodes named in the graph prompt:
    lane   : named lane L_k the ego is on at t0+6 s ("none" if not on a named lane)
    lead   : named agent A_k that occupies the ego's future path ahead of it (closest gap), or "none"
    v3, v6 : ego speed at t0+3 s / t0+6 s, binned to 2 m/s (bin index 0..NV-1, value = 2*bin m/s)
Text form (LLM target):  "Decision: lane L2; lead A3; speed@3s 8 m/s; speed@6s 10 m/s"
"""
import re

import numpy as np

from . import config as C

NV = 11            # speed bins: 0,2,...,20 m/s
LAT_TOL = 2.0      # lateral tolerance to the ego path for the lead agent (m)
GAP_MAX = 30.0     # max longitudinal gap for the lead agent (m)


def _proj_polyline(p, line):
    """p [K,2], line [P,2] -> lateral dist [K], arc length of the projection [K]"""
    a, b = line[:-1], line[1:]
    ab = b - a
    L = np.linalg.norm(ab, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])[:-1]
    t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum(L ** 2, 1e-6)[None], 0, 1)
    q = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(p[:, None] - q, axis=-1)                     # [K,S]
    k = d.argmin(1)
    return d[np.arange(len(p)), k], cum[k] + t[np.arange(len(p)), k] * L[k]


def speed_bin(v):
    return int(np.clip(np.round(v / 2.0), 0, NV - 1))


def derive(scene, names):
    """names: prompt name -> graph node index (sage.serialize.node_map). Returns dict of names / bins."""
    fut = scene["agent_pos"][0, C.T0 + 1:C.T0 + 1 + C.N_PLAN]
    path = np.concatenate([np.zeros((1, 2)), fut], 0)
    # extend the path GAP_MAX metres beyond the 6 s endpoint along the final heading, so that a vehicle that
    # stays ahead of the ego still projects onto the path
    u = fut[-1] - fut[-3]
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 0.5 else np.array([np.cos(scene["agent_head"][0, C.T0 + 11]),
                                                                       np.sin(scene["agent_head"][0, C.T0 + 11])])
    path_ext = np.concatenate([path, fut[-1] + GAP_MAX * u[None]], 0)
    Na = len(scene["agent_pos"])
    # ---- target lane at 6 s
    end = fut[-1]
    if np.linalg.norm(fut[-1] - fut[-3]) > 0.5:
        hd = np.arctan2(*(fut[-1] - fut[-3])[::-1])
    else:
        hd = scene["agent_head"][0, C.T0 + C.N_PLAN]
    lane, best = "none", 2.5
    for nm, idx in names.items():
        if nm[0] != "L":
            continue
        pts = scene["lane_pts"][idx - Na]
        d, s = _proj_polyline(end[None], pts)
        seg = np.clip(np.searchsorted(np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1)), s[0]), 0, len(pts) - 2)
        lh = np.arctan2(*(pts[seg + 1] - pts[seg])[::-1])
        dh = abs((lh - hd + np.pi) % (2 * np.pi) - np.pi)
        if d[0] < best and dh < np.radians(60):
            lane, best = nm, d[0]
    # ---- lead agent on the ego's future path
    _, s_ego = _proj_polyline(fut, path_ext)
    lead, gap_best = "none", GAP_MAX
    if np.linalg.norm(fut[-1]) > 1.0:
        for nm, idx in names.items():
            if nm[0] != "A":
                continue
            v = scene["agent_valid"][idx, C.T0 + 1:C.T0 + 1 + C.N_PLAN].astype(bool)
            if not v.any():
                continue
            d, s = _proj_polyline(scene["agent_pos"][idx, C.T0 + 1:C.T0 + 1 + C.N_PLAN][v], path_ext)
            gap = s - s_ego[v]
            ok = (d < LAT_TOL) & (gap > 0) & (gap < GAP_MAX)
            if ok.any() and gap[ok].min() < gap_best:
                lead, gap_best = nm, gap[ok].min()
    # speeds from position differences (AV2 velocity at the last raw frame is often 0, and the last position is
    # lagged by track smoothing) -> v3 over 2.5-3 s, v6 over 5-5.5 s
    v3 = np.linalg.norm(path[6] - path[5]) / C.DT
    v6 = np.linalg.norm(path[11] - path[10]) / C.DT
    return dict(lane=lane, lead=lead, v3=speed_bin(v3), v6=speed_bin(v6))


def to_text(d):
    return f"Decision: lane {d['lane']}; lead {d['lead']}; speed@3s {2 * d['v3']} m/s; speed@6s {2 * d['v6']} m/s"


_RE = re.compile(r"lane (L\d+|none); lead (A\d+|none); speed@3s (\d+) m/s; speed@6s (\d+) m/s")


def parse(text):
    m = _RE.search(text)
    if m is None:
        return None
    return dict(lane=m[1], lead=m[2], v3=speed_bin(float(m[3])), v6=speed_bin(float(m[4])))


def to_index(d, names):
    """decision dict -> (lane node idx or -1, lead node idx or -1, v3, v6)"""
    return (names.get(d["lane"], -1) if d["lane"] != "none" else -1,
            names.get(d["lead"], -1) if d["lead"] != "none" else -1, d["v3"], d["v6"])

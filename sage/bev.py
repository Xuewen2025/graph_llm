"""Experiment 1 (image vs graph for a VLM): bird's-eye-view raster of a scene.

The raster carries the same information as the history/current scene graph (hist variant): lanes (width,
direction, intersection flag), crosswalks, all valid agents with class, current pose and the 2 s history
(C.HIST_IDX), and the ego. Ego frame: x forward (image up), y left (image left).
"""
import numpy as np
from PIL import Image, ImageDraw

from . import config as C

SIZE = 448                 # 16 x 28 px -> 256 visual tokens in Qwen2.5-VL
FWD, BACK, LAT = 60.0, 30.0, 45.0
RES = SIZE / (FWD + BACK)  # px per metre (~4.98)

CLS_COLOR = [(40, 120, 255), (255, 60, 60), (255, 165, 0), (190, 70, 255), (220, 220, 40)]
CLS_SIZE = [(4.6, 1.9), (0.9, 0.9), (1.9, 0.8), (12.0, 2.6), (1.2, 1.2)]   # length, width (m)
EGO_COLOR = (0, 230, 0)

IMG_DESC = ("Bird's-eye-view image of the scene: ego = green box near the bottom centre, facing up (up = x forward, "
            f"left = y left), {FWD + BACK:.0f} m x {2 * LAT:.0f} m, {1 / RES:.1f} m per pixel. Boxes are agents "
            "(blue vehicle, red pedestrian, orange cyclist, purple bus, yellow object) with a white mark at the front; "
            "fading trails are their positions over the past 2 s. Grey areas are lanes (darker centre line with "
            "direction ticks; bluish = inside an intersection); white striped areas are crosswalks.")


def px(p):
    """ego-frame metres [...,2] -> pixel (col,row)"""
    p = np.asarray(p, np.float64)
    return np.stack([(LAT - p[..., 1]) * RES, (FWD - p[..., 0]) * RES], -1)


def _box(c, h, L, W):
    d, n = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    pts = [c + d * L / 2 + n * W / 2, c + d * L / 2 - n * W / 2, c - d * L / 2 - n * W / 2, c - d * L / 2 + n * W / 2]
    return [tuple(q) for q in px(np.array(pts))]


def _agent(dr, scene, i, color, L, W):
    valid = scene["agent_valid"][i]
    hist = [t for t in C.HIST_IDX if valid[t]]
    for a, b in zip(hist[:-1], hist[1:]):   # trail, fading towards the past
        f = 0.35 + 0.65 * (C.HIST_IDX.index(b) / (len(C.HIST_IDX) - 1))
        col = tuple(int(f * c) for c in color)
        dr.line([tuple(px(scene["agent_pos"][i, a])), tuple(px(scene["agent_pos"][i, b]))], fill=col, width=3)
    for t in hist[:-1]:
        q = px(scene["agent_pos"][i, t])
        dr.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=tuple(int(0.6 * c) for c in color))
    c, h = scene["agent_pos"][i, C.T0], scene["agent_head"][i, C.T0]
    dr.polygon(_box(c, h, L, W), fill=color, outline=(255, 255, 255))
    fr = c + np.array([np.cos(h), np.sin(h)]) * L / 2
    dr.line([tuple(px(c + np.array([np.cos(h), np.sin(h)]) * L * 0.25)), tuple(px(fr))], fill=(255, 255, 255), width=2)


def render(scene):
    img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    dr = ImageDraw.Draw(img)
    lp = scene["lane_pts"]
    for l in range(len(lp)):   # drivable surface
        col = (70, 70, 95) if scene["lane_inter"][l] else (75, 75, 75)
        w = max(2, int(round(float(scene["lane_width"][l]) * RES)))
        dr.line([tuple(q) for q in px(lp[l])], fill=col, width=w, joint="curve")
    for x in range(len(scene["xw_seg"])):  # crosswalks: striped rectangles
        a, b = scene["xw_seg"][x]
        d = b - a
        ln = np.linalg.norm(d)
        if ln < 1e-3:
            continue
        d = d / ln
        n = np.array([-d[1], d[0]])
        hw = float(scene["xw_width"][x]) / 2
        for s in np.arange(0, ln, 1.0):
            p0, p1 = a + d * s, a + d * min(s + 0.5, ln)
            dr.polygon([tuple(q) for q in px(np.array([p0 + n * hw, p1 + n * hw, p1 - n * hw, p0 - n * hw]))],
                       fill=(230, 230, 230))
    for l in range(len(lp)):   # centre lines with direction ticks
        q = px(lp[l])
        dr.line([tuple(v) for v in q], fill=(140, 140, 140), width=1)
        for k in range(1, len(q) - 1, 3):
            d = q[k + 1] - q[k]
            nd = np.linalg.norm(d)
            if nd < 1e-3:
                continue
            d /= nd
            n = np.array([-d[1], d[0]])
            tip = q[k + 1]
            dr.line([tuple(tip - 4 * d + 3 * n), tuple(tip), tuple(tip - 4 * d - 3 * n)], fill=(170, 170, 170), width=1)
    order = [i for i in range(1, len(scene["agent_pos"])) if scene["agent_valid"][i, C.T0]]
    for i in order:
        k = min(int(scene["agent_cls"][i]), 4)
        _agent(dr, scene, i, CLS_COLOR[k], *CLS_SIZE[k])
    _agent(dr, scene, 0, EGO_COLOR, 4.6, 1.9)
    return img

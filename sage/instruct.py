"""D: natural-language driving instructions generated from the GT decision (sage/decision.py).

Nodes are referred to by description (class, relative position, lane relation), never by prompt ids, so the
model has to ground language to the scene. Two disjoint template sets: 'seen' (training) and 'unseen'
(held-out paraphrases / synonyms, test only). Each component is mentioned with probability P_MENTION.
"""
import numpy as np

from . import config as C

P_MENTION = 0.7
CLS_WORDS = {"seen": ["vehicle", "pedestrian", "cyclist", "bus", "object"],
             "unseen": ["car", "person on foot", "bike rider", "coach", "obstacle"]}

LANE_DESC = {
    "seen": {"same": "your current lane", "left": "the lane to your left", "right": "the lane to your right",
             "succ": "the lane continuing straight ahead", "turn_left": "the lane that turns left",
             "turn_right": "the lane that turns right", "other": "the lane ahead"},
    "unseen": {"same": "the lane you are already in", "left": "the adjacent left lane",
               "right": "the adjacent right lane", "succ": "the road segment directly in front of you",
               "turn_left": "the left-turn lane", "turn_right": "the right-turn lane", "other": "the upcoming lane"},
}
LANE_T = {"seen": ["Be in {} in 6 seconds.", "End up in {}.", "Drive into {}.", "Use {}."],
          "unseen": ["Six seconds from now you should be travelling in {}.", "Make sure you finish in {}.",
                     "Your target is {}."]}
LEAD_T = {"seen": ["Follow the {} {}.", "Stay behind the {} {}.", "Yield to the {} {}.", "Keep a gap to the {} {}."],
          "unseen": ["Don't overtake the {} {}.", "Let the {} {} go first.", "Hang back behind the {} {}."]}
NOLEAD_T = {"seen": ["No one is blocking your path.", "The path ahead is free."],
            "unseen": ["Nothing is in your way.", "You have a clear road."]}
SPEED_T = {"seen": ["Reach about {} m/s in 3 s and {} m/s in 6 s.", "Target speed: {} m/s at 3 s, {} m/s at 6 s."],
           "unseen": ["Be going roughly {} metres per second after three seconds and {} after six.",
                      "Aim for {} m/s by the 3-second mark, then {} m/s by 6 seconds."]}
STOP_T = {"seen": ["Come to a stop.", "Stop the car."], "unseen": ["Bring the vehicle to a halt.", "Pull up and wait."]}
POS_DESC = {"seen": ["{} m ahead", "on your left about {} m away", "{} m behind", "on your right about {} m away"],
            "unseen": ["roughly {} metres in front", "about {} metres off to the left", "around {} metres back",
                       "about {} metres off to the right"]}


def lane_relation(scene, g0, lane_local):
    el = int(g0["agent_lane"][0])
    if el >= 0:
        if lane_local == el:
            return "same"
        if scene["lane_left"][el] == lane_local:
            return "left"
        if scene["lane_right"][el] == lane_local:
            return "right"
        if any(a == el and b == lane_local for a, b in scene["lane_succ"]):
            pts = scene["lane_pts"][lane_local]
            h = np.arctan2(*(pts[-1] - pts[-2])[::-1])
            if h > np.radians(30):
                return "turn_left"
            if h < -np.radians(30):
                return "turn_right"
            return "succ"
    pts = scene["lane_pts"][lane_local]
    h = np.arctan2(*(pts[-1] - pts[-2])[::-1])
    return "turn_left" if h > np.radians(45) else "turn_right" if h < -np.radians(45) else "other"


def agent_desc(scene, idx, ts):
    p = scene["agent_pos"][idx, C.T0]
    ang = np.degrees(np.arctan2(p[1], p[0]))
    rel = 0 if abs(ang) <= 45 else 1 if 45 < ang <= 135 else 3 if -135 <= ang < -45 else 2
    dist = int(max(5, 5 * round(np.linalg.norm(p) / 5)))
    cls = CLS_WORDS[ts][min(int(scene["agent_cls"][idx]), 4)]
    return cls, POS_DESC[ts][rel].format(dist)


def make_instruction(scene, g0, names, dec, rng, ts="seen", rng_m=None):
    """rng: template choice; rng_m: which components are mentioned (same seed -> same content, other wording)"""
    rng_m = rng if rng_m is None else rng_m
    Na = len(scene["agent_pos"])
    parts = []
    if dec["lane"] != "none" and rng_m.random() < P_MENTION:
        rel = lane_relation(scene, g0, names[dec["lane"]] - Na)
        parts.append(rng.choice(LANE_T[ts]).format(LANE_DESC[ts][rel]))
    if rng_m.random() < P_MENTION:
        if dec["lead"] != "none":
            cls, pos = agent_desc(scene, names[dec["lead"]], ts)
            parts.append(rng.choice(LEAD_T[ts]).format(cls, pos))
        else:
            parts.append(rng.choice(NOLEAD_T[ts]))
    if rng_m.random() < P_MENTION:
        if dec["v6"] == 0 and dec["v3"] <= 1:
            parts.append(rng.choice(STOP_T[ts]))
        else:
            parts.append(rng.choice(SPEED_T[ts]).format(2 * dec["v3"], 2 * dec["v6"]))
    rng.shuffle(parts)
    return " ".join(parts) if parts else "Drive normally."

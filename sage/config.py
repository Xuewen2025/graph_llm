"""Global constants for SAGE-Drive on Argoverse 2 Motion Forecasting.

AV2 scenarios: 11 s @ 10 Hz (110 steps), current frame index 49 (5 s history, 6 s future).
We resample everything to dt = 0.5 s:
    sub-step index s -> raw timestep 4 + 5*s, s = 0..21 ; current t0 = s 9 (raw 49)
"""

DT = 0.5
RAW_STEPS = [4 + 5 * s for s in range(22)]  # 22 sub-steps of 0.5 s
T0 = 9                                       # index of current frame in sub-steps

# Historical window T_hist = {t0 - Th, ..., t0}   (Eq. 1): 2 s @ 0.5 s -> 5 graphs
TH = 4
HIST_IDX = list(range(T0 - TH, T0 + 1))
# Future graph prediction horizon T_f (Eq. 10): 3 s @ 0.5 s -> 6 graphs
TF = 6
# Planning horizon: N = ceil(T_plan / dt) = 12 points (6 s) -- AV2 provides 6 s of future
T_PLAN = 6.0
N_PLAN = 12

# Scene selection
AGENT_RADIUS = 50.0
MAX_AGENTS = 32          # surrounding agents (+ ego)
LANE_RADIUS = 60.0
MAX_LANES = 48
XW_RADIUS = 50.0
MAX_XW = 8
LANE_PTS = 10

# Edge rules R(i, j, t)
AA_RADIUS = 30.0         # agent-agent proximity
LANE_ASSIGN_DIST = 2.5   # agent on lane
XW_NEAR_DIST = 10.0      # regulatory influence of a crosswalk

# Node types
NT_AGENT, NT_LANE, NT_XW = 0, 1, 2
NODE_TYPES = ["agent", "lane", "crosswalk"]
NODE_DIM = 16

# Agent classes
AGENT_CLASSES = ["vehicle", "pedestrian", "cyclist", "bus", "other"]
_CLS_MAP = {
    "vehicle": 0, "pedestrian": 1, "cyclist": 2, "motorcyclist": 2, "riderless_bicycle": 2,
    "bus": 3, "static": 4, "background": 4, "construction": 4, "unknown": 4,
}


def agent_class(object_type: str) -> int:
    return _CLS_MAP.get(object_type, 4)


LANE_TYPES = ["VEHICLE", "BIKE", "BUS"]

# Edge types (directed j -> i, features expressed in the frame of target i)
EDGE_TYPES = [
    "agent-agent",   # spatial proximity
    "lane-next",     # j is a successor lane of i
    "lane-prev",     # j is a predecessor lane of i
    "lane-left",     # j is the left neighbour of i
    "lane-right",    # j is the right neighbour of i
    "agent-on-lane", # agent j occupies lane i
    "lane-has-agent",# lane j is occupied by agent i
    "xw-agent",      # crosswalk j regulates agent i
    "agent-xw",      # agent j is near crosswalk i
]
RELATIONS = ["in_front_of", "left_of", "behind", "right_of"]
EDGE_DIM = 8 + len(RELATIONS) + len(EDGE_TYPES)  # r_geo (8) + r_lang (13) = 21

# Normalisation scales for network inputs
POS_SCALE = 20.0
VEL_SCALE = 10.0
ACC_SCALE = 3.0

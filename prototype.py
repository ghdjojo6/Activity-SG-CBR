"""
Floor Plan Recommendation System — local run app
Run: python app.py
Browser: http://localhost:8888
"""

import json, pickle, threading, webbrowser, time, os, re, datetime
import numpy as np
from flask import Flask, request, jsonify
from copy import deepcopy
from collections import defaultdict

import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry.base import BaseGeometry

app = Flask(__name__)

# ════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════
WHERE_MAP = {
    "bedroom1":"bedroom1","bedroom2":"bedroom2","bedroom3":"bedroom3","bedroom4":"bedroom4",
    "bedroom5":"bedroom1","bedroom6":"bedroom1",
    "bathroom1":"bathroom1","bathroom2":"bathroom2","bathroom3":"bathroom3",
    "kitchen_dining":"kitchen","kitchen":"kitchen","dining_room":"kitchen",
    "living_room":"living","living":"living",
    "storage_balcony":"storage_balcony","garden":"garden","front_door":"front_door","garage":"garage",
}
ACT_MAP = {
    32:"garden",16:"front_door",17:"front_door",18:"front_door",19:"front_door",
    20:"front_door",21:"front_door",22:"front_door",8:"front_door",10:"front_door",
    11:"front_door",30:"front_door",58:"front_door",62:"front_door",63:"front_door",
    64:"front_door",65:"front_door",66:"front_door",67:"front_door",
}
ROOM_TYPES = ["bedroom","bathroom","living","kitchen","storage_balcony","front_door","garden"]
EDGE_PAIRS = [tuple(sorted([a,b])) for i,a in enumerate(ROOM_TYPES) for b in ROOM_TYPES[i+1:]]

def pairs_from_vec(vec):
    """
    Converts an adjacency/connection 0-1 vector into a human-readable list of "type-type" pair strings.
    For debugging: lets you visually confirm which pairs are actually being counted.
    """
    return [f"{a}-{b}" for (a, b), v in zip(EDGE_PAIRS, vec) if v > 0.5]

def dirs_from_vec(vec):
    """
    Converts a 28-dimensional direction (ROOM_TYPES x N/S/E/W) 0-1 vector into a list of "type:direction" strings.
    For debugging: lets you visually confirm which type's which direction is actually being counted.
    """
    dirs = ["N","S","E","W"]
    out = []
    for ti, t in enumerate(ROOM_TYPES):
        for di, d in enumerate(dirs):
            if vec[ti*4+di] > 0.5:
                out.append(f"{t}:{d}")
    return out

VIZ_COLOR_MAP = {
    'land':'#F7F3EE','wall':'#111111','living':'#7FD1B9','kitchen':'#F5F79A',
    'bedroom':'#7DA7D9','bathroom':'#F28FA9','balcony':'#DDDDDD',
    'storage_balcony':'#DDDDDD','storage':'#DDDDDD','garden':'#A8E6A3',
    'front_door':'#FF6B35','door':'#AAAAAA','window':'#7FDBFF',
    'stair':'#C8A882','parking':'#B0B0B0','inner':'#E8D5C4',
    'veranda':'#C8E6C9','pool':'#B3E5FC','neighbor':'#EEEEEE',
}
VIZ_LABEL_MAP = {
    'living':'LIVING','bedroom':'BEDROOM','kitchen':'KITCHEN',
    'bathroom':'BATHROOM','storage_balcony':'STORAGE/BALCONY',
    'storage':'STORAGE/BALCONY','balcony':'STORAGE/BALCONY',
    'garden':'GARDEN','parking':'GARAGE','inner':'HALL','veranda':'VERANDA',
}
VIZ_DRAW_ORDER = [
    'land','inner','veranda','pool','neighbor','wall','living','kitchen',
    'bedroom','bathroom','balcony','storage','storage_balcony',
    'garden','parking','front_door','stair','window','door'
]

# ════════════════════════════════════════════════════════
# JSON -> sequence
# ════════════════════════════════════════════════════════
def parse_people_count(val):
    if val == "6_or_more": return 6
    try: return int(val)
    except: return 1

def get_size(p, room=None):
    if room == "front_door": return [(10,10)]
    if room == "living":
        if p<=1: return [(10,20),(20,10)]
        elif p==2: return [(20,20)]
        elif p<=4: return [(20,30),(30,20)]
        else: return [(30,30)]
    if p<=1: return [(10,10)]
    elif p==2: return [(20,10),(10,20)]
    elif p<=4: return [(20,20)]
    elif p<=6: return [(20,30),(30,20)]
    else: return [(30,30)]

def json_to_sequence(jd):
    raw, room_max = [], {}
    for e in jd["entries"]:
        act = int(e["mainActivity"])
        rtp = (e.get("spatialPreferences",{}).get("roomType","") or "").strip().lower()
        wh  = (e.get("where","") or "").strip().lower()
        if rtp == "outside":
            # Skip Outside activities instead of turning them into a room
            # so the sequence connects straight to the next activity
            continue
        room = WHERE_MAP.get(rtp) or WHERE_MAP.get(wh) or ACT_MAP.get(act)
        if not room: continue
        pc = parse_people_count(e.get("peopleCount",1))
        raw.append({"startTime":e["startTime"],"activity":act,"room":room,"pc":pc})
        room_max[room] = max(room_max.get(room,0), pc)
    room_sizes = {r: get_size(mp, r) for r,mp in room_max.items()}
    return raw, room_sizes

# ════════════════════════════════════════════════════════
# Shape Grammar
# ════════════════════════════════════════════════════════
def generate_floorplans(seq, room_sizes, max_iters=200000):
    full_seq = [s["room"] for s in seq]
    unique_rooms = list(dict.fromkeys(full_seq))
    req_pairs = set()
    for k in range(len(full_seq)-1):
        r1,r2 = full_seq[k], full_seq[k+1]
        if r1!=r2: req_pairs.add(tuple(sorted([r1,r2])))

    def get_offsets(sb, d, nw, nh):
        minx,miny,maxx,maxy = sb
        sw,sh = maxx-minx, maxy-miny
        if d in ("right","left"): mn,mx = -(nh-10), sh-10
        else:                     mn,mx = -(nw-10), sw-10
        offs = list(range(int(mn//10)*10, int(mx)+1, 10))
        if 0 not in offs: offs.append(0)
        return offs

    def overlaps(b1, b2):
        return b1[0]<b2[2] and b1[2]>b2[0] and b1[1]<b2[3] and b1[3]>b2[1]

    def touching(b1, b2):
        shareX = min(b1[2],b2[2]) > max(b1[0],b2[0])
        shareY = min(b1[3],b2[3]) > max(b1[1],b2[1])
        return ((b1[2]==b2[0] or b2[2]==b1[0]) and shareY) or \
               ((b1[3]==b2[1] or b2[3]==b1[1]) and shareX)

    def place(sb, d, w, h, off):
        minx,miny,maxx,maxy = sb
        if d=="right":  x,y = maxx,     miny+off
        elif d=="left": x,y = minx-w,   miny+off
        elif d=="top":  x,y = minx+off, maxy
        else:           x,y = minx+off, miny-h
        return (x, y, x+w, y+h)

    DIRS = ["right","top","bottom","left"]
    completed = []
    queue = []
    for w,h in room_sizes.get(unique_rooms[0], [(10,10)]):
        queue.append({"rooms":{unique_rooms[0]:(0,0,w,h)}, "rem":unique_rooms[1:]})

    iters = 0
    while queue and iters<max_iters:
        iters += 1
        state = queue.pop(0)
        rooms, rem = state["rooms"], state["rem"]
        if not rem:
            valid = all(
                (rooms.get(a) and rooms.get(b) and touching(rooms[a], rooms[b]))
                for a,b in req_pairs if a in rooms and b in rooms
            )
            if valid:
                # via_door connection: if two rooms that appear consecutively in the
                # circulation sequence (req_pairs) actually touch, treat them as
                # "connected by a door (via_door)".
                # Why we don't use all touching relations in the completed floor plan:
                # when placing rooms, two rooms that don't need to be connected by the
                # activity schedule can end up next to each other by pure coincidence,
                # and counting that incidental touching as a "connection" would
                # overestimate connections that aren't actually needed.
                # (This approach used to leave conn_pairs empty, which always produced
                # 0% in DB comparisons, but containment_ratio now treats an empty query
                # requirement (=0 pairs) as 1.0, so this is no longer an issue.)
                conn_pairs = set()
                for a, b in req_pairs:
                    if a in rooms and b in rooms and touching(rooms[a], rooms[b]):
                        conn_pairs.add(tuple(sorted([a, b])))
                completed.append({"rooms": dict(rooms), "conn_pairs": conn_pairs})
            continue
        rtype = rem[0]; rest = rem[1:]
        req_nb = set()
        for a,b in req_pairs:
            if a==rtype and b in rooms: req_nb.add(b)
            if b==rtype and a in rooms: req_nb.add(a)
        sources = [(k,rooms[k]) for k in req_nb] if req_nb else list(rooms.items())
        for src_k, sb in sources:
            for w,h in room_sizes.get(rtype, [(10,10)]):
                for d in DIRS:
                    for off in get_offsets(sb, d, w, h):
                        nb = place(sb, d, w, h, off)
                        if any(overlaps(nb, eb) for eb in rooms.values()): continue
                        if not touching(sb, nb): continue
                        new_rooms = dict(rooms); new_rooms[rtype] = nb
                        queue.append({"rooms":new_rooms, "rem":rest})
    return completed

# ════════════════════════════════════════════════════════
# Deduplication & rotation grouping
# ════════════════════════════════════════════════════════
def normalize_plan(rooms):
    """Normalize all room coordinates relative to the origin"""
    if not rooms: return {}
    min_x = min(b[0] for b in rooms.values())
    min_y = min(b[1] for b in rooms.values())
    return {k: (b[0]-min_x, b[1]-min_y, b[2]-min_x, b[3]-min_y) for k,b in rooms.items()}

def plan_signature(rooms):
    """Normalized hash of a floor plan (for exact-duplicate detection)"""
    n = normalize_plan(rooms)
    return tuple(sorted((k, tuple(round(x) for x in v)) for k,v in n.items()))

def rotate_90(rooms):
    """Rotate the floor plan 90 degrees (x,y) -> (-y, x) then re-normalize"""
    rotated = {k: (-b[3], b[0], -b[1], b[2]) for k,b in rooms.items()}
    return normalize_plan(rotated)

def get_all_rotations(rooms):
    """Return all 4 rotational orientations"""
    rots = [rooms]
    for _ in range(3):
        rots.append(rotate_90(rots[-1]))
    return rots

def rotation_canonical(rooms):
    """Use the lexicographically smallest of the 4 rotations as the canonical form"""
    return min(plan_signature(r) for r in get_all_rotations(rooms))

def deduplicate_plans(plans):
    """Remove exact-duplicate floor plans"""
    seen = set()
    result = []
    for p in plans:
        sig = plan_signature(p["rooms"])
        if sig not in seen:
            seen.add(sig)
            result.append(p)
    return result

def group_by_rotation(plans):
    """Group floor plans that are identical under rotation. Returns: [{canonical_sig, plans:[...]}]"""
    groups = {}
    for p in plans:
        canon = rotation_canonical(p["rooms"])
        if canon not in groups:
            groups[canon] = []
        groups[canon].append(p)
    return list(groups.values())

# ════════════════════════════════════════════════════════
# Fingerprint
# ════════════════════════════════════════════════════════
def normalize_type(t):
    if t.startswith("bedroom"):  return "bedroom"
    if t.startswith("bathroom"): return "bathroom"
    if t in ("storage","balcony"): return "storage_balcony"
    return t

def touching_bounds(b1, b2, eps=1.0):
    """
    Determines whether two rooms' bounding boxes are "touching". (Coordinate-based
    approximation — for DB candidates, the actual graph edges
    (db_adjacency_pairs_from_graph) are now used first; this function is only a
    fallback approximation when there's no graph info (generated query floor plans,
    direction computation, etc.).)
    eps: tolerance for coordinate-match detection. Only allows for floating point /
    digitization noise (roughly 0.0001). Widening "touching" to actual wall-thickness
    gaps was found to incorrectly flag rooms that aren't touching in the graph
    (e.g. kitchen-bathroom), so it's kept narrow again.
    """
    shareX = min(b1[2],b2[2]) > max(b1[0],b2[0])
    shareY = min(b1[3],b2[3]) > max(b1[1],b2[1])
    return ((abs(b1[2]-b2[0])<=eps or abs(b2[2]-b1[0])<=eps) and shareY) or \
           ((abs(b1[3]-b2[1])<=eps or abs(b2[3]-b1[1])<=eps) and shareX)

def compute_room_orientations(rooms):
    """
    For each room (room key), computes whether the N/S/E/W directions are "not
    blocked (open)". Since Shape Grammar generates plans by attaching rooms to the
    north/south/east/west, if another room actually touches a given room in a
    particular direction, that direction is "blocked"; if not, it's considered an
    "open direction" that could face the outside/a wall.

    Returns: {room_key: ["N","S",...]} — the directions here mean only "not blocked
    (= selectable)" directions. So if the user checks "North" in the filter, it means
    they only want to see floor plans where nothing is attached to the north side of
    that room, so "N" must be present in the set this function produces to match.

    (The previous version computed this based on relative position to the overall
    floor plan center, which incorrectly judged a direction as "facing that way" even
    when it was actually blocked by another room.)
    """
    if not rooms:
        return {}

    orientations = {}
    for key, my_b in rooms.items():
        blocked = set()  # directions among this room's N/S/E/W that are blocked by another room
        for k2, b2 in rooms.items():
            if k2 == key:
                continue
            if not touching_bounds(my_b, b2):
                continue
            dx = (b2[0]+b2[2])/2 - (my_b[0]+my_b[2])/2
            dy = (b2[1]+b2[3])/2 - (my_b[1]+my_b[3])/2
            if abs(dx) > abs(dy):
                blocked.add("E" if dx > 0 else "W")
            else:
                blocked.add("N" if dy > 0 else "S")
        # Tag only the unblocked directions as "open"
        open_dirs = [d for d in ("N", "S", "E", "W") if d not in blocked]
        orientations[key] = open_dirs
    return orientations


def compute_room_orientations_from_pairs(rooms, norm_adj_type_pairs):
    """
    Graph-based version of compute_room_orientations().
    For DB candidates, adjacency itself is already determined from actual graph
    edges, but if direction only uses coordinate touching_bounds, it misses actual
    wall gaps wider than eps (e.g. two rooms connected via_door that are 3-4
    coordinate units apart), producing the contradictory result of "no blocked
    direction". So this reuses the graph-based type pairs (norm_adj_type_pairs)
    already used for adjacency computation, using "this type-type pair is confirmed
    adjacent in the graph" as the basis for the blocked judgment (coordinates are
    only used to determine the direction N/S/E/W).
    """
    if not rooms:
        return {}
    orientations = {}
    for key, my_b in rooms.items():
        my_t = normalize_type(key)
        blocked = set()
        for k2, b2 in rooms.items():
            if k2 == key:
                continue
            t2 = normalize_type(k2)
            if tuple(sorted([my_t, t2])) not in norm_adj_type_pairs:
                continue
            dx = (b2[0]+b2[2])/2 - (my_b[0]+my_b[2])/2
            dy = (b2[1]+b2[3])/2 - (my_b[1]+my_b[3])/2
            if abs(dx) > abs(dy):
                blocked.add("E" if dx > 0 else "W")
            else:
                blocked.add("N" if dy > 0 else "S")
        open_dirs = [d for d in ("N", "S", "E", "W") if d not in blocked]
        orientations[key] = open_dirs
    return orientations


def orient_type_vec(orientations_by_key):
    """
    Converts a {room_key: ["N","E",...]} shape (e.g. the result of
    compute_room_orientations(), or the values the user checked in the frontend
    direction filter panel) into a 28-dimensional ROOM_TYPES x N/S/E/W 0/1 vector.
    If there are multiple rooms of the same type, they're combined with OR
    (1 if any room of that type has that direction).
    """
    vec = [0.0]*(len(ROOM_TYPES)*4)
    didx = {"N":0,"S":1,"E":2,"W":3}
    for key, dirs in (orientations_by_key or {}).items():
        t = normalize_type(key)
        if t not in ROOM_TYPES:
            continue
        ti = ROOM_TYPES.index(t)
        for d in dirs:
            if d in didx:
                vec[ti*4+didx[d]] = 1.0
    return vec


def extract_fingerprint(rooms, conn_pairs=None, direction_filter=None, adj_pairs=None):
    """
    conn_pairs: {(type_a, type_b), ...} — room type pairs "actually connected by a
        door (via_door/direct)".
        - DB floor plans: extracted from via_door/direct edges in plan["graph"] (see db_conn_pairs_from_graph)
        - Generated floor plans: passed through as-is from the conn_pairs generate_floorplans built from circulation adjacency + touching
        - If None (no info), treated the same as adjacency (backward-compat fallback)
    adj_pairs: {(type_a, type_b), ...} — room type pairs that are "actually physically
        touching (adjacency/via_door/direct/via_window)". For DB floor plans, passing
        the actual graph edges is far more accurate than guessing from coordinate gaps
        (touching_bounds) (see db_adjacency_pairs_from_graph).
        If None (e.g. generated query floor plans with no graph info), computed from coordinate touching.
    direction_filter: {room_key: ["N","E",...]} — the values the user actually checked
        in the direction filter panel. If provided, the direction fingerprint uses
        exactly "the directions the user wants checked" rather than "the directions
        this floor plan is actually open on" (used only on the query side).
        If None (backward-compat/candidate floor plan), computed from the floor plan's
        actual open directions.
    adjacency = whether two rooms are "physically attached" — static structure
    connection = whether two rooms are "actually connected so you can move between
        them via a door" — functional connection
    """
    def get_type_rooms(t):
        return [(k,v) for k,v in rooms.items() if normalize_type(k)==t]
    present = {normalize_type(k) for k in rooms}
    type_vec = [1.0 if t in present else 0.0 for t in ROOM_TYPES]
    total_area = sum((b[2]-b[0])*(b[3]-b[1]) for b in rooms.values()) or 1.0
    area_map = defaultdict(float)
    for k,b in rooms.items():
        area_map[normalize_type(k)] += (b[2]-b[0])*(b[3]-b[1])
    area_vec = [area_map.get(t,0.0)/total_area for t in ROOM_TYPES]

    if adj_pairs is None:
        # Approximate via coordinate touching only when there's no graph info (e.g. generated query floor plans)
        adj_vec = []
        for a,b_type in EDGE_PAIRS:
            ra = [v for _,v in get_type_rooms(a)]
            rb = [v for _,v in get_type_rooms(b_type)]
            adj_vec.append(1.0 if ra and rb and any(touching_bounds(x,y) for x in ra for y in rb) else 0.0)
        norm_adj_pairs = None
    else:
        norm_adj_pairs = {tuple(sorted([normalize_type(a), normalize_type(b)])) for a, b in adj_pairs}
        adj_vec = [1.0 if pair in norm_adj_pairs else 0.0 for pair in EDGE_PAIRS]

    if conn_pairs is None:
        conn_vec = list(adj_vec)  # fallback only when info is missing
    else:
        norm_conn_pairs = {tuple(sorted([normalize_type(a), normalize_type(b)])) for a, b in conn_pairs}
        conn_vec = [1.0 if pair in norm_conn_pairs else 0.0 for pair in EDGE_PAIRS]

    if direction_filter is not None:
        # Query: use exactly what the user checked in the direction filter panel as the "requirement"
        dir_vec = orient_type_vec(direction_filter)
    elif norm_adj_pairs is not None:
        # DB candidates: reuse the graph-based type pairs already used for adjacency
        # computation for the direction-blocked judgment too
        # (using only coordinate touching_bounds produces the contradiction of a
        # via_door-connected room showing as "not blocked" due to the coordinate gap —
        # keep this consistent so adjacency and direction don't reference different answers)
        dir_vec = orient_type_vec(compute_room_orientations_from_pairs(rooms, norm_adj_pairs))
    else:
        # Candidates (and queries without a filter): compute from the floor plan's own actual open directions
        dir_vec = orient_type_vec(compute_room_orientations(rooms))
    return {"room_type":type_vec,"room_size":area_vec,"adjacency":adj_vec,"connection":conn_vec,"direction":dir_vec}

def cos_sim(a, b):
    """For single-pair comparison (used in the small-scale candidates search path)"""
    a,b = np.array(a,dtype=float), np.array(b,dtype=float)
    na,nb = np.linalg.norm(a), np.linalg.norm(b)
    if na<1e-8 or nb<1e-8: return 0.0
    return float(np.dot(a,b)/(na*nb))

def containment_ratio(q, c):
    """
    The ratio of how much the selected (query) floor plan's requirements
    (adjacency/connection/direction, etc. as 0-1 bit vectors) are contained in a
    DB candidate floor plan.
    score = |query intersect candidate| / |query|
    -> An asymmetric metric representing what % of the query's requirements the
       candidate satisfies (unlike cosine similarity, the candidate isn't penalized
       for having more requirements than the query).
    adjacency, connection, and direction are all 0/1 bit vector structures, so this
    applies identically to all of them.
    If no bits are set in the query (no requirements), treated as 1.0 (satisfied).
    """
    q = np.array(q, dtype=float)
    c = np.array(c, dtype=float)
    q_total = float((q > 0.5).sum())
    if q_total < 1e-8:
        return 1.0
    overlap = float(((q > 0.5) & (c > 0.5)).sum())
    return overlap / q_total

CONTAINMENT_KEYS = {"adjacency", "connection", "direction"}

def compute_similarity(fp1, fp2, weights):
    """For single-pair comparison (used in the small-scale candidates search path)"""
    total_w = sum(weights.values()) or 1.0
    w = {k:v/total_w for k,v in weights.items()}
    breakdown = {}
    for k in fp1:
        if k in CONTAINMENT_KEYS:
            breakdown[k] = containment_ratio(fp1[k], fp2[k])
        else:
            breakdown[k] = cos_sim(fp1[k], fp2[k])
    combined = sum(w.get(k,0)*v for k,v in breakdown.items())
    return combined, breakdown

FP_KEYS = ["room_type", "room_size", "adjacency", "connection", "direction"]

def batch_cos_sim(query_vec, db_matrix):
    """
    query_vec: (dim,) a single query vector
    db_matrix: (N, dim) matrix of all DB vectors
    Returns: (N,) array of cosine similarities
    """
    q = np.asarray(query_vec, dtype=float)
    qn = np.linalg.norm(q)
    db_norms = np.linalg.norm(db_matrix, axis=1)
    dots = db_matrix @ q
    denom = db_norms * qn
    sims = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 1e-8)
    return sims

def batch_containment_ratio(query_vec, db_matrix):
    """
    Vectorized (batch) version of containment_ratio().
    query_vec: (dim,) / db_matrix: (N, dim) -> Returns: (N,)
    score_i = |query intersect candidate_i| / |query|
    """
    q = np.asarray(query_vec, dtype=float) > 0.5
    q_total = float(q.sum())
    c_mask = db_matrix > 0.5
    if q_total < 1e-8:
        return np.ones(db_matrix.shape[0], dtype=float)
    overlap = (c_mask & q[None, :]).sum(axis=1).astype(float)
    return overlap / q_total

def batch_compute_similarity(query_fp, db_matrices, weights):
    """
    Computes similarity against all N DB entries in a single matrix operation (no Python loop).
    """
    total_w = sum(weights.values()) or 1.0
    w = {k: v/total_w for k, v in weights.items()}
    breakdowns = {}
    n = None
    for key in FP_KEYS:
        if key in CONTAINMENT_KEYS:
            sims = batch_containment_ratio(query_fp[key], db_matrices[key])
        else:
            sims = batch_cos_sim(query_fp[key], db_matrices[key])
        breakdowns[key] = sims
        n = sims.shape[0]
    combined = np.zeros(n, dtype=float)
    for key in FP_KEYS:
        combined += w.get(key, 0.0) * breakdowns[key]
    return combined, breakdowns

# ════════════════════════════════════════════════════════
# DB index
# ════════════════════════════════════════════════════════
_db_index = None
DB_ALLOWED = {'living','kitchen','bedroom','bathroom','balcony','storage','garden','front_door','parking','storage_balcony'}

def base_type(key):
    return re.sub(r'\d+$', '', key)

def db_conn_pairs_from_graph(plan):
    """
    Finds via_door/direct edges in the DB plan dictionary's 'graph' (networkx Graph)
    and extracts the room type pairs that are actually reachable.
    Compares by normalizing both edge endpoint nodes' type attributes via normalize_type.
    Returns an empty set if there's no graph or the format differs (treated the same
    as None by the caller, as a fallback).
    """
    g = plan.get("graph")
    if g is None:
        return None
    connected = set()
    try:
        for u, v, attr in g.edges(data=True):
            if attr.get("type") not in ("via_door", "direct"):
                continue
            ut = normalize_type(g.nodes[u].get("type", ""))
            vt = normalize_type(g.nodes[v].get("type", ""))
            if ut and vt:
                connected.add(tuple(sorted([ut, vt])))
    except Exception as e:
        print(f"[DB] graph parsing failed, falling back to adjacency: {e}")
        return None
    return connected


def db_adjacency_pairs_from_graph(plan):
    """
    Extracts adjacent room type pairs from the DB plan's graph, taking all edge
    types that mean "physically touching" (adjacency/via_door/direct/via_window).
    This is far more accurate ground truth than guessing "touching" from coordinate
    gaps (eps).
    (Even if two rooms are actually next to each other, if there's no edge between
    them in the graph at all, they're treated as not touching — e.g. if kitchen and
    bathroom appear a few coordinate units apart but there's no edge between them in
    the graph, it means they're actually separate structures divided by a wall.)
    Returns None if there's no graph (caller falls back to coordinate-based touching_bounds).
    """
    g = plan.get("graph")
    if g is None:
        return None
    ADJ_EDGE_TYPES = ("adjacency", "via_door", "direct", "via_window")
    adjacent = set()
    try:
        for u, v, attr in g.edges(data=True):
            if attr.get("type") not in ADJ_EDGE_TYPES:
                continue
            ut = normalize_type(g.nodes[u].get("type", ""))
            vt = normalize_type(g.nodes[v].get("type", ""))
            if ut and vt:
                adjacent.add(tuple(sorted([ut, vt])))
    except Exception as e:
        print(f"[DB] graph parsing failed, falling back to coordinate-based: {e}")
        return None
    return adjacent

def build_db_index(db_path):
    """
    On DB load, the fingerprint is 1) computed once as a list, then
    2) stacked once more into an (N, dim) numpy matrix for fast reuse on every search.
    -> In search (search_db), this allows comparing against all N entries with a
    single matrix operation instead of a Python loop.
    connection is computed precisely based on the via_door/direct edges in
    plan["graph"] (plans without a graph fall back to adjacency).
    """
    global _db_index
    print(f"[DB] Loading: {db_path}")
    with open(db_path,"rb") as f:
        raw = pickle.load(f)
    plans = []
    for p in raw:
        rooms = {}
        seen = {}
        for key, val in p.items():
            if not isinstance(val, BaseGeometry): continue
            if val.is_empty: continue
            bt = base_type(key)
            if bt not in DB_ALLOWED: continue
            b = val.bounds
            if (b[2]-b[0]) < 0.5 or (b[3]-b[1]) < 0.5: continue
            if bt not in seen:
                seen[bt] = 0; rooms[bt] = b
            else:
                seen[bt] += 1; rooms[f"{bt}{seen[bt]+1}"] = b
        if rooms:
            conn_pairs = db_conn_pairs_from_graph(p)  # based on via_door/direct edges, None if unavailable
            adj_pairs = db_adjacency_pairs_from_graph(p)  # based on adjacency/via_door/direct/via_window edges, None if unavailable
            plans.append({"rooms": rooms, "raw": p, "conn_pairs": conn_pairs, "adj_pairs": adj_pairs})

    no_graph_count = sum(1 for p in plans if p["conn_pairs"] is None)
    if no_graph_count:
        print(f"[DB] Warning: {no_graph_count}/{len(plans)} plans have no graph info -> falling back to coordinate touching")

    fps = [extract_fingerprint(p["rooms"], conn_pairs=p["conn_pairs"], adj_pairs=p["adj_pairs"]) for p in plans]

    # Pre-build an (N, dim) matrix per fingerprint type — so search can be handled with a single matrix operation
    db_matrices = {}
    for key in FP_KEYS:
        db_matrices[key] = np.array([fp[key] for fp in fps], dtype=float)

    _db_index = {"plans": plans, "fps": fps, "matrices": db_matrices}
    print(f"[DB] Indexed {len(plans)} plans (vectorized matrices built; connection based on via_door/direct graph)")

def search_db(query_fp, weights, top_k=5):
    """
    Computes similarity against all N DB entries with a single matrix operation,
    then extracts only the top_k.
    (Previously this looped N times in Python, creating a new small numpy array each
    time, which was slow.)
    """
    if not _db_index: return []
    combined, breakdowns = batch_compute_similarity(query_fp, _db_index["matrices"], weights)

    n = combined.shape[0]
    k = min(top_k, n)
    top_idx = np.argpartition(-combined, k-1)[:k]
    top_idx = top_idx[np.argsort(-combined[top_idx])]

    results = []
    for rank, i in enumerate(top_idx):
        i = int(i)
        breakdown = {key: float(breakdowns[key][i]) for key in FP_KEYS}
        pair_debug = {}
        for key in ("adjacency", "connection"):
            q_pairs = set(pairs_from_vec(query_fp[key]))
            c_pairs = set(pairs_from_vec(_db_index["fps"][i][key]))
            pair_debug[key] = {
                "query": sorted(q_pairs),
                "candidate": sorted(c_pairs),
                "overlap": sorted(q_pairs & c_pairs),
            }
        q_dirs = set(dirs_from_vec(query_fp["direction"]))
        c_dirs = set(dirs_from_vec(_db_index["fps"][i]["direction"]))
        pair_debug["direction"] = {
            "query": sorted(q_dirs),
            "candidate": sorted(c_dirs),
            "overlap": sorted(q_dirs & c_dirs),
        }
        results.append({
            "rank": rank+1,
            "plan_idx": i,
            "score": float(combined[i]),
            "breakdown": breakdown,
            "pair_debug": pair_debug,
            "plan": _db_index["plans"][i],
        })
    return results

# ════════════════════════════════════════════════════════
# Rendering
# ════════════════════════════════════════════════════════
def render_db_plan_png(plan_raw):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100, facecolor='#F7F3EE')
    ax.set_facecolor('#F7F3EE')
    ax.set_aspect('equal')
    ax.axis('off')
    for key in VIZ_DRAW_ORDER:
        geom = plan_raw.get(key)
        if geom is None or not isinstance(geom, BaseGeometry) or geom.is_empty: continue
        color  = VIZ_COLOR_MAP.get(key, '#CCCCCC')
        zorder = VIZ_DRAW_ORDER.index(key)
        geoms  = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        if key == 'wall':
            for geo in geoms:
                x, y = geo.exterior.xy
                ax.fill(x, y, facecolor='#2C2C2C', edgecolor='none', alpha=1.0, zorder=zorder)
                for interior in geo.interiors:
                    ix, iy = interior.xy
                    ax.fill(ix, iy, facecolor='#F7F3EE', edgecolor='none', alpha=1.0, zorder=zorder+0.1)
        else:
            ec = 'none' if key == 'land' else '#00000040'
            lw = 0   if key == 'land' else 0.6
            for geo in geoms:
                x, y = geo.exterior.xy
                ax.fill(x, y, facecolor=color, edgecolor=ec, alpha=1.0, linewidth=lw, zorder=zorder)
                for interior in geo.interiors:
                    ix, iy = interior.xy
                    ax.fill(ix, iy, facecolor='#F7F3EE', edgecolor='none', alpha=1.0, zorder=zorder+0.1)
        if key in VIZ_LABEL_MAP:
            largest = max(geoms, key=lambda g: g.area)
            ax.text(largest.centroid.x, largest.centroid.y, VIZ_LABEL_MAP[key],
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='#1A1A1A', zorder=zorder+0.3)
    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#F7F3EE', pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def render_bounds_png(rooms, highlight=False):
    """Render a generated plan (bounds dict) with matplotlib — includes labels"""
    CM = {
        'living':'#7FD1B9','kitchen':'#F5F79A',
        'bedroom1':'#7DA7D9','bedroom2':'#7DA7D9','bedroom3':'#7DA7D9','bedroom4':'#7DA7D9',
        'bathroom1':'#F28FA9','bathroom2':'#F28FA9','bathroom3':'#F28FA9',
        'storage_balcony':'#DDDDDD','garden':'#A8E6A3','front_door':'#FF6B35','default':'#CCCCCC',
    }
    LM = {
        'living':'Living','kitchen':'Kitchen',
        'bedroom1':'Bed 1','bedroom2':'Bed 2','bedroom3':'Bed 3','bedroom4':'Bed 4',
        'bathroom1':'Bath 1','bathroom2':'Bath 2','bathroom3':'Bath 3',
        'storage_balcony':'Storage','garden':'Garden','front_door':'Entry',
    }
    DPI = 150
    fig, ax = plt.subplots(figsize=(5, 5), dpi=DPI, facecolor='#F7F3EE')
    ax.set_facecolor('#F7F3EE')
    ax.set_aspect('equal')
    ax.axis('off')
    # Compute overall bounds
    all_coords = [(b[0],b[1],b[2],b[3]) for b in rooms.values()]
    total_w = max(b[2] for b in all_coords) - min(b[0] for b in all_coords) if all_coords else 1
    total_h = max(b[3] for b in all_coords) - min(b[1] for b in all_coords) if all_coords else 1
    scale = max(total_w, total_h)  # scale in coordinate units
    for rkey, b in rooms.items():
        x1,y1,x2,y2 = b
        bt = re.sub(r'\d+$','',rkey)
        color = CM.get(rkey) or CM.get(bt) or CM['default']
        label = LM.get(rkey) or LM.get(bt) or bt.capitalize()
        rect = plt.Rectangle((x1,y1),(x2-x1),(y2-y1),
                              facecolor=color, edgecolor='#00000080', linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        cx,cy = (x1+x2)/2,(y1+y2)/2
        rw, rh = (x2-x1), (y2-y1)
        # Determine font size from room size ratio (normalized by scale)
        size_ratio = min(rw, rh) / scale
        fs = max(7, min(13, size_ratio * 55))
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='#1A1A1A', zorder=3)
    ax.autoscale()
    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight', facecolor='#F7F3EE', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ════════════════════════════════════════════════════════
# Log saving
# ════════════════════════════════════════════════════════
@app.route("/api/save_log", methods=["POST"])
def api_save_log():
    data = request.json
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prolific_id = data.get("prolificId", "unknown")
    fname = f"session_log_{prolific_id}_{ts}.json"
    save_dir = data.get("save_dir", ".")
    if not os.path.exists(save_dir):
        save_dir = "."
    fpath = os.path.join(save_dir, fname)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return jsonify({"ok": True, "path": os.path.abspath(fpath)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ════════════════════════════════════════════════════════
# API
# ════════════════════════════════════════════════════════
@app.route("/api/parse", methods=["POST"])
def api_parse():
    jd = request.json
    try:
        seq, room_sizes = json_to_sequence(jd)
        return jsonify({"ok":True,"seq":seq,"room_sizes":{k:[list(s) for s in v] for k,v in room_sizes.items()}})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 400

MAX_RENDERED_PLANS = 300  # upper limit to protect rendering/transfer cost

@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    Returns all floor plans generated by Shape Grammar (with only deduplication
    applied, no rotation grouping) along with their rendering + orientation metadata.
    The frontend filters this full list using per-room N/S/E/W checkboxes.
    If there are too many floor plans (over MAX_RENDERED_PLANS), only the first ones
    are rendered, and the actual total generated count (total_generated) is reported separately.
    """
    data = request.json
    seq = data["seq"]
    room_sizes = {k:[tuple(s) for s in v] for k,v in data["room_sizes"].items()}
    try:
        plans = generate_floorplans(seq, room_sizes)
        # Remove only exact duplicates (no rotation grouping — expose all of them)
        plans = deduplicate_plans(plans)
        total_generated = len(plans)
        truncated = total_generated > MAX_RENDERED_PLANS
        plans_to_render = plans[:MAX_RENDERED_PLANS]

        plans_out = []
        for idx, p in enumerate(plans_to_render):
            rooms = p["rooms"]
            try: png = render_bounds_png(rooms)
            except: png = None
            orient = compute_room_orientations(rooms)
            # conn_pairs (set of tuple) can't be sent as JSON, so convert to a list of lists
            conn_pairs_list = [list(pair) for pair in p.get("conn_pairs", set())]
            plans_out.append({
                "plan_idx": idx,
                "rooms": rooms,
                "png": png,
                "orientations": orient,   # {room_key: ["N","E"], ...}
                "conn_pairs": conn_pairs_list,  # room type pairs actually connected via_door
            })
        return jsonify({
            "ok": True,
            "total": len(plans_out),
            "total_generated": total_generated,
            "truncated": truncated,
            "plans": plans_out,
        })
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 400

@app.route("/api/render_query", methods=["POST"])
def api_render_query():
    rooms = request.json["rooms"]
    try:
        png = render_bounds_png(rooms)
        return jsonify({"ok":True,"png":png})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 400

@app.route("/api/search", methods=["POST"])
def api_search():
    """
    query_conn_pairs / candidates[].conn_pairs: also receives via_door-based
    connection info in the format [[type_a, type_b], ...]. If missing (backward
    compat), falls back to adjacency.
    """
    data = request.json
    query_rooms = data["query_rooms"]
    weights     = data["weights"]
    top_k       = data.get("top_k", 5)
    candidates  = data.get("candidates", [])
    query_conn_pairs = data.get("query_conn_pairs")  # [[a,b], ...] or None
    query_conn_set = {tuple(p) for p in query_conn_pairs} if query_conn_pairs else None
    # direction_filters: {room_key: ["N","E",...]} — values the user actually checked
    # in the direction filter panel. If present, the "query side" of the direction
    # fingerprint uses this filter as-is; if absent (filter untouched), it's computed
    # from the selected floor plan's own actual open directions.
    direction_filters = data.get("direction_filters")
    query_fp = extract_fingerprint(query_rooms, conn_pairs=query_conn_set, direction_filter=direction_filters)

    if _db_index:
        results = search_db(query_fp, weights, top_k)
        res_out = []
        for r in results:
            rooms_serial = {k: list(v) for k, v in r["plan"]["rooms"].items()}
            try: png_b64 = render_db_plan_png(r["plan"]["raw"])
            except Exception as e:
                print(f"[render error] {e}"); png_b64 = None
            res_out.append({
                "rank":r["rank"],"plan_idx":r["plan_idx"],
                "score":r["score"],"breakdown":r["breakdown"],
                "pair_debug":r["pair_debug"],
                "rooms":rooms_serial,"png":png_b64,
            })
        return jsonify({"ok":True,"results":res_out,"source":"db"})
    else:
        results = []
        for c in candidates:
            c_conn_pairs = {tuple(p) for p in c.get("conn_pairs", [])} if c.get("conn_pairs") else None
            fp = extract_fingerprint(c["rooms"], conn_pairs=c_conn_pairs)
            score, breakdown = compute_similarity(query_fp, fp, weights)
            pair_debug = {}
            for key in ("adjacency", "connection"):
                q_pairs = set(pairs_from_vec(query_fp[key]))
                c_pairs = set(pairs_from_vec(fp[key]))
                pair_debug[key] = {
                    "query": sorted(q_pairs),
                    "candidate": sorted(c_pairs),
                    "overlap": sorted(q_pairs & c_pairs),
                }
            q_dirs = set(dirs_from_vec(query_fp["direction"]))
            c_dirs = set(dirs_from_vec(fp["direction"]))
            pair_debug["direction"] = {
                "query": sorted(q_dirs),
                "candidate": sorted(c_dirs),
                "overlap": sorted(q_dirs & c_dirs),
            }
            results.append({"plan_idx":c["idx"],"score":score,"breakdown":breakdown,"pair_debug":pair_debug,"rooms":c["rooms"]})
        results.sort(key=lambda x:-x["score"])
        for i,r in enumerate(results[:top_k]):
            r["rank"]=i+1
            try: r["png"] = render_bounds_png(r["rooms"])
            except: r["png"] = None
        return jsonify({"ok":True,"results":results[:top_k],"source":"generated"})

@app.route("/api/load_db", methods=["POST"])
def api_load_db():
    path = request.json.get("path","")
    if not os.path.exists(path):
        return jsonify({"ok":False,"error":f"File not found: {path}"}), 400
    try:
        build_db_index(path)
        return jsonify({"ok":True,"count":len(_db_index["plans"])})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 400

@app.route("/api/db_status")
def api_db_status():
    if _db_index:
        return jsonify({"loaded":True,"count":len(_db_index["plans"])})
    return jsonify({"loaded":False})

# ════════════════════════════════════════════════════════
# Frontend
# ════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Floor Plan Recommendation System</title>
<style>
:root{
  --bg:#0d0d0f;--bg2:#141417;--bg3:#1c1c21;--bg4:#232329;
  --border:rgba(255,255,255,0.07);--border2:rgba(255,255,255,0.13);
  --text:#f0ede8;--text2:#9a9590;--text3:#5c5955;
  --accent:#a78bfa;--accent2:#7c3aed;
  --teal:#5eead4;--amber:#fbbf24;--coral:#fb7185;--blue:#60a5fa;
  --font:'DM Sans',system-ui,sans-serif;--mono:'JetBrains Mono','Fira Code',monospace;
  --r:10px;--rl:16px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:14px;min-height:100vh}
.layout{display:grid;grid-template-columns:270px 1fr;grid-template-rows:52px 1fr;grid-template-areas:"hd hd""sb main";height:100vh;overflow:hidden}
header{grid-area:hd;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 24px;gap:12px;z-index:10}
.logo{font-size:15px;font-weight:500;letter-spacing:-.02em;display:flex;align-items:center;gap:8px}
.logo-mark{width:26px;height:26px;background:var(--accent2);border-radius:7px;display:grid;grid-template-columns:1fr 1fr;gap:3px;padding:5px}
.logo-mark div{background:rgba(255,255,255,.55);border-radius:2px}
.logo-mark div:first-child{background:rgba(255,255,255,.9)}
.steps-nav{margin-left:auto;display:flex;gap:3px}
.step-pill{padding:5px 13px;border-radius:20px;font-size:12px;color:var(--text3);cursor:pointer;transition:all .2s;border:1px solid transparent}
.step-pill.active{background:var(--bg4);border-color:var(--border2);color:var(--text)}
.step-pill.done{color:var(--teal)}
aside{grid-area:sb;background:var(--bg2);border-right:1px solid var(--border);padding:18px 14px;overflow-y:auto;display:flex;flex-direction:column;gap:18px}
.sb-sec h4{font-size:10px;font-weight:500;letter-spacing:.1em;text-transform:uppercase;color:var(--text3);margin-bottom:10px}
main{grid-area:main;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:var(--rl);padding:18px}
.pt{font-size:13px;font-weight:500;color:var(--text2);margin-bottom:14px;display:flex;align-items:center;gap:8px}
.badge{background:var(--bg4);border:1px solid var(--border);color:var(--text3);font-size:10px;padding:2px 8px;border-radius:20px;font-family:var(--mono)}
textarea{width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);color:var(--text);font-family:var(--mono);font-size:11.5px;padding:11px 13px;resize:vertical;min-height:140px;outline:none;transition:border-color .2s;line-height:1.6}
textarea:focus{border-color:var(--accent)}
.btn{padding:7px 16px;border-radius:var(--r);border:1px solid var(--border2);background:var(--bg3);color:var(--text);font-size:13px;cursor:pointer;transition:all .15s;display:inline-flex;align-items:center;gap:6px}
.btn:hover{background:var(--bg4);border-color:rgba(255,255,255,.2)}
.btn.primary{background:var(--accent2);border-color:transparent;color:#fff}
.btn.primary:hover{background:#6d28d9}
.btn.danger{background:rgba(251,113,133,.15);border-color:rgba(251,113,133,.3);color:var(--coral)}
.btn.sm{padding:5px 11px;font-size:12px}
.btn:disabled{opacity:.4;cursor:not-allowed;pointer-events:none}
/* File upload */
.file-drop{border:1.5px dashed var(--border2);border-radius:var(--r);padding:18px;text-align:center;cursor:pointer;transition:all .2s;background:var(--bg3)}
.file-drop:hover,.file-drop.dragover{border-color:var(--accent);background:rgba(124,58,237,.07)}
.file-drop input[type=file]{display:none}
.file-drop-label{font-size:12px;color:var(--text3);margin-top:6px}
/* Floor plan grid (single step) */
.group-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.group-card{background:var(--bg3);border:1.5px solid var(--border);border-radius:var(--r);overflow:hidden;cursor:pointer;transition:all .2s;position:relative}
.group-card:hover{border-color:var(--border2);transform:translateY(-2px)}
.group-card.selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.group-card img{width:100%;display:block}
.group-footer{padding:7px 9px;border-top:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;font-size:11px;color:var(--text3);font-family:var(--mono)}
.sel-badge{position:absolute;top:6px;right:6px;background:var(--accent);color:#fff;font-size:10px;padding:2px 7px;border-radius:20px;display:none}
.group-card.selected .sel-badge{display:block}
.group-card.hidden-by-filter{display:none}
/* Direction filter panel */
.orient-filter-panel{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:14px 16px;margin-bottom:14px}
.orient-filter-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.orient-filter-head .pt{margin-bottom:0}
.orient-room-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
.orient-room-card{background:var(--bg4);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.orient-room-name{display:flex;align-items:center;gap:7px;font-size:12px;font-weight:500;color:var(--text);margin-bottom:8px}
.orient-room-name .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.orient-checks{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.orient-check{display:flex;align-items:center;gap:5px;font-size:11.5px;color:var(--text2);cursor:pointer;padding:3px 6px;border-radius:5px;transition:background .15s}
.orient-check:hover{background:var(--bg3)}
.orient-check input{accent-color:var(--accent);cursor:pointer;width:13px;height:13px}
.filter-count-badge{font-size:11px;color:var(--accent);font-family:var(--mono)}
/* Results */
.results-list{display:flex;flex-direction:column;gap:9px}
.rc{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:13px 15px;transition:all .15s}
.rc.top{border-color:rgba(167,139,250,.3)}
.rc.rated{border-color:rgba(94,234,212,.3)}
.rc-row{display:grid;grid-template-columns:34px 1fr auto;gap:11px;align-items:start}
.rn{width:34px;height:34px;border-radius:8px;background:var(--bg4);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:500;font-family:var(--mono);color:var(--text2)}
.rn.top{background:rgba(167,139,250,.15);color:var(--accent)}
.sbar{width:100%;height:3px;background:var(--bg4);border-radius:2px;margin-top:5px;overflow:hidden}
.sfill{height:100%;background:var(--accent);border-radius:2px;transition:width .6s}
.rmeta{font-size:11px;color:var(--text3);font-family:var(--mono);white-space:nowrap}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px;padding-top:9px;border-top:1px solid var(--border)}
.chip{padding:3px 9px;border-radius:20px;font-size:11px;background:var(--bg4);color:var(--text2);font-family:var(--mono)}
.chip.hi{background:rgba(167,139,250,.15);color:var(--accent)}
/* Rating */
.score-row{display:flex;align-items:center;gap:8px;margin-top:10px;padding-top:10px;border-top:1px solid var(--border);flex-wrap:wrap}
.score-row span{font-size:11px;color:var(--text3);white-space:nowrap}
.score-btns{display:flex;gap:4px}
.sc-btn{width:28px;height:28px;border-radius:6px;border:1px solid var(--border2);background:var(--bg4);color:var(--text2);font-size:12px;font-family:var(--mono);cursor:pointer;transition:all .15s;display:flex;align-items:center;justify-content:center;font-weight:500}
.sc-btn:hover{border-color:var(--amber);color:var(--amber)}
.sc-btn.on{background:var(--amber);border-color:var(--amber);color:#111;font-weight:700}
.score-val{font-size:13px;font-family:var(--mono);color:var(--amber);min-width:36px;font-weight:600}
.memo-row{margin-top:8px;display:flex;flex-direction:column;gap:4px}
.memo-row label{font-size:11px;color:var(--text3)}
.memo-input{width:100%;box-sizing:border-box;resize:vertical;min-height:38px;padding:7px 9px;border-radius:7px;border:1px solid var(--border2);background:var(--bg4);color:var(--text2);font-size:12px;font-family:inherit;line-height:1.4}
.memo-input:focus{outline:none;border-color:var(--amber)}
.memo-input::placeholder{color:var(--text3);opacity:.7}
/* Other */
.ws{display:flex;flex-direction:column;gap:12px}
.wr{display:flex;flex-direction:column;gap:5px}
.wh{display:flex;justify-content:space-between;align-items:center}
.wn{font-size:12px;color:var(--text2);display:flex;align-items:center;gap:6px}
.wv{font-size:12px;font-family:var(--mono);color:var(--accent)}
input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:var(--bg4);border-radius:2px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:var(--accent);cursor:pointer}
.wt{font-size:11px;color:var(--text3);font-family:var(--mono)}
.wt.ok{color:var(--teal)}.wt.warn{color:var(--amber)}
.log{background:var(--bg);border-radius:var(--r);padding:11px 13px;font-family:var(--mono);font-size:11px;color:var(--text3);border:1px solid var(--border);max-height:120px;overflow-y:auto;line-height:1.9}
.lok{color:var(--teal)}.lwarn{color:var(--amber)}.linfo{color:var(--blue)}
.ibar{display:flex;gap:14px;flex-wrap:wrap;padding:11px 14px;background:var(--bg3);border-radius:var(--r);border:1px solid var(--border)}
.ii{display:flex;flex-direction:column;gap:2px}
.il{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.08em}
.iv{font-size:13px;font-weight:500;font-family:var(--mono)}
.notice{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.2);border-radius:var(--r);padding:11px 14px;font-size:12px;color:var(--amber);line-height:1.6}
.dbloaded{background:rgba(94,234,212,.07);border:1px solid rgba(94,234,212,.2);border-radius:var(--r);padding:11px 14px;font-size:12px;color:var(--teal);line-height:1.6}
.spinner{width:18px;height:18px;border:2px solid var(--border2);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.lrow{display:flex;align-items:center;gap:9px;color:var(--text3);font-size:12px;padding:14px 0}
.fade{animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:10px}
.chip-grp{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.divider{height:1px;background:var(--border);margin:14px 0}
.finish-bar{background:rgba(94,234,212,.07);border:1px solid rgba(94,234,212,.2);border-radius:var(--r);padding:14px 16px;display:flex;align-items:center;justify-content:space-between;gap:12px}
.finish-bar p{font-size:12px;color:var(--teal)}
</style>
</head>
<body>
<div class="layout">
<header>
  <div class="logo">
    <div class="logo-mark"><div></div><div></div><div></div><div></div></div>
    Floor Plan Recommendation System
  </div>
  <div class="steps-nav">
    <div class="step-pill active" id="pill-1" onclick="goStep(1)">1 · Input</div>
    <div class="step-pill" id="pill-2" onclick="goStep(2)">2 · Generate</div>
    <div class="step-pill" id="pill-3" onclick="goStep(3)">3 · Select</div>
    <div class="step-pill" id="pill-4" onclick="goStep(4)">4 · Match</div>
  </div>
</header>

<aside>
  <div class="sb-sec">
    <h4>Room Legend</h4>
    <div id="legend" style="display:flex;flex-direction:column;gap:5px"></div>
  </div>
  <div class="sb-sec">
    <h4>Similarity Weights</h4>
    <div class="ws" id="weight-sliders"></div>
    <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border)">
      <div class="wt" id="wtotal">Total: 1.00</div>
    </div>
    <button class="btn sm" style="margin-top:9px;width:100%" onclick="normalizeWeights()">Normalize to 1.00</button>
  </div>
  <div class="sb-sec">
    <h4>DB Connection</h4>
    <div id="db-status-box" class="notice">DB not connected — searching within generated plans</div>
    <div style="margin-top:8px;display:flex;gap:6px">
      <input id="db-path-input" style="flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);color:var(--text);font-size:11px;padding:5px 8px;outline:none" placeholder="/path/to/ResPlan.pkl">
      <button class="btn sm" onclick="loadDB()">Connect</button>
    </div>
  </div>
  <div class="sb-sec">
    <h4>Log Save Directory</h4>
    <input id="save-dir-input" style="width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);color:var(--text);font-size:11px;padding:5px 8px;outline:none" placeholder="/Users/hongyoonjae/Downloads" value="/Users/hongyoonjae/Downloads">
  </div>
</aside>

<main>

<!-- ── STEP 1: Input ── -->
<div id="step-1" class="fade" style="display:flex;flex-direction:column;gap:16px">
  <div class="panel">
    <div class="pt">Activity JSON Input <span class="badge">Step 1</span></div>
    <!-- File upload -->
    <div class="file-drop" id="file-drop" onclick="document.getElementById('file-input').click()"
         ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
      <input type="file" id="file-input" accept=".json" onchange="onFileSelect(event)">
      <div style="font-size:28px;margin-bottom:4px">📂</div>
      <div style="font-size:13px;color:var(--text2);font-weight:500">Select or drag & drop a JSON file</div>
      <div class="file-drop-label" id="file-name">Click to select a file</div>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin:10px 0;color:var(--text3);font-size:11px">
      <div style="flex:1;height:1px;background:var(--border)"></div>or paste directly<div style="flex:1;height:1px;background:var(--border)"></div>
    </div>
    <textarea id="json-input" placeholder='{"prolificId":"test","entries":[...]}'></textarea>
    <div style="display:flex;gap:8px;margin-top:10px;align-items:center">
      <button class="btn" onclick="loadSample()">Load sample</button>
      <button class="btn primary" onclick="parseAndGenerate()">▶ Parse &amp; Generate</button>
      <span style="font-size:11px;color:var(--text3);margin-left:4px" id="jstatus"></span>
    </div>
  </div>
  <div class="panel" id="preview-panel" style="display:none">
    <div class="pt">Parsed Activity</div>
    <div id="act-chips" class="chip-grp"></div>
    <div style="margin-top:12px" id="room-seq"></div>
  </div>
</div>

<!-- ── STEP 2: Generate ── -->
<div id="step-2" style="display:none;flex-direction:column;gap:16px">
  <div class="panel">
    <div class="pt">Shape Grammar Generation <span class="badge">Step 2</span></div>
    <div class="log" id="gen-log"><span class="linfo">Waiting...</span></div>
    <div class="lrow" id="gen-spin" style="display:none"><div class="spinner"></div>Generating...</div>
  </div>

  <!-- Direction filter panel -->
  <div class="panel orient-filter-panel" id="orient-filter-panel" style="display:none">
    <div class="orient-filter-head">
      <div class="pt">Direction Filter (North / South / East / West)</div>
      <div style="display:flex;align-items:center;gap:10px">
        <span class="filter-count-badge" id="filter-count-badge"></span>
        <button class="btn sm" onclick="clearOrientFilters()">Clear all</button>
      </div>
    </div>
    <p style="font-size:12px;color:var(--text2);margin-bottom:12px">Checking a direction for a room means "no other room is attached in that direction (open direction)". For example, checking North for the bathroom shows only floor plans where nothing blocks the bathroom's north side. (Multiple selections use an OR condition.)</p>
    <div class="orient-room-grid" id="orient-room-grid"></div>
  </div>

  <div class="panel" id="gen-panel" style="display:none">
    <div class="pt" id="gen-title">Generated Floor Plans</div>
    <p style="font-size:12px;color:var(--text2);margin-bottom:14px">Click a card to select the floor plan to use for the search.</p>
    <div class="group-grid" id="group-grid"></div>
  </div>
  <div style="display:flex;justify-content:flex-end;gap:8px" id="step2-actions" style="display:none">
    <button class="btn primary" id="btn-to-search" onclick="confirmVariantAndSearch()" disabled>Similarity Search →</button>
  </div>
</div>

<!-- ── STEP 3: (formerly select — now merged into step2, empty passthrough) ── -->
<div id="step-3" style="display:none;flex-direction:column;gap:16px">
  <div class="panel">
    <div class="pt">Selected Query Floor Plan <span class="badge">Step 3 · Match</span></div>
    <div class="ibar" id="qinfo"></div>
    <div style="margin-top:14px" id="qvis"></div>
  </div>
  <div class="panel">
    <div class="pt" id="result-title">Similarity Search Results</div>
    <div class="results-list" id="results-list"></div>
  </div>
  <div class="finish-bar" id="finish-bar" style="display:none">
    <p>After finishing your evaluation, click Finish to save the log.</p>
    <button class="btn primary" onclick="saveLog()">✅ Finish</button>
  </div>
</div>

<!-- step-4 is unused (merged into step-3) -->
<div id="step-4" style="display:none"></div>

</main>
</div>

<script>
// ── Constants ──
const CM={living:'#7FD1B9',kitchen:'#F5F79A',bedroom1:'#7DA7D9',bedroom2:'#7DA7D9',bedroom3:'#7DA7D9',bedroom4:'#7DA7D9',bathroom1:'#F28FA9',bathroom2:'#F28FA9',bathroom3:'#F28FA9',storage_balcony:'#DDDDDD',garden:'#A8E6A3',front_door:'#FF6B35',default:'#6b7280'};
const RL={living:'Living',kitchen:'Kitchen',bedroom1:'Bed 1',bedroom2:'Bed 2',bedroom3:'Bed 3',bedroom4:'Bed 4',bathroom1:'Bath 1',bathroom2:'Bath 2',bathroom3:'Bath 3',storage_balcony:'Storage',garden:'Garden',front_door:'Entry'};
const FPS=[
  {id:'room_type',name:'Room Type',color:'#a78bfa'},
  {id:'room_size',name:'Room Size',color:'#5eead4'},
  {id:'adjacency',name:'Adjacency',color:'#fbbf24'},
  {id:'connection',name:'Connection',color:'#fb7185'},
  {id:'direction',name:'Direction',color:'#60a5fa'},
];
let weights={room_type:.2,room_size:.2,adjacency:.2,connection:.2,direction:.2};
let allPlans=[];            // all floor plans returned by /api/generate (deduplicated only, no grouping)
let selPlanIdx=null, selVariantRooms=null, searchResults=[];
let orientFilters={};       // {roomKey: Set(["N","S","E","W"])} — contains only checked directions
let parsedData=null;        // the entire parsed JSON

// ── Session log ──
const sessionLog = {
  prolificId: null,
  sessionStart: new Date().toISOString(),
  steps: [],
};
function logEvent(type, data){
  sessionLog.steps.push({type, timestamp: new Date().toISOString(), ...data});
}

// ── Legend ──
document.getElementById('legend').innerHTML=[['living','Living'],['kitchen','Kitchen'],['bedroom1','Bedroom'],['bathroom1','Bathroom'],['front_door','Entry'],['storage_balcony','Storage'],['garden','Garden']].map(([t,l])=>`<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text2)"><div style="width:8px;height:8px;border-radius:50%;background:${CM[t]}"></div>${l}</div>`).join('');

// ── Weights ──
function buildSliders(){
  document.getElementById('weight-sliders').innerHTML=FPS.map(fp=>`
    <div class="wr">
      <div class="wh">
        <span class="wn"><span style="width:7px;height:7px;border-radius:50%;background:${fp.color};display:inline-block"></span>${fp.name}</span>
        <span class="wv" id="wv-${fp.id}">${(weights[fp.id]*100).toFixed(0)}%</span>
      </div>
      <input type="range" min="0" max="100" step="1" value="${Math.round(weights[fp.id]*100)}" oninput="onW('${fp.id}',this.value)">
    </div>`).join('');
  updateTotal();
}
function onW(id,v){weights[id]=parseInt(v)/100;document.getElementById('wv-'+id).textContent=parseInt(v)+'%';updateTotal()}
function updateTotal(){const t=Object.values(weights).reduce((a,b)=>a+b,0);const el=document.getElementById('wtotal');el.textContent='Total: '+t.toFixed(2);el.className='wt '+(Math.abs(t-1)<.02?'ok':'warn')}
function normalizeWeights(){const t=Object.values(weights).reduce((a,b)=>a+b,0)||1;for(const k in weights)weights[k]/=t;buildSliders()}
buildSliders();

// ── Step nav ──
function goStep(n){
  for(let i=1;i<=4;i++){
    const el=document.getElementById('step-'+i);
    el.style.display=i===n?'flex':'none';
    if(i===n)el.style.flexDirection='column';
    document.getElementById('pill-'+i).className='step-pill'+(i===n?' active':i<n?' done':'');
  }
}

// ── DB ──
async function loadDB(){
  const path=document.getElementById('db-path-input').value.trim();
  if(!path)return;
  const r=await fetch('/api/load_db',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});
  const d=await r.json();
  const box=document.getElementById('db-status-box');
  if(d.ok){box.className='dbloaded';box.textContent=`✓ DB connected — ${d.count.toLocaleString()} plans`}
  else{box.className='notice';box.textContent='❌ '+d.error}
}
fetch('/api/db_status').then(r=>r.json()).then(d=>{
  if(d.loaded){document.getElementById('db-status-box').className='dbloaded';document.getElementById('db-status-box').textContent=`✓ DB connected — ${d.count.toLocaleString()} plans`}
});

// ── File upload ──
function onDragOver(e){e.preventDefault();document.getElementById('file-drop').classList.add('dragover')}
function onDragLeave(e){document.getElementById('file-drop').classList.remove('dragover')}
function onDrop(e){
  e.preventDefault();
  document.getElementById('file-drop').classList.remove('dragover');
  const file=e.dataTransfer.files[0];
  if(file) readJsonFile(file);
}
function onFileSelect(e){
  const file=e.target.files[0];
  if(file) readJsonFile(file);
}
function readJsonFile(file){
  document.getElementById('file-name').textContent=file.name;
  const reader=new FileReader();
  reader.onload=ev=>{
    document.getElementById('json-input').value=ev.target.result;
    document.getElementById('jstatus').textContent='File loaded: '+file.name;
  };
  reader.readAsText(file);
}

// ── Sample ──
function loadSample(){
  const s={prolificId:"test_user",entries:[
    {startTime:"07:00",mainActivity:16,where:"front_door",spatialPreferences:{roomType:"front_door"},peopleCount:1},
    {startTime:"08:00",mainActivity:1,where:"living_room",spatialPreferences:{roomType:"living_room"},peopleCount:4},
    {startTime:"09:00",mainActivity:5,where:"kitchen",spatialPreferences:{roomType:"kitchen_dining"},peopleCount:3},
    {startTime:"10:00",mainActivity:2,where:"bedroom1",spatialPreferences:{roomType:"bedroom1"},peopleCount:2},
    {startTime:"11:00",mainActivity:3,where:"bathroom1",spatialPreferences:{roomType:"bathroom1"},peopleCount:1},
    {startTime:"14:00",mainActivity:1,where:"living_room",spatialPreferences:{roomType:"living_room"},peopleCount:4},
    {startTime:"20:00",mainActivity:2,where:"bedroom1",spatialPreferences:{roomType:"bedroom1"},peopleCount:2},
  ]};
  document.getElementById('json-input').value=JSON.stringify(s,null,2);
  document.getElementById('jstatus').textContent='Sample loaded';
  document.getElementById('file-name').textContent='Click to select a file';
}

// ── Parse & Generate ──
async function parseAndGenerate(){
  const raw=document.getElementById('json-input').value.trim();
  if(!raw)return;
  let jd;
  try{jd=JSON.parse(raw)}catch(e){document.getElementById('jstatus').textContent='❌ JSON error: '+e.message;return}
  parsedData=jd;
  sessionLog.prolificId=jd.prolificId||'unknown';

  const pr=await fetch('/api/parse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(jd)});
  const pd=await pr.json();
  if(!pd.ok){document.getElementById('jstatus').textContent='❌ '+pd.error;return}
  document.getElementById('jstatus').textContent=`✓ ${pd.seq.length} activities parsed`;
  document.getElementById('act-chips').innerHTML=pd.seq.map(s=>`<div style="display:inline-flex;align-items:center;gap:5px;padding:4px 9px;border-radius:6px;font-size:11.5px;background:var(--bg3);border:1px solid var(--border);color:var(--text2);margin:2px"><div style="width:7px;height:7px;border-radius:50%;background:${CM[s.room]||'#666'}"></div><span style="color:var(--text3);font-size:10px">${s.startTime}</span> ${RL[s.room]||s.room}</div>`).join('');
  const ur=[...new Set(pd.seq.map(s=>s.room))];
  document.getElementById('room-seq').innerHTML=`<div style="font-size:11px;color:var(--text3);margin-bottom:5px">Room Types (${ur.length})</div><div class="chip-grp">${ur.map(r=>`<div class="chip">${RL[r]||r}</div>`).join('')}</div>`;
  document.getElementById('preview-panel').style.display='block';

  logEvent('parse',{prolificId:sessionLog.prolificId, actCount:pd.seq.length, rooms:ur});

  goStep(2);
  const logEl=document.getElementById('gen-log');
  const spinEl=document.getElementById('gen-spin');
  logEl.innerHTML=''; spinEl.style.display='flex';
  function log(m,c=''){logEl.innerHTML+=`<span class="${c}">${m}</span>\n`;logEl.scrollTop=9999}
  log(`Rooms to place: ${ur.join(', ')}`,'linfo');

  const genStart=Date.now();
  const gr=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seq:pd.seq,room_sizes:pd.room_sizes})});
  const gd=await gr.json();
  spinEl.style.display='none';
  if(!gd.ok){log('❌ '+gd.error,'lwarn');return}

  allPlans=gd.plans;
  const genMs=Date.now()-genStart;
  if(gd.truncated){
    log(`✓ ${gd.total_generated} generated after deduplication → showing only the top ${gd.total} to limit rendering cost.`,'lwarn');
  } else {
    log(`✓ ${gd.total} floor plans generated in total (deduplication complete)`,'lok');
  }
  logEvent('generate',{totalPlans:gd.total, totalGenerated:gd.total_generated, truncated:gd.truncated, durationMs:genMs});

  document.getElementById('gen-title').innerHTML=`Generated Floor Plans <span class="badge">${gd.total}</span>`;
  buildOrientFilterPanel();
  renderGroupGrid();
  document.getElementById('orient-filter-panel').style.display='block';
  document.getElementById('gen-panel').style.display='block';
  document.getElementById('step2-actions').style.display='flex';
  selPlanIdx=null; selVariantRooms=null;
  document.getElementById('btn-to-search').disabled=true;
}

// ── Build direction filter panel ──
// Room type display order: living -> kitchen -> bedroom -> bathroom -> front_door -> other
const ROOM_ORDER_PREFIX=['living','kitchen','bedroom','bathroom','front_door'];
function roomSortKey(key){
  const idx=ROOM_ORDER_PREFIX.findIndex(p=>key.startsWith(p));
  const groupRank=idx===-1?ROOM_ORDER_PREFIX.length:idx;
  return [groupRank, key]; // secondary sort by key name within the same group (bedroom1 < bedroom2, etc.)
}

function buildOrientFilterPanel(){
  // Collect every room key that appears across all currently generated floor plans
  const roomKeys=new Set();
  allPlans.forEach(p=>Object.keys(p.rooms).forEach(k=>roomKeys.add(k)));
  const sortedKeys=[...roomKeys].sort((a,b)=>{
    const [ga,ka]=roomSortKey(a), [gb,kb]=roomSortKey(b);
    if(ga!==gb) return ga-gb;
    return ka.localeCompare(kb);
  });
  orientFilters={};
  sortedKeys.forEach(k=>{orientFilters[k]=new Set()});

  document.getElementById('orient-room-grid').innerHTML=sortedKeys.map(k=>{
    const color=CM[k]||CM.default;
    const label=RL[k]||k;
    return `<div class="orient-room-card">
      <div class="orient-room-name"><span class="dot" style="background:${color}"></span>${label}</div>
      <div class="orient-checks">
        ${['N','S','E','W'].map(d=>`
          <label class="orient-check">
            <input type="checkbox" onchange="toggleOrientFilter('${k}','${d}',this.checked)">
            ${ {N:'North',S:'South',E:'East',W:'West'}[d] }
          </label>`).join('')}
      </div>
    </div>`;
  }).join('');
  updateFilterCountBadge();
}

function toggleOrientFilter(roomKey, dir, checked){
  if(!orientFilters[roomKey]) orientFilters[roomKey]=new Set();
  if(checked) orientFilters[roomKey].add(dir);
  else orientFilters[roomKey].delete(dir);
  renderGroupGrid();
  updateFilterCountBadge();
}

function clearOrientFilters(){
  Object.keys(orientFilters).forEach(k=>orientFilters[k].clear());
  document.querySelectorAll('.orient-check input').forEach(el=>el.checked=false);
  renderGroupGrid();
  updateFilterCountBadge();
}

// Checks whether floor plan p satisfies the current filter conditions.
// Rule: for a room type that has any direction checked, that room's orientation
//       tags must include at least one checked direction to pass (OR). Rooms with
//       nothing checked have no requirement (pass).
function planMatchesFilters(plan){
  for(const roomKey in orientFilters){
    const wanted=orientFilters[roomKey];
    if(!wanted || wanted.size===0) continue; // no filtering for this room
    const tags=plan.orientations[roomKey];
    if(!tags) return false; // this floor plan doesn't have that room -> exclude
    const matches=tags.some(t=>wanted.has(t));
    if(!matches) return false;
  }
  return true;
}

function updateFilterCountBadge(){
  const visible=allPlans.filter(planMatchesFilters).length;
  const anyActive=Object.values(orientFilters).some(s=>s.size>0);
  const badge=document.getElementById('filter-count-badge');
  badge.textContent = anyActive ? `Showing ${visible} / ${allPlans.length}` : `All ${allPlans.length}`;
}

// ── Floor plan grid (single step) ──
function renderGroupGrid(){
  document.getElementById('group-grid').innerHTML=allPlans.map((p)=>{
    const visible=planMatchesFilters(p);
    return `<div class="group-card ${selPlanIdx===p.plan_idx?'selected':''} ${visible?'':'hidden-by-filter'}" onclick="selectPlan(${p.plan_idx})">
      <div class="sel-badge">Selected</div>
      <img src="data:image/png;base64,${p.png}" alt="plan ${p.plan_idx}">
      <div class="group-footer">
        <span>plan_${p.plan_idx}</span>
        <span>${Object.keys(p.rooms).length}r</span>
      </div>
    </div>`;
  }).join('');
  updateFilterCountBadge();
}

// ── Floor plan selection ──
let selConnPairs=null; // via_door connection pairs of the selected floor plan [[a,b],...]
function selectPlan(planIdx){
  const plan=allPlans.find(p=>p.plan_idx===planIdx);
  if(!plan) return;
  selPlanIdx=planIdx;
  selVariantRooms=plan.rooms;
  selConnPairs=plan.conn_pairs||[];
  renderGroupGrid();
  document.getElementById('btn-to-search').disabled=false;
  logEvent('select_plan',{planIdx});
}

// ── Run search ──
async function confirmVariantAndSearch(){
  if(selVariantRooms===null)return;
  // Before starting a new search, save the current session if it has ratings (uses the query recorded in the current searchResults)
  commitCurrentSession();
  currentQueryRooms = selVariantRooms;
  currentQueryMeta  = {planIdx:selPlanIdx, weights:{...weights}};
  goStep(3);
  const qrooms=selVariantRooms;
  const totalArea=Object.values(qrooms).reduce((s,b)=>s+(b[2]-b[0])*(b[3]-b[1]),0);
  document.getElementById('qinfo').innerHTML=[
    ['Selected Plan',`plan_${selPlanIdx}`],
    ['Room Count',Object.keys(qrooms).length+''],
    ['Area',totalArea.toFixed(0)+' u²'],
    ['Types',[...new Set(Object.keys(qrooms).map(k=>RL[k]||k))].join(', ')],
  ].map(([l,v])=>`<div class="ii"><div class="il">${l}</div><div class="iv">${v}</div></div>`).join('');

  document.getElementById('qvis').innerHTML='<div class="lrow"><div class="spinner"></div>Rendering...</div>';
  fetch('/api/render_query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rooms:qrooms})})
    .then(r=>r.json()).then(d=>{
      document.getElementById('qvis').innerHTML=d.ok
        ?`<img src="data:image/png;base64,${d.png}" style="max-width:300px;width:100%;border-radius:10px;display:block">`
        :'<span style="color:var(--text3);font-size:12px">Rendering failed</span>';
    });

  document.getElementById('result-title').innerHTML='Similarity Search Results <span class="badge">Searching...</span>';
  document.getElementById('results-list').innerHTML='<div class="lrow"><div class="spinner"></div>Searching...</div>';

  // Fallback candidates when the DB isn't connected: floor plans from allPlans other than the query
  // conn_pairs must also be sent so the connection (via_door-based) score is computed differently from adjacency
  const allCandidates=allPlans.filter(p=>p.plan_idx!==selPlanIdx).map(p=>({rooms:p.rooms, idx:p.plan_idx, conn_pairs:p.conn_pairs||[]}));
  const searchStart=Date.now();
  const directionFiltersOut = {};
  Object.entries(orientFilters||{}).forEach(([k,set])=>{
    if(set && set.size>0) directionFiltersOut[k] = [...set];
  });
  const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query_rooms:qrooms,weights,top_k:5,candidates:allCandidates,query_conn_pairs:selConnPairs,direction_filters:directionFiltersOut})});
  const d=await r.json();
  const searchMs=Date.now()-searchStart;
  if(!d.ok)return;
  searchResults=d.results;
  logEvent('search',{source:d.source, topScore:d.results[0]?.score, durationMs:searchMs, queryRooms:qrooms});

  const src=d.source==='db'?'DB':'Generated Plans';
  document.getElementById('result-title').innerHTML=`Similarity Search Results <span class="badge">${src}</span>`;
  document.getElementById('results-list').innerHTML=d.results.map((res,i)=>{
    const bd=res.breakdown;
    const roomList=[...new Set(Object.keys(res.rooms).map(k=>RL[k]||k))].join(', ');
    const visHtml=res.png
      ?`<img src="data:image/png;base64,${res.png}" style="max-width:280px;width:100%;border-radius:10px;display:block;margin-top:10px">`
      :'';
    return`<div class="rc ${i===0?'top':''}" id="rc-${i}">
      <div class="rc-row">
        <div class="rn ${i===0?'top':''}">${i+1}</div>
        <div>
          <div style="display:flex;align-items:center;gap:7px;margin-bottom:3px">
            <span style="font-size:13px;font-weight:500">plan_${res.plan_idx}</span>
            <span style="font-size:10px;color:var(--text3)">${Object.keys(res.rooms).length} rooms</span>
          </div>
          <div style="font-size:11px;color:var(--text3);margin-bottom:4px">${roomList}</div>
          <div class="sbar"><div class="sfill" style="width:${(res.score*100).toFixed(1)}%"></div></div>
        </div>
        <div class="rmeta">${(res.score*100).toFixed(1)}%</div>
      </div>
      <div class="chips">
        ${FPS.map(fp=>{
          const pd = res.pair_debug && res.pair_debug[fp.id];
          const tip = pd ? `Query: ${pd.query.join(', ')||'(none)'}\nCandidate: ${pd.candidate.join(', ')||'(none)'}\nOverlap: ${pd.overlap.join(', ')||'(none)'}` : '';
          return `<div class="chip ${bd[fp.id]>.7?'hi':''}" ${tip?`title="${tip}"`:''}><span style="color:${fp.color};margin-right:2px">■</span>${fp.name}: ${(bd[fp.id]*100).toFixed(0)}%</div>`;
        }).join('')}
      </div>
      ${visHtml}
      <!-- Rating -->
      <div class="score-row">
        <span>Score</span>
        <div class="score-btns" id="scbtns-${i}">${[1,2,3,4,5,6,7,8,9,10].map(s=>`<div class="sc-btn" data-ri="${i}" data-sv="${s}" onclick="ratePlan(${i},${s})">${s}</div>`).join('')}</div>
        <span class="score-val" id="starval-${i}">-</span>
      </div>
      <!-- Revision notes -->
      <div class="memo-row">
        <label for="memo-${i}">Revision Notes</label>
        <textarea id="memo-${i}" class="memo-input" rows="2"
          placeholder="Briefly note what you'd like to see revised"
          oninput="updateMemo(${i}, this.value)"></textarea>
      </div>
    </div>`;
  }).join('');
  // Reset for each new search
  searchResults.forEach((_,i)=>{_._rating=null;_._memo='';});
  sessionCommitted = false;
  checkAllRated();
}

// ── Revision notes ──
function updateMemo(ri, value){
  searchResults[ri]._memo = value;
  markSessionDirty();   // mark the session dirty so it's saved even if only a note is left
}

// ── Rating ──
function ratePlan(ri, sv){
  searchResults[ri]._rating = sv;
  document.getElementById('starval-'+ri).textContent = sv + '/10';
  document.getElementById('rc-'+ri).classList.add('rated');
  // Highlight the selected button
  document.querySelectorAll(`#scbtns-${ri} .sc-btn`).forEach(b=>{
    b.classList.toggle('on', parseInt(b.dataset.sv) === sv);
  });
  markSessionDirty();
  logEvent('rate_result',{resultIdx:ri, planIdx:searchResults[ri].plan_idx, score:searchResults[ri].score, rating:sv});
  checkAllRated();
}
function checkAllRated(){
  const anyRated=searchResults.some(r=>r._rating!==null&&r._rating!==undefined);
  document.getElementById('finish-bar').style.display=anyRated?'flex':'none';
}

// ── Accumulated search session management ──
// State variables to ensure each search result is saved only once
const allSearchSessions = [];
let currentQueryRooms = null;   // query rooms used for the current search
let currentQueryMeta  = null;   // {planIdx}
let sessionCommitted  = false;  // whether the current session has already been saved

// Save the current searchResults entries that have a rating or note into allSearchSessions exactly once
function commitCurrentSession(){
  if(sessionCommitted) return;            // already saved
  if(!currentQueryRooms) return;          // no search has been performed yet
  const rated = searchResults.filter(r=>r._rating!=null || (r._memo && r._memo.trim()!==''));
  if(rated.length === 0) return;          // don't save if there's neither a rating nor a note
  allSearchSessions.push({
    timestamp: new Date().toISOString(),
    query: {
      queryPlanIdx: currentQueryMeta?.planIdx,
      rooms: currentQueryRooms,
      weights: currentQueryMeta?.weights,
    },
    ratings: rated.map(r=>({planIdx:r.plan_idx, score:r.score, rating:r._rating, memo:(r._memo||'').trim()})),
  });
  sessionCommitted = true;
}

// Reset sessionCommitted whenever a rating is left (so new ratings can still be added)
function markSessionDirty(){ sessionCommitted = false; }

// ── Save log (Finish) ──
async function saveLog(){
  commitCurrentSession(); // save the final session
  const saveDir=document.getElementById('save-dir-input').value.trim()||'/Users/hongyoonjae/Downloads';
  const logPayload = {
    prolificId: sessionLog.prolificId,
    sessionStart: sessionLog.sessionStart,
    sessionEnd: new Date().toISOString(),
    allSearchSessions: allSearchSessions,
    save_dir: saveDir,
  };
  const r=await fetch('/api/save_log',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(logPayload)});
  const d=await r.json();
  if(d.ok){
    alert(`✓ Log saved\n${d.path}`);
  } else {
    alert('Save failed: '+d.error);
  }
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML

if __name__ == "__main__":
    db_path = os.environ.get("FLOORPLAN_DB", "")
    if db_path and os.path.exists(db_path):
        build_db_index(db_path)

    PORT = 8888
    print(f"""
╔═══════════════════════════════════════════════════╗
║   Floor Plan Recommendation System — local server  ║
╠═══════════════════════════════════════════════════╣
║  Browser: http://localhost:{PORT}                  ║
║  Quit:    Ctrl+C                                   ║
╚═══════════════════════════════════════════════════╝
""")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)

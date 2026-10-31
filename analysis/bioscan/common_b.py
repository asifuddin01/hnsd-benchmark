"""Shared utilities for the BIOSCAN-5M analyses.
Splits come from data/bioscan, features from FEATURES_BIOSCAN, results go to outputs/bioscan (see hnsd_paths.py)."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root, for hnsd_paths
from hnsd_paths import FEATURES_BIOSCAN, SPLITS_BIOSCAN, out
D    = FEATURES_BIOSCAN                  # feature files (.npz)
OUT  = out("bioscan")                   # results are written here
RANKS  = ["kingdom", "phylum", "class", "order", "family", "genus"]
SEED = 42
MISSING = "__missing__"

def l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)

def load_split():
    S = json.load(open(SPLITS_BIOSCAN / "bioscan_split.json"))
    return S, {s["category_id"]: s for s in S}

def load_rep(name):
    """name in {'concat', 'fusion_bioscan', 'fusion_inat'} -> (X float32 L2-normalised, cat_ids)."""
    if name == "concat":
        parts = [np.load(D / f"features_{n}_v2.npz") for n in ["siglip2", "dinov3", "dinov2"]]
        cat = parts[0]["cat_ids"].astype(np.int64)
        for p in parts[1:]:
            assert np.array_equal(p["cat_ids"], cat)
        X = np.concatenate([p["X"].astype(np.float32) for p in parts], axis=1)
        del parts
        return l2(X).astype(np.float32), cat
    f = {"fusion_bioscan": "features_trifusion_bioscan.npz",
         "fusion_inat": "features_trifusion_inat_transfer.npz"}[name]
    z = np.load(D / f)
    return l2(z["X"].astype(np.float32)).astype(np.float32), z["cat_ids"].astype(np.int64)

REPS = ["concat", "fusion_bioscan", "fusion_inat"]

def fusion_folds():
    return json.load(open(SPLITS_BIOSCAN / "fusion_bioscan_species_folds.json"))

def prefix(s, r):
    return tuple(s[x] for x in RANKS[:RANKS.index(r) + 1])

def codes(records, levels):
    out = {}
    for r in levels:
        keys = [prefix(s, r) for s in records]
        lut = {k: i for i, k in enumerate(sorted(set(keys)))}
        out[r] = np.array([lut[k] for k in keys])
    return out

def censored(s):
    """True if the computed ground truth stops at a rank only because the next finer label is missing."""
    tr = s.get("true_rank")
    if tr not in RANKS or tr == "genus":
        return False
    nxt = RANKS[RANKS.index(tr) + 1]
    return str(s[nxt]).startswith(MISSING)

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def save(name, obj):
    json.dump(obj, open(OUT / name, "w"), indent=1, default=float)

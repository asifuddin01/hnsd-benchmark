"""Shared utilities for the iNaturalist v2 analyses.
Splits come from data/inat_v2, features from FEATURES_INAT, results go to outputs/inat_v2 (see hnsd_paths.py)."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root, for hnsd_paths
from hnsd_paths import FEATURES_INAT, SPLITS_INAT, out
D   = FEATURES_INAT                     # feature files (.npz)
OUT = out("inat_v2")                    # results are written here
RANKS  = ["kingdom", "phylum", "class", "order", "family", "genus"]
LEVELS = ["genus", "family", "order", "class", "phylum", "kingdom"]   # finest first
SEED = 42

def load_split():
    S = json.load(open(SPLITS_INAT / "species_split_v2.json"))
    return S, {s["category_id"]: s for s in S}

def l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)

def load_feats(path):
    z = np.load(path)
    return z["X"].astype(np.float32), z["cat_ids"].astype(np.int64)

def concat_feats():
    parts = [np.load(D / f"features_{n}_v2.npz") for n in ["siglip2", "dinov3", "dinov2"]]
    X = l2(np.concatenate([p["X"].astype(np.float32) for p in parts], axis=1))
    return X.astype(np.float32), parts[0]["cat_ids"].astype(np.int64)

def centroids(X, cat):
    ids, inv = np.unique(cat, return_inverse=True)
    C = np.zeros((len(ids), X.shape[1]), np.float64)
    np.add.at(C, inv, X)
    C /= np.bincount(inv)[:, None]
    return ids, l2(C).astype(np.float32)

def prefix(s, r):
    return tuple(s[x] for x in RANKS[:RANKS.index(r) + 1])

def codes(records, levels=LEVELS):
    """Integer code per species per level, computed on lineage prefixes (homonym safe)."""
    out = {}
    for r in levels:
        keys = [prefix(s, r) for s in records]
        lut = {k: i for i, k in enumerate(sorted(set(keys)))}
        out[r] = np.array([lut[k] for k in keys])
    return out

def lca_levels(cd, i, j, levels=LEVELS):
    """Index into `levels` of the finest shared rank for each pair; len(levels) if nothing shared."""
    lvl = np.full(len(i), len(levels))
    for k in range(len(levels) - 1, -1, -1):          # coarse -> fine, finer overwrites
        lvl[cd[levels[k]][i] == cd[levels[k]][j]] = k
    return lvl

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def save(name, obj):
    json.dump(obj, open(OUT / name, "w"), indent=1, default=float)

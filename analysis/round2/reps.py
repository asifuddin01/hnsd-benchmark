"""Shared loaders for round-2 analyses. Every representation is returned on the SAME 90,850 images, ordered as
the SimGCD split (labelled first, then unlabelled), L2-normalised."""
import json
from pathlib import Path
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]           # repository root
sys.path.insert(0, str(ROOT))
from hnsd_paths import (FEATURES_INAT, SPLITS_INAT, SPLITS_GCD, RUN_SIMGCD_DINO, RUN_SIMGCD_DINOV2,
                        RUN_HYPCD_DINOV2, out)
V2 = FEATURES_INAT
OUT = out("round2")
SEED = 42
FILES = {"simgcd_dino": RUN_SIMGCD_DINO / "features_simgcd_v2.npz",
         "simgcd_dinov2": RUN_SIMGCD_DINOV2 / "features_simgcd_dinov2_v2.npz",
         "hypcd_dinov2": RUN_HYPCD_DINOV2 / "features_hypcd_dinov2_v2.npz",
         "fusion": V2 / "features_trifusion_v2.npz"}
HEADS = {"simgcd_dino": RUN_SIMGCD_DINO / "simgcd_results.json",
         "simgcd_dinov2": RUN_SIMGCD_DINOV2 / "results.json",
         "hypcd_dinov2": RUN_HYPCD_DINOV2 / "results.json"}
SPL = json.load(open(SPLITS_GCD / "simgcd_split.json"))
for f in [RUN_SIMGCD_DINOV2 / "simgcd_split.json", RUN_HYPCD_DINOV2 / "simgcd_split.json"]:
    if f.exists():                                    # each GCD run keeps a copy of the split it used
        assert json.load(open(f))["labelled"] == SPL["labelled"], f"split differs: {f}"
SPLIT_RECS = json.load(open(SPLITS_INAT / "species_split_v2.json"))
BY = {s["category_id"]: s for s in SPLIT_RECS}
ROLE = {c: s["role"] for c, s in BY.items()}
OLD, NEW = SPL["old_species"], SPL["new_species"]
CLASS_OF = {c: i for i, c in enumerate(OLD + NEW)}; N_OLD, K_TRUE = len(OLD), len(OLD) + len(NEW)
LAB, UNL = SPL["labelled"], SPL["unlabelled"]; NL = len(LAB)
REL = [r for r, _ in LAB] + [r for r, _ in UNL]
CAT = np.array([c for _, c in LAB] + [c for _, c in UNL])

def l2(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

def load(rep):
    if rep == "concat":
        ps = [np.load(V2 / f"features_{n}_v2.npz") for n in ["siglip2", "dinov3", "dinov2"]]
        rel = ps[0]["relpaths"]
        for p in ps[1:]:
            assert np.array_equal(p["relpaths"], rel)
        X = np.concatenate([p["X"].astype(np.float32) for p in ps], 1); del ps
    else:
        z = np.load(FILES[rep]); X, rel = z["X"], z["relpaths"]
    pos = {r: i for i, r in enumerate(rel.tolist())}
    return l2(X[[pos[r] for r in REL]].astype(np.float32)).astype(np.float32)

def save(name, obj):
    json.dump(obj, open(OUT / name, "w"), indent=1, default=float)

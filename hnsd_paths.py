"""Locations of the large files that are not part of the repository (cached features and GCD runs).

Resolution order for every setting: environment variable, then `paths.local.json` in the repository root
(copy `paths.example.json`), then the default below. Relative paths are taken relative to the repository root.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
SPLITS_INAT, SPLITS_BIOSCAN, SPLITS_GCD = DATA / "inat_v2", DATA / "bioscan", DATA / "gcd"

_local = REPO / "paths.local.json"
_cfg = json.loads(_local.read_text()) if _local.exists() else {}


def _path(key, default):
    p = Path(os.environ.get(key) or _cfg.get(key) or default).expanduser()
    return p if p.is_absolute() else REPO / p


# feature folders written by the stage notebooks (files/v2 and files/v2_bioscan on the GPU machine)
FEATURES_INAT = _path("HNSD_FEATURES_INAT", "features/inat_v2")
FEATURES_BIOSCAN = _path("HNSD_FEATURES_BIOSCAN", "features/bioscan")
# GCD runs (notebooks 04, 05a, 05b write into these sub-folders of FEATURES_INAT)
RUN_SIMGCD_DINO = _path("HNSD_RUN_SIMGCD_DINO", FEATURES_INAT / "simgcd_full")
RUN_SIMGCD_DINOV2 = _path("HNSD_RUN_SIMGCD_DINOV2", FEATURES_INAT / "simgcd_dinov2")
RUN_HYPCD_DINOV2 = _path("HNSD_RUN_HYPCD_DINOV2", FEATURES_INAT / "hypcd_dinov2")
# where re-run analyses write their JSON (reference copies live in results/)
OUTPUTS = _path("HNSD_OUTPUTS", "outputs")


def out(part):
    d = OUTPUTS / part
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    for k in ["FEATURES_INAT", "FEATURES_BIOSCAN", "RUN_SIMGCD_DINO", "RUN_SIMGCD_DINOV2", "RUN_HYPCD_DINOV2", "OUTPUTS"]:
        p = globals()[k]
        print(f"{k:<18} {'ok     ' if p.exists() else 'missing'} {p}")

"""Run an unchanged analysis/inat_v2 script (a1 / a2 / a4) on another feature file.
The script's source is read, its feature path is swapped, and its save() writes to outputs/round2/
with a tag prefix, so nothing in outputs/inat_v2/ is overwritten.
usage: python analysis/round2/r1_run_v2_on.py <a1_router_clusters|a2_construction|a4_placement> <features.npz> <tag> [args]"""
import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "inat_v2"))
import common
OUT = common.out("round2")
script, feats, tag = sys.argv[1], sys.argv[2], sys.argv[3]
def save(name, obj):
    json.dump(obj, open(OUT / f"{tag}__{name}", "w"), indent=1, default=float)
    print("saved", OUT / f"{tag}__{name}")
common.save = save
src = (REPO / "analysis" / "inat_v2" / f"{script}.py").read_text()
old = 'D / "features_trifusion_v2.npz"'
assert src.count(old) >= 1, "feature path not found"
src = src.replace(old, repr(str(feats)))
sys.argv = [script] + sys.argv[4:]
exec(compile(src, script, "exec"), {"__name__": "__main__"})

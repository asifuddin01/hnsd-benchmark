"""R3. Placement with the novel rank prior ESTIMATED from the unlabelled novel species (BBSE, EM), instead of the
pseudo-training prior or the leaky evaluation prior. Same signals, classifier and baselines as A4 (v2) and B2
(BIOSCAN); the natural-prior rows reproduce those analyses.
usage: python analysis/round2/r3_prior.py v2 fusion concat
       python analysis/round2/r3_prior.py bioscan concat fusion_bioscan fusion_inat"""
import sys, json, time
import numpy as np
from pathlib import Path
from placement_core import Placement, PL, DEPTH

ROOT = Path(__file__).resolve().parents[2]           # repository root
sys.path.insert(0, str(ROOT))
from hnsd_paths import FEATURES_INAT, SPLITS_INAT, RUN_SIMGCD_DINOV2, out
OUT = out("round2")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
prefix = lambda s, r: tuple(s[x] for x in RANKS[:RANKS.index(r) + 1])
l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
ds, reps = sys.argv[1], sys.argv[2:]

if ds == "v2":
    V2 = FEATURES_INAT
    S = json.load(open(SPLITS_INAT / "species_split_v2.json"))
    def feats(rep):
        if rep == "concat":
            ps = [np.load(V2 / f"features_{n}_v2.npz") for n in ["siglip2", "dinov3", "dinov2"]]
            return l2(np.concatenate([p["X"].astype(np.float32) for p in ps], 1)), ps[0]["cat_ids"]
        f = {"fusion": V2 / "features_trifusion_v2.npz",
             "simgcd_dinov2": RUN_SIMGCD_DINOV2 / "features_simgcd_dinov2_v2.npz"}[rep]
        z = np.load(f); return z["X"].astype(np.float32), z["cat_ids"]
else:
    sys.path.insert(0, str(ROOT / "analysis" / "bioscan"))
    from common_b import load_split, load_rep
    S, _ = load_split()
    feats = load_rep
BY = {s["category_id"]: s for s in S}
known = sorted(c for c, s in BY.items() if s["role"] == "known")
test = sorted(c for c, s in BY.items() if s["role"] == "novel_known_branch")
recs = [BY[c] for c in known + test]; nk = len(known)
cd = {}
for r in PL:
    keys = [prefix(s, r) for s in recs]; lut = {k: i for i, k in enumerate(sorted(set(keys)))}; cd[r] = np.array([lut[k] for k in keys])
K = {r: cd[r][:nk] for r in PL}; T = {r: cd[r][nk:] for r in PL}
rep_of = {r: {} for r in PL}
for i in range(len(recs)):
    for r in PL:
        rep_of[r].setdefault(cd[r][i], {x: cd[x][i] for x in PL[:DEPTH[r] + 1]})
true_rank = np.array([DEPTH[BY[c]["true_rank"]] for c in test])
subsets = {"all": None}
if ds == "bioscan":
    src = np.array([BY[c]["holdout"] for c in test])
    subsets.update({"unseen": src == "bioscan_unseen", "other_heldout": src == "bioscan_other_heldout"})

out = {}
for rep in reps:
    t0 = time.time(); X, cat = feats(rep)
    ids, inv = np.unique(cat, return_inverse=True)
    C = np.zeros((len(ids), X.shape[1])); np.add.at(C, inv, X); C = l2(C); del X
    pos = {c: i for i, c in enumerate(ids.tolist())}
    P = Placement(C[[pos[c] for c in known]], C[[pos[c] for c in test]], K, T, rep_of, true_rank)
    res, pri, diag, tp = P.evaluate_all(subsets)
    out[rep] = {"results": res, "priors": pri, "diagnostics": diag, "true_prior": tp.tolist()}
    print(f"\n[{ds}/{rep}] ({time.time()-t0:.0f}s) true prior (order,family,genus) {np.round(tp,3).tolist()} | OOF rank acc {diag['oof_acc']:.3f}")
    for k, v in pri.items():
        print(f"   prior {k:<40} {v['prior(order,family,genus)']}  L1 {v['L1_to_true']:.3f}")
    print(f"   {'method':<52}" + "".join(f"{s+' Par [CI] / hF1':>34}" for s in subsets))
    for k, v in res.items():
        print(f"   {k:<52}" + "".join(f"{v[s]['ParentAcc']:>10.3f} [{v[s]['ParentAcc_CI'][0]:.2f},{v[s]['ParentAcc_CI'][1]:.2f}] {v[s]['hF1']:>7.3f}   " for s in subsets), flush=True)
    json.dump(out, open(OUT / f"r3_prior_{ds}.json", "w"), indent=1, default=float)

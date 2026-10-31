"""R2b. End-to-end construction on DISCOVERED clusters (no oracle species units).
Pipeline on the 61,425 unlabelled images (29,425 known + 32,000 novel):
  1. semi-supervised k-means with k = estimated k (R2a) or the oracle k = 1,817;
     images in a cluster seeded by a labelled species are routed to that known species, the rest form new units;
  2. rank radii from the known species (centroids of their LABELLED images), 25th percentile per rank, transferred
     by quantile (or used as absolute values) to the new units, average linkage, cut per rank;
  3. every novel_unseen_branch image gets a predicted group per rank: its new unit's group, or, if it was routed to a
     known species, that species' taxon. Scored per image against the true lineage (ARI, pair-F1, NMI with the
     all-singletons NMI beside it), per hold-out mechanism.
Reference: the same construction with ORACLE units (true novel species, perfect routing), scored the same way.
usage: python analysis/round2/r2_end_to_end.py fusion concat"""
import sys, json, time
from reps import *
sys.path.insert(0, str(ROOT / "analysis" / "inat_v2"))
from common import codes, lca_levels, LEVELS, RANKS, prefix
from gcdlib import ss_kmeans, pair_f1
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI

CUT = ["genus", "family", "order"]
UNLc = np.array([c for _, c in UNL]); y_lab = np.array([CLASS_OF[c] for _, c in LAB])
SUBSETS = {"order hold-outs": lambda s: s["holdout"] == "order",
           "Fungi": lambda s: s["holdout"] == "supercategory" and s["supercategory"] == "Fungi",
           "Reptiles": lambda s: s["holdout"] == "supercategory" and s["supercategory"] == "Reptiles",
           "all unseen-branch": lambda s: s["role"] == "novel_unseen_branch"}
EVAL = {t: np.array([f(BY[c]) for c in UNLc]) for t, f in SUBSETS.items()}

def radii(Ck, recs):
    cd = codes(recs); i, j = np.triu_indices(len(Ck), 1); s = (Ck @ Ck.T)[i, j]; lv = lca_levels(cd, i, j)
    tau = {r: float(np.percentile(s[lv == LEVELS.index(r)], 25)) for r in CUT}
    srt = np.sort(s); q = {r: float(np.searchsorted(srt, t) / len(srt)) for r, t in tau.items()}
    return tau, q

def groups(Cu, taus):
    D_ = np.clip(1 - Cu @ Cu.T, 0, None); np.fill_diagonal(D_, 0); D_ = (D_ + D_.T) / 2
    Z = linkage(squareform(D_, checks=False), "average")
    return {r: fcluster(Z, t=1 - taus[r], criterion="distance") for r in CUT}

def score(unit_of_img, known_of_img, unit_groups):
    """unit_of_img: unit index or -1; known_of_img: known species id or -1 (exactly one is set)."""
    res = {}
    for r in CUT:
        true_key = [prefix(BY[c], r) for c in UNLc]
        pred_key = [("U", int(unit_groups[r][u])) if u >= 0 else ("K",) + prefix(BY[k], r)
                    for u, k in zip(unit_of_img, known_of_img)]
        for t, m in EVAL.items():
            idx = np.where(m)[0]
            tl = np.unique([true_key[i] for i in idx], axis=0, return_inverse=True)[1].ravel()
            pk = {}; pl = np.array([pk.setdefault(pred_key[i], len(pk)) for i in idx])
            res.setdefault(t, {})[r] = {"ARI": ARI(tl, pl), "pairF1": pair_f1(tl, pl), "NMI": NMI(tl, pl),
                                       "NMI_singletons": NMI(tl, np.arange(len(tl))), "n_pred": int(len(pk)),
                                       "n_true": int(tl.max() + 1)}
    return res

def run_setting(X, tau, q, mode, k=None):
    U = X[NL:]
    if mode == "oracle units":
        nov = sorted(set(UNLc[[ROLE[c] != "known" for c in UNLc]].tolist()))
        uid = {c: i for i, c in enumerate(nov)}
        unit_of = np.array([uid.get(c, -1) for c in UNLc]); known_of = np.where(unit_of < 0, UNLc, -1)
        routing = {}
    else:
        a = ss_kmeans(X, NL, y_lab, k)
        new_ids = np.unique(a[a >= N_OLD]); uid = {c: i for i, c in enumerate(new_ids.tolist())}
        unit_of = np.array([uid.get(int(x), -1) for x in a])
        known_of = np.where(a < N_OLD, np.array(OLD)[np.minimum(a, N_OLD - 1)], -1)
        roles = np.array([ROLE[c] for c in UNLc])
        routing = {f"{r}_to_new_units": float(np.mean(unit_of[roles == r] >= 0))
                   for r in ["known", "novel_known_branch", "novel_unseen_branch"]}
    n_units = int(unit_of.max() + 1)
    Cu = np.zeros((n_units, X.shape[1])); np.add.at(Cu, unit_of[unit_of >= 0], U[unit_of >= 0]); Cu = l2(Cu)
    iu, ju = np.triu_indices(n_units, 1); su = (Cu @ Cu.T)[iu, ju]
    tau_q = {r: float(np.quantile(su, qq)) for r, qq in q.items()}
    out = {"k": k, "n_units": n_units, "routing": routing}
    for name, taus in [("quantile", tau_q), ("absolute", tau)]:
        g = groups(Cu, taus)
        out[name] = score(unit_of, known_of, g)
        out[name]["n_groups"] = {r: int(len(set(g[r]))) for r in CUT}
    return out

kest = json.load(open(OUT / "r2_estimate_k.json"))
res = {}
for rep in (sys.argv[1:] or ["fusion"]):
    X = load(rep); t0 = time.time()
    Ck = l2(np.stack([X[:NL][CAT[:NL] == c].mean(0) for c in OLD])); tau, q = radii(Ck, [BY[c] for c in OLD])
    print(f"[{rep}] known radii {np.round(list(tau.values()),3).tolist()} quantiles {np.round(list(q.values()),3).tolist()}", flush=True)
    settings = [("oracle units", None), ("end-to-end, oracle k", K_TRUE)]
    if rep in kest:
        settings.append(("end-to-end, estimated k", kest[rep]["k_hat"]))
    res[rep] = {}
    for name, k in settings:
        r = res[rep][name] = run_setting(X, tau, q, name if name == "oracle units" else "e2e", k)
        o = r["quantile"]["order hold-outs"]["order"]; a = r["quantile"]["all unseen-branch"]
        print(f"[{rep}] {name:<26} k={k} units {r['n_units']:>5} routing {r['routing']} | order hold-outs ORDER ARI "
              f"{o['ARI']:.3f} pF1 {o['pairF1']:.3f} NMI {o['NMI']:.3f} (sing {o['NMI_singletons']:.3f}) | all unseen GENUS ARI "
              f"{a['genus']['ARI']:.3f} pF1 {a['genus']['pairF1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        save("r2_end_to_end.json", res)
    del X

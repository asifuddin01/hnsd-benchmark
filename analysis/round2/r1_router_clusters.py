"""R1b. Router and novel-species clustering (A1 protocol) for SimGCD features vs the trained fusion vs concat,
on exactly the same 90,850 images. Known reference/test images follow the SimGCD split: prototypes from the
labelled half of each known species, router test on the unlabelled half. Clustering: k-means with oracle k on
each novel role, with NMI next to the all-singletons partition, plus ARI and pair-F1.
Caveats: SimGCD saw every image (unlabelled) during training, as the GCD setting allows; the fusion was trained
with labels of all images of 1,057 known species."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.metrics.cluster import contingency_matrix

from reps import OUT, SEED, ROLE, CAT, NL, load, save, l2
import sys

def auroc(neg, pos):
    r = rankdata(np.r_[neg, pos]); n0, n1 = len(neg), len(pos)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

def pair_f1(y, p):
    M = contingency_matrix(y, p, sparse=True); c2 = lambda x: float((x * (x - 1) / 2).sum())
    tp = c2(M.data.astype(float)); pp = c2(np.asarray(M.sum(0)).ravel().astype(float)); tt = c2(np.asarray(M.sum(1)).ravel().astype(float))
    pr, rc = tp / pp, tp / tt
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

roles = np.array([ROLE[c] for c in CAT])
idx_unl = np.arange(NL, len(CAT))
test = idx_unl[roles[idx_unl] == "known"]
nov = idx_unl[np.isin(roles[idx_unl], ["novel_known_branch", "novel_unseen_branch"])]
out = {}
for rep in (sys.argv[1:] or ["simgcd_dinov2", "hypcd_dinov2"]):
    X = load(rep)
    sp, inv = np.unique(CAT[:NL], return_inverse=True)
    P = np.zeros((len(sp), X.shape[1])); np.add.at(P, inv, X[:NL]); P = l2(P).astype(np.float32)
    sc = lambda idx: 1 - np.concatenate([(X[idx[i:i + 4096]] @ P.T).max(1) for i in range(0, len(idx), 4096)])
    st, sn = sc(test), sc(nov)
    clus = lambda s, idx: np.array([s[CAT[idx] == c].mean() for c in np.unique(CAT[idx])])
    r = {"router_auroc_image": float(auroc(st, sn)), "router_auroc_cluster": float(auroc(clus(st, test), clus(sn, nov)))}
    for role in ["novel_known_branch", "novel_unseen_branch"]:
        idx = idx_unl[roles[idx_unl] == role]; y = CAT[idx]; k = len(np.unique(y))
        p = KMeans(k, random_state=SEED, n_init=3).fit_predict(X[idx])
        r[role] = {"NMI": NMI(y, p), "NMI_singletons": NMI(y, np.arange(len(y))), "ARI": ARI(y, p), "pairF1": pair_f1(y, p)}
    out[rep] = r
    a, b = r["novel_known_branch"], r["novel_unseen_branch"]
    print(f"[{rep:<7}] router img {r['router_auroc_image']:.3f} cluster {r['router_auroc_cluster']:.3f} | "
          f"place NMI {a['NMI']:.3f} (sing {a['NMI_singletons']:.3f}) ARI {a['ARI']:.3f} pF1 {a['pairF1']:.3f} | "
          f"construct NMI {b['NMI']:.3f} (sing {b['NMI_singletons']:.3f}) ARI {b['ARI']:.3f} pF1 {b['pairF1']:.3f}", flush=True)
    del X
save("r1_router_clusters.json", out)

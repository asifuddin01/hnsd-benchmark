"""R1a. GCD protocol, same images and labels as the SimGCD run (data/gcd/simgcd_split.json):
1,177 old species (50% of their images labelled) + 640 new species, k = 1,817.
Compared: SimGCD's parametric head (from step 4) vs semi-supervised k-means (GCD, Vaze et al. 2022) on
SimGCD CLS features, the trained fusion and the frozen concatenation.
Reported: ACC All/Old/New (SimGCD split_cluster_acc_v2) on the 61,425 unlabelled images, plus NMI/ARI/pair-F1
with the all-singletons baseline on each novel role.
Caveat: the fusion was trained with the labels of ALL images of 1,057 known species (more supervision than the
GCD protocol's 50%); it is reported for reference, not as a like-for-like GCD method.
usage: python analysis/round2/r1_gcd_protocol.py [simgcd_dinov2 hypcd_dinov2]"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.metrics.cluster import contingency_matrix

from reps import ROOT, OUT, SEED, SPL, ROLE, OLD, NEW, CLASS_OF, N_OLD, LAB, UNL, load, HEADS
K = len(OLD) + len(NEW)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def acc_v2(y, p, mask):                     # SimGCD split_cluster_acc_v2
    D = max(p.max(), y.max()) + 1; w = np.zeros((D, D), np.int64); np.add.at(w, (p, y), 1)
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T; m = {j: i for i, j in ind}
    og, ng = set(y[mask]), set(y[~mask])
    return (float(sum(w[i, j] for i, j in ind) / p.size),
            float(sum(w[m[i], i] for i in og) / sum(w[:, i].sum() for i in og)),
            float(sum(w[m[i], i] for i in ng) / sum(w[:, i].sum() for i in ng)))

def pair_f1(y, p):
    M = contingency_matrix(y, p, sparse=True); c2 = lambda x: float((x * (x - 1) / 2).sum())
    tp = c2(M.data.astype(float)); pp = c2(np.asarray(M.sum(0)).ravel().astype(float)); tt = c2(np.asarray(M.sum(1)).ravel().astype(float))
    pr, rc = tp / pp, tp / tt
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

def metrics(pred_unl):
    y = np.array([CLASS_OF[c] for _, c in UNL]); roles = np.array([ROLE[c] for _, c in UNL])
    out = dict(zip(["ACC_all", "ACC_old", "ACC_new"], acc_v2(y, pred_unl, y < N_OLD)))
    for r in ["novel_known_branch", "novel_unseen_branch"]:
        m = roles == r
        out[r] = {"NMI": NMI(y[m], pred_unl[m]), "ARI": ARI(y[m], pred_unl[m]), "pairF1": pair_f1(y[m], pred_unl[m]),
                  "NMI_singletons": NMI(y[m], np.arange(m.sum())), "n_pred_clusters": int(len(np.unique(pred_unl[m])))}
    return out

def nearest(X, C):
    cc = (C * C).sum(1); a = torch.empty(X.shape[0], dtype=torch.long, device=DEV); d = torch.empty(X.shape[0], device=DEV)
    for i in range(0, X.shape[0], 8192):
        s = cc[None] - 2 * X[i:i + 8192] @ C.T
        d[i:i + 8192], a[i:i + 8192] = s.min(1)
    return a, d

def ss_kmeans(X, n_lab, y_lab, k, n_init=3, max_iter=200):
    """Semi-supervised k-means as in GCD: labelled points stay in their class; old-class centroids start at
    labelled means; the remaining k - n_old start by k-means++ on the unlabelled points."""
    Xt = torch.tensor(X, device=DEV); yl = torch.tensor(y_lab, device=DEV); n_old = int(y_lab.max()) + 1
    U = Xt[n_lab:]; xxu = (U * U).sum(1)
    C0 = torch.zeros(n_old, X.shape[1], device=DEV).index_add_(0, yl, Xt[:n_lab]) / torch.bincount(yl, minlength=n_old)[:, None].float()
    best = None
    for r in range(n_init):
        g = torch.Generator().manual_seed(SEED + r)
        a0, d0 = nearest(U, C0); dmin = (d0 + xxu).clamp_min(0); extra = []
        for _ in range(k - n_old):
            j = int(torch.multinomial((dmin / dmin.sum()).cpu().double(), 1, generator=g)); extra.append(j)
            dmin = torch.minimum(dmin, ((U - U[j]) ** 2).sum(1))
        C = torch.cat([C0, U[extra]]); prev = None
        for it in range(max_iter):
            a, _ = nearest(U, C)
            lab_all = torch.cat([yl, a])
            C_new = torch.zeros_like(C).index_add_(0, lab_all, Xt); cnt = torch.bincount(lab_all, minlength=k).float()
            keep = cnt == 0; C_new = C_new / cnt.clamp_min(1)[:, None]; C_new[keep] = C[keep]; C = C_new
            if prev is not None and torch.equal(a, prev):
                break
            prev = a
        a, d = nearest(U, C); inertia = float((d + xxu).clamp_min(0).sum())
        print(f"   restart {r}: iters {it+1}, unlabelled inertia {inertia:.1f}", flush=True)
        if best is None or inertia < best[1]:
            best = (a.cpu().numpy(), inertia)
    del Xt, U
    return best[0]

res = {}
for h, p in HEADS.items():
    res[f"{h} head"] = json.load(open(p))["final"]
y_lab = np.array([CLASS_OF[c] for _, c in LAB])
for rep in (sys.argv[1:] or ["simgcd_dinov2", "hypcd_dinov2"]):
    t0 = time.time(); X = load(rep)
    print(f"[{rep}] {X.shape}", flush=True)
    pred = ss_kmeans(X, len(LAB), y_lab, K)
    res[f"SS-k-means on {rep}"] = r = metrics(pred); r["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "r1_gcd_protocol.json", "w"), indent=1, default=float)
    del X
    if DEV == "mps":
        torch.mps.empty_cache()
print(f"\n{'method':<26}{'All':>7}{'Old':>7}{'New':>7} | novel_known_branch NMI(sing) ARI pF1 | novel_unseen_branch NMI(sing) ARI pF1")
for k_, v in res.items():
    a, b = v["novel_known_branch"], v["novel_unseen_branch"]
    print(f"{k_:<26}{v['ACC_all']:>7.3f}{v['ACC_old']:>7.3f}{v['ACC_new']:>7.3f} | {a['NMI']:.3f}({a['NMI_singletons']:.3f}) {a['ARI']:.3f} {a['pairF1']:.3f}"
          f" | {b['NMI']:.3f}({b['NMI_singletons']:.3f}) {b['ARI']:.3f} {b['pairF1']:.3f}")

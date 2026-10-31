"""B1c. Novel-species clustering with a GPU (MPS) k-means, used because sklearn KMeans with k=2,914 on the
3,200-d concat features would take hours on this Mac.  Same protocol as B1: oracle k, k-means++ init, Lloyd
iterations, best of 5 restarts by inertia.  Run on all three representations so the two fusion rows can be
checked against the sklearn numbers in b1_*.json.
Needs PyTorch (uses CUDA or the Apple GPU when available): python analysis/bioscan/b1c_cluster_torch.py"""
import sys, time
import torch
from common_b import *
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.metrics.cluster import contingency_matrix

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
S, BY = load_split()
ROLE = {c: s["role"] for c, s in BY.items()}

def pair_f1(y, p):
    M = contingency_matrix(y, p, sparse=True)
    c2 = lambda x: (x * (x - 1) / 2).sum()
    tp = c2(M.data.astype(np.float64))
    pp = c2(np.asarray(M.sum(0)).ravel().astype(np.float64)); tt = c2(np.asarray(M.sum(1)).ravel().astype(np.float64))
    pr = tp / pp if pp else 0.0; rc = tp / tt if tt else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc

def sqdist_min(X, C, xx):
    """min squared distance and argmin, chunked."""
    cc = (C * C).sum(1)
    best = torch.empty(X.shape[0], device=DEV); arg = torch.empty(X.shape[0], dtype=torch.long, device=DEV)
    for i in range(0, X.shape[0], 8192):
        d = xx[i:i + 8192, None] - 2 * X[i:i + 8192] @ C.T + cc[None]
        best[i:i + 8192], arg[i:i + 8192] = d.min(1)
    return best.clamp_min(0), arg

def kmeanspp(X, k, g):
    n = X.shape[0]; xx = (X * X).sum(1)
    idx = [int(torch.randint(n, (1,), generator=g))]
    dmin = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        # greedy-free k-means++: sample proportional to D^2 (sampling on CPU for reproducibility)
        p = (dmin / dmin.sum()).cpu().double()
        j = int(torch.multinomial(p, 1, generator=g)); idx.append(j)
        dmin = torch.minimum(dmin, ((X - X[j]) ** 2).sum(1))
    return X[idx].clone(), xx

def lloyd(X, C, xx, max_iter=300, tol=1e-4):
    prev = None
    for it in range(max_iter):
        d, a = sqdist_min(X, C, xx)
        newC = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=C.shape[0]).float()
        empty = cnt == 0
        newC = newC / cnt.clamp_min(1)[:, None]
        if empty.any():                       # re-seed empty clusters at the farthest points
            far = torch.topk(d, int(empty.sum())).indices; newC[empty] = X[far]
        shift = ((newC - C) ** 2).sum()
        C = newC
        if prev is not None and torch.equal(a, prev):
            break
        prev = a
        if shift < tol * 1e-3:
            break
    d, a = sqdist_min(X, C, xx)
    return a.cpu().numpy(), float(d.sum()), it + 1

out = {}
for name in (sys.argv[1:] or REPS):
    X, cat = load_rep(name)
    idx = np.where(np.array([ROLE[c] == "novel_known_branch" for c in cat]))[0]
    y = cat[idx]; k = len(np.unique(y))
    Xt = torch.tensor(X[idx], device=DEV); del X
    t0 = time.time(); best = None
    for r in range(5):
        g = torch.Generator().manual_seed(SEED + r)
        C, xx = kmeanspp(Xt, k, g)
        a, inertia, iters = lloyd(Xt, C, xx)
        print(f"[{name}] restart {r} inertia {inertia:.1f} iters {iters} ({time.time()-t0:.0f}s)", flush=True)
        if best is None or inertia < best[1]:
            best = (a, inertia)
    pred = best[0]; f1, pr, rc = pair_f1(y, pred)
    res = {"impl": f"torch-{DEV} kmeans++ + Lloyd, best of 5", "images": len(y), "k": k, "seconds": time.time() - t0,
           "NMI": NMI(y, pred), "ARI": ARI(y, pred), "pairF1": f1, "pairP": pr, "pairR": rc,
           "trivial_singletons_NMI": NMI(y, np.arange(len(y)))}
    print(f"[{name}] NMI {res['NMI']:.3f} (singletons {res['trivial_singletons_NMI']:.3f}) | ARI {res['ARI']:.3f} "
          f"| pair-F1 {f1:.3f} (P {pr:.3f} R {rc:.3f})", flush=True)
    out[name] = res
    save(f"b1c_cluster_{name}.json", res)
    del Xt
    if DEV == "mps":
        torch.mps.empty_cache()

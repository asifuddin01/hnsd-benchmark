"""k-means tools on the Apple GPU (MPS): plain k-means, semi-supervised k-means (GCD), clustering accuracy."""
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def nearest(X, C):
    cc = (C * C).sum(1); a = torch.empty(X.shape[0], dtype=torch.long, device=DEV); d = torch.empty(X.shape[0], device=DEV)
    for i in range(0, X.shape[0], 8192):
        s = cc[None] - 2 * X[i:i + 8192] @ C.T
        d[i:i + 8192], a[i:i + 8192] = s.min(1)
    return a, d

def _pp(U, k_extra, g, C0=None):
    """k-means++ seeding of k_extra centres on U, continuing from existing centres C0 if given."""
    xx = (U * U).sum(1)
    if C0 is None:
        j = int(torch.randint(U.shape[0], (1,), generator=g)); idx = [j]; dmin = (xx - 2 * (U @ U[j]) + xx[j]).clamp_min(0); k_extra -= 1
    else:
        _, d0 = nearest(U, C0); dmin = (d0 + xx).clamp_min(0); idx = []
    for _ in range(k_extra):
        # D^2 sampling by inverse CDF on the GPU (one scalar transfer per centre; uniform drawn from the CPU generator)
        cs = torch.cumsum(dmin.double() if U.device.type == "cpu" else dmin, 0)
        u = torch.rand(1, generator=g).item() * float(cs[-1])
        j = min(int(torch.searchsorted(cs, torch.tensor([u], device=cs.device, dtype=cs.dtype))), U.shape[0] - 1); idx.append(j)
        dmin = torch.minimum(dmin, (xx - 2 * (U @ U[j]) + xx[j]).clamp_min(0))
    return U[idx]

def kmeans(X, k, seed=42, max_iter=100):
    Xt = torch.as_tensor(X, device=DEV); g = torch.Generator().manual_seed(seed)
    C = _pp(Xt, k, g); prev = None
    for it in range(max_iter):
        a, _ = nearest(Xt, C)
        Cn = torch.zeros_like(C).index_add_(0, a, Xt); cnt = torch.bincount(a, minlength=k).float()
        keep = cnt == 0; Cn = Cn / cnt.clamp_min(1)[:, None]; Cn[keep] = C[keep]; C = Cn
        if prev is not None and torch.equal(a, prev):
            break
        prev = a
    a, _ = nearest(Xt, C)
    return a.cpu().numpy()

def ss_kmeans(X, n_lab, y_lab, k, n_init=3, max_iter=200, seed=42, verbose=True):
    """GCD semi-supervised k-means: labelled points stay in their class; old centres start at labelled means;
    the other k - n_old centres by k-means++ on the unlabelled points. Returns assignments of the unlabelled points."""
    Xt = torch.as_tensor(X, device=DEV); yl = torch.as_tensor(y_lab, device=DEV); n_old = int(y_lab.max()) + 1
    U = Xt[n_lab:]; xxu = (U * U).sum(1)
    C0 = torch.zeros(n_old, X.shape[1], device=DEV).index_add_(0, yl, Xt[:n_lab]) / torch.bincount(yl, minlength=n_old)[:, None].float()
    best = None
    for r in range(n_init):
        g = torch.Generator().manual_seed(seed + r)
        C = torch.cat([C0, _pp(U, k - n_old, g, C0)]); prev = None
        for it in range(max_iter):
            a, _ = nearest(U, C); lab_all = torch.cat([yl, a])
            Cn = torch.zeros_like(C).index_add_(0, lab_all, Xt); cnt = torch.bincount(lab_all, minlength=k).float()
            keep = cnt == 0; Cn = Cn / cnt.clamp_min(1)[:, None]; Cn[keep] = C[keep]; C = Cn
            if prev is not None and torch.equal(a, prev):
                break
            prev = a
        a, d = nearest(U, C); inertia = float((d + xxu).clamp_min(0).sum())
        if verbose:
            print(f"   ss-kmeans k={k} restart {r}: iters {it+1}, inertia {inertia:.1f}", flush=True)
        if best is None or inertia < best[1]:
            best = (a.cpu().numpy(), inertia)
    return best[0]

def cluster_acc(y, p):
    """Hungarian clustering accuracy (rectangular allowed)."""
    yi = np.unique(y, return_inverse=True)[1]; pi = np.unique(p, return_inverse=True)[1]
    w = np.zeros((pi.max() + 1, yi.max() + 1), np.int64); np.add.at(w, (pi, yi), 1)
    r, c = linear_sum_assignment(-w)
    return float(w[r, c].sum() / len(y))

def acc_v2(y, p, mask):
    D = max(p.max(), y.max()) + 1; w = np.zeros((D, D), np.int64); np.add.at(w, (p, y), 1)
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T; m = {j: i for i, j in ind}
    og, ng = set(y[mask]), set(y[~mask])
    return (float(sum(w[i, j] for i, j in ind) / p.size),
            float(sum(w[m[i], i] for i in og) / sum(w[:, i].sum() for i in og)),
            float(sum(w[m[i], i] for i in ng) / sum(w[:, i].sum() for i in ng)))

def pair_f1(y, p):
    M = contingency_matrix(y, p, sparse=True); c2 = lambda x: float((x * (x - 1) / 2).sum())
    tp = c2(M.data.astype(float)); pp = c2(np.asarray(M.sum(0)).ravel().astype(float)); tt = c2(np.asarray(M.sum(1)).ravel().astype(float))
    pr, rc = (tp / pp if pp else 0.0), (tp / tt if tt else 0.0)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

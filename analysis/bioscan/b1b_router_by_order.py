"""B1b. Is the below-chance router AUROC a composition effect?  Known species are mostly Diptera, novel species
mostly Hymenoptera. Router AUROC is recomputed within each order (known-test clusters vs novel clusters of the
same order), and pooled with order-stratified weighting.  Protocol P0 (30% of every known species held out) is
leak-free for concat and the iNat fusion (neither saw BIOSCAN); for the BIOSCAN fusion it is reported for
completeness only, since 98% of those known clusters were in its training data."""
import sys
from common_b import *
from collections import Counter
from scipy.stats import rankdata

S, BY = load_split()
ROLE = {c: s["role"] for c, s in BY.items()}
ORD = {c: s["order"] for c, s in BY.items()}

def auroc(neg, pos):
    r = rankdata(np.r_[neg, pos]); n0, n1 = len(neg), len(pos)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

out = {}
for name in (sys.argv[1:] or REPS):
    X, cat = load_rep(name)
    rng = np.random.default_rng(SEED)
    ref, test = [], []
    for c in sorted(c for c, r in ROLE.items() if r == "known"):
        m = np.where(cat == c)[0].copy(); rng.shuffle(m); k = int(round(0.7 * len(m)))
        ref += list(m[:k]); test += list(m[k:])
    ref, test = np.array(ref), np.array(test)
    sp, inv = np.unique(cat[ref], return_inverse=True)
    P = np.zeros((len(sp), X.shape[1]), np.float64); np.add.at(P, inv, X[ref]); P = l2(P).astype(np.float32)
    nov = np.where(np.array([ROLE[c] != "known" for c in cat]))[0]
    def sc(idx):
        s = np.empty(len(idx), np.float32)
        for i in range(0, len(idx), 4096):
            s[i:i + 4096] = 1 - (X[idx[i:i + 4096]] @ P.T).max(1)
        cs = np.unique(cat[idx]); return cs, np.array([s[cat[idx] == c].mean() for c in cs])
    ck_ids, ck = sc(test); cn_ids, cn = sc(nov)
    ok_ = np.array([ORD[c] for c in ck_ids]); on_ = np.array([ORD[c] for c in cn_ids])
    rows, wsum, wtot = {}, 0.0, 0
    for o, n_nov in Counter(on_.tolist()).most_common():
        a, b = ck[ok_ == o], cn[on_ == o]
        if len(a) >= 10 and len(b) >= 10:
            v = float(auroc(a, b)); rows[o] = {"known": len(a), "novel": len(b), "auroc": v,
                                              "mean_known": float(a.mean()), "mean_novel": float(b.mean())}
            wsum += v * len(b); wtot += len(b)
    res = {"pooled": float(auroc(ck, cn)), "within_order_weighted": wsum / wtot, "by_order": rows}
    print(f"[{name}] pooled {res['pooled']:.3f} | within-order (novel-weighted) {res['within_order_weighted']:.3f}")
    for o, v in rows.items():
        print(f"    {o:<14} known {v['known']:>5} novel {v['novel']:>5}  AUROC {v['auroc']:.3f}  "
              f"mean score known {v['mean_known']:.3f} novel {v['mean_novel']:.3f}")
    out[name] = res; del X
save("b1b_router_by_order.json", out)

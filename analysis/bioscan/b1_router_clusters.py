"""B1. Router and novel-species clustering on BIOSCAN, for concat, BIOSCAN fusion and iNat fusion (transfer).
Router protocols:
  P0  notebook reproduction: known test = 30% of images of ALL known species (98% were in fusion training).
  P1  leak-free: known test = 30% held-out images of the 120 fusion-VALIDATION species (never trained on).
  P2  P1 with novel clusters subsampled to the known-test cluster sizes (removes a cluster-size confound).
Clustering: KMeans with oracle k on all novel images; NMI, ARI, pair-F1, each next to the all-singletons
partition; plus 1-NN species purity (needs no k)."""
import os, sys, time
from common_b import *
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from sklearn.cluster import KMeans

S, BY = load_split()
ROLE = {c: s["role"] for c, s in BY.items()}
SRC = {c: s["holdout"] for c, s in BY.items()}
VAL = set(fusion_folds()["val"])
reps = sys.argv[1:] or REPS

def auroc(neg, pos):
    r = rankdata(np.r_[neg, pos]); n0, n1 = len(neg), len(pos)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

def boot_ci(neg, pos, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = [auroc(rng.choice(neg, len(neg)), rng.choice(pos, len(pos))) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def pair_f1(y, p):
    from sklearn.metrics.cluster import contingency_matrix
    M = contingency_matrix(y, p, sparse=True)
    c2 = lambda x: (x * (x - 1) / 2).sum()
    tp = c2(M.data.astype(np.float64))
    pp = c2(np.asarray(M.sum(0)).ravel().astype(np.float64)); tt = c2(np.asarray(M.sum(1)).ravel().astype(np.float64))
    pr = tp / pp if pp else 0.0; rc = tp / tt if tt else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc

def router(X, cat, test_species):
    rng = np.random.default_rng(SEED)
    known = sorted(c for c, r in ROLE.items() if r == "known")
    ref, test = [], []
    for c in known:
        m = np.where(cat == c)[0]
        if c in test_species:
            m = m.copy(); rng.shuffle(m); k = int(round(0.7 * len(m)))
            ref += list(m[:k]); test += list(m[k:])
        else:
            ref += list(m)
    ref, test = np.array(ref), np.array(test)
    sp, inv = np.unique(cat[ref], return_inverse=True)
    P = np.zeros((len(sp), X.shape[1]), np.float64); np.add.at(P, inv, X[ref]); P = l2(P).astype(np.float32)
    nov = np.where(np.array([ROLE[c] != "known" for c in cat]))[0]
    def score(idx):
        s, a = np.empty(len(idx), np.float32), np.empty(len(idx), np.int64)
        for i in range(0, len(idx), 4096):
            sim = X[idx[i:i + 4096]] @ P.T
            a[i:i + 4096] = sim.argmax(1); s[i:i + 4096] = 1 - sim.max(1)
        return s, sp[a]
    st, nn_t = score(test); sn, _ = score(nov)
    own = float(np.mean(nn_t == cat[test]))
    def clus(s, idx):
        cs = np.unique(cat[idx]); return np.array([s[cat[idx] == c].mean() for c in cs]), cs
    ck, _ = clus(st, test); cn, csn = clus(sn, nov)
    res = {"n_known_test_clusters": len(ck), "n_novel_clusters": len(cn),
           "auroc_image": float(auroc(st, sn)), "auroc_cluster": float(auroc(ck, cn)),
           "auroc_cluster_CI": boot_ci(ck, cn), "known_test_top1_own_species": own,
           "mean_score_known": float(ck.mean()), "mean_score_novel": float(cn.mean())}
    for src in ["bioscan_unseen", "bioscan_other_heldout"]:
        m = np.array([SRC[c] == src for c in csn]); res[f"auroc_cluster_{src}"] = float(auroc(ck, cn[m]))
    # P2: novel clusters subsampled to the size distribution of known-test clusters
    sizes = np.array([np.sum(cat[test] == c) for c in np.unique(cat[test])])
    rng2 = np.random.default_rng(SEED + 1); cn2 = []
    for c in csn:
        m = np.where(cat[nov] == c)[0]; k = min(len(m), int(rng2.choice(sizes)))
        cn2.append(sn[rng2.choice(m, k, replace=False)].mean())
    res["auroc_cluster_size_matched"] = float(auroc(ck, np.array(cn2)))
    return res

def clustering(X, cat):
    idx = np.where(np.array([ROLE[c] == "novel_known_branch" for c in cat]))[0]
    y = cat[idx]; k = len(np.unique(y)); Xn = X[idx]
    t0 = time.time()
    pred = KMeans(k, random_state=SEED, n_init=5).fit_predict(Xn)
    f1, pr, rc = pair_f1(y, pred)
    # 1-NN species purity (leave-one-out, no k needed)
    nn_ok = 0
    for i in range(0, len(idx), 2048):
        sim = Xn[i:i + 2048] @ Xn.T
        sim[np.arange(sim.shape[0]), np.arange(i, i + sim.shape[0])] = -9
        nn_ok += int(np.sum(y[sim.argmax(1)] == y[i:i + sim.shape[0]]))
    single = np.arange(len(y))
    return {"images": len(y), "k": k, "kmeans_seconds": time.time() - t0,
            "NMI": NMI(y, pred), "ARI": ARI(y, pred), "pairF1": f1, "pairP": pr, "pairR": rc,
            "trivial_singletons": {"NMI": NMI(y, single), "ARI": ARI(y, single), "pairF1": 0.0},
            "nn1_species_purity": nn_ok / len(y)}

out = {}
for name in reps:
    X, cat = load_rep(name)
    known_all = {c for c, r in ROLE.items() if r == "known"}
    r = {"router_P0_notebook": router(X, cat, known_all), "router_P1_leakfree": router(X, cat, VAL)}
    p0, p1 = r["router_P0_notebook"], r["router_P1_leakfree"]
    print(f"[{name}] router P0 cluster {p0['auroc_cluster']:.3f} | P1 leak-free {p1['auroc_cluster']:.3f} "
          f"CI {np.round(p1['auroc_cluster_CI'],3).tolist()} | P1 size-matched {p1['auroc_cluster_size_matched']:.3f} "
          f"| P1 image {p1['auroc_image']:.3f} | own-top1 P0 {p0['known_test_top1_own_species']:.3f} P1 {p1['known_test_top1_own_species']:.3f}", flush=True)
    if os.environ.get("SKIP_CLUSTER") == "1":
        out[name] = r; save(f"b1_{name}_router.json", r); del X; continue
    r["clustering"] = c = clustering(X, cat)
    t = c["trivial_singletons"]
    print(f"[{name}] NMI {c['NMI']:.3f} (singletons {t['NMI']:.3f}) | ARI {c['ARI']:.3f} | pair-F1 {c['pairF1']:.3f} "
          f"| 1-NN purity {c['nn1_species_purity']:.3f} | kmeans {c['kmeans_seconds']:.0f}s", flush=True)
    out[name] = r
    save(f"b1_{name}.json", r)
    del X

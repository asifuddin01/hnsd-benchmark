"""A1. Router AUROC and novel-species cluster quality on the v2 benchmark: trained fusion vs concatenation."""
from common import *
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score as NMI
from sklearn.cluster import KMeans

S, BY = load_split()
role = {c: s["role"] for c, s in BY.items()}

def evaluate(X, cat, label):
    rng = np.random.default_rng(SEED)
    known = sorted(c for c, r in role.items() if r == "known")
    ref, test = [], []
    for c in known:
        m = np.where(cat == c)[0].copy(); rng.shuffle(m)
        k = int(round(0.7 * len(m))); ref += list(m[:k]); test += list(m[k:])
    ref, test = np.array(ref), np.array(test)
    sp = np.unique(cat[ref])
    protos = l2(np.stack([X[ref[cat[ref] == c]].mean(0) for c in sp]))
    roles = np.array([role[c] for c in cat])
    nov = np.where(np.isin(roles, ["novel_known_branch", "novel_unseen_branch"]))[0]
    s_test = 1 - (X[test] @ protos.T).max(1)
    s_nov = 1 - (X[nov] @ protos.T).max(1)
    img_auc = roc_auc_score(np.r_[np.zeros(len(test)), np.ones(len(nov))], np.r_[s_test, s_nov])
    clus = lambda s, idx: np.array([s[cat[idx] == c].mean() for c in np.unique(cat[idx])])
    ck, cn = clus(s_test, test), clus(s_nov, nov)
    cl_auc = roc_auc_score(np.r_[np.zeros(len(ck)), np.ones(len(cn))], np.r_[ck, cn])
    res = {"router_auroc_image": img_auc, "router_auroc_cluster": cl_auc}
    for name, mask in [("novel_known_branch", roles == "novel_known_branch"),
                       ("novel_unseen_branch", roles == "novel_unseen_branch")]:
        idx = np.where(mask)[0]; k = len(np.unique(cat[idx]))
        pred = KMeans(k, random_state=SEED, n_init=3).fit_predict(X[idx])
        res[f"nmi_{name}"] = NMI(cat[idx], pred)
    print(f"{label:<24} router img {img_auc:.4f} | cluster {cl_auc:.4f} | "
          f"NMI place {res['nmi_novel_known_branch']:.4f} | NMI construct {res['nmi_novel_unseen_branch']:.4f}", flush=True)
    return res

Xf, cf = load_feats(D / "features_trifusion_v2.npz")
Xc, cc = concat_feats()
assert np.array_equal(cf, cc)
out = {"concat": evaluate(Xc, cc, "concat (training-free)"),
       "fusion": evaluate(Xf, cf, "trained fusion")}
save("a1_router_clusters.json", out)

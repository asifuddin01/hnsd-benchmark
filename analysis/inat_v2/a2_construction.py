"""A2/A3. Sub-taxonomy construction with absolute vs quantile-transferred rank radii.
A2: main split (trained fusion), per hold-out mechanism.
A3: five per-split fusions (one hold-out pair each), plus the concatenation for comparison."""
import sys
from common import *
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI

CUT = ["genus", "family", "order", "class"]

def pairs(n):
    return np.triu_indices(n, 1)

def radii(Ck, cdk):
    i, j = pairs(len(Ck)); s = (Ck @ Ck.T)[i, j]; lv = lca_levels(cdk, i, j)
    tau = {r: float(np.percentile(s[lv == k], 25)) for k, r in enumerate(CUT) if (lv == k).sum() >= 5}
    srt = np.sort(s)
    q = {r: float(np.searchsorted(srt, t) / len(srt)) for r, t in tau.items()}
    return tau, q, s

def pair_f1(pred, true):
    i, j = pairs(len(pred))
    ps, ts = pred[i] == pred[j], true[i] == true[j]
    tp = (ps & ts).sum()
    p = tp / max(ps.sum(), 1); r = tp / max(ts.sum(), 1)
    return 2 * p * r / max(p + r, 1e-12)

def construct(Cu, recs_u, tau):
    D_ = np.clip(1 - Cu @ Cu.T, 0, None); np.fill_diagonal(D_, 0); D_ = (D_ + D_.T) / 2
    Z = linkage(squareform(D_, checks=False), "average")
    cdu = codes(recs_u); i, j = pairs(len(Cu))
    coph = spearmanr(cophenet(Z), lca_levels(cdu, i, j)).correlation
    res = {"cophenetic": float(coph)}
    for r in CUT:
        if r not in tau:
            continue
        pred = fcluster(Z, t=1 - tau[r], criterion="distance")
        true = cdu[r]
        res[r] = {"nmi": NMI(true, pred), "ari": ARI(true, pred), "pair_f1": pair_f1(pred, true),
                  "n_pred": int(len(set(pred))), "n_true": int(len(set(true))),
                  "singleton_nmi": NMI(true, np.arange(len(true)))}
    return res

def run(C_by_id, known_ids, unseen_ids, BY):
    kr = [BY[c] for c in known_ids]; ur = [BY[c] for c in unseen_ids]
    Ck = np.stack([C_by_id[c] for c in known_ids]); Cu = np.stack([C_by_id[c] for c in unseen_ids])
    tau, q, sk = radii(Ck, codes(kr))
    iu, ju = pairs(len(Cu)); su = (Cu @ Cu.T)[iu, ju]
    tau_q = {r: float(np.quantile(su, qq)) for r, qq in q.items()}
    return {"n_unseen": len(unseen_ids), "shift": float(su.mean() - sk.mean()),
            "absolute": construct(Cu, ur, tau), "quantile": construct(Cu, ur, tau_q)}

def show(tag, r):
    a, qn = r["absolute"]["genus"], r["quantile"]["genus"]
    print(f"{tag:<30} n={r['n_unseen']:>3} shift {r['shift']:+.3f} | genera true {qn['n_true']:>3} "
          f"abs {a['n_pred']:>3} quant {qn['n_pred']:>3} | NMI abs {a['nmi']:.3f} quant {qn['nmi']:.3f} "
          f"(singletons {qn['singleton_nmi']:.3f}) | pairF1 abs {a['pair_f1']:.3f} quant {qn['pair_f1']:.3f} "
          f"| coph {r['quantile']['cophenetic']:.3f}", flush=True)

S, BY = load_split()
mode = sys.argv[1] if len(sys.argv) > 1 else "main"

if mode == "main":
    X, cat = load_feats(D / "features_trifusion_v2.npz")
    ids, C = centroids(X, cat); Cm = dict(zip(ids.tolist(), C))
    known = sorted(c for c, s in BY.items() if s["role"] == "known")
    subsets = {
        "order hold-outs":   [c for c, s in BY.items() if s["holdout"] == "order"],
        "Fungi":             [c for c, s in BY.items() if s["holdout"] == "supercategory" and s["supercategory"] == "Fungi"],
        "Reptiles":          [c for c, s in BY.items() if s["holdout"] == "supercategory" and s["supercategory"] == "Reptiles"],
        "all unseen-branch": [c for c, s in BY.items() if s["role"] == "novel_unseen_branch"],
    }
    out = {}
    for tag, u in subsets.items():
        out[tag] = run(Cm, known, sorted(u), BY); show(tag, out[tag])
    save("a2_construction_main.json", out)

else:   # alt splits: trained per-split fusion vs concatenation
    pairs_ = json.load(open(SPLITS_INAT / "alt_splits.json"))
    Xc, cc = concat_feats(); idc, Cc = centroids(Xc, cc); Ccm = dict(zip(idc.tolist(), Cc))
    out = {}
    for pr in pairs_:
        tag = "+".join(pr).replace(" ", "_")
        Xf, cf = load_feats(D / "alt_splits" / f"features_trifusion_{tag}.npz")
        idf, Cf = centroids(Xf, cf); Cfm = dict(zip(idf.tolist(), Cf))
        unseen = sorted(c for c, s in BY.items() if s["supercategory"] in pr)
        known = sorted(c for c, s in BY.items() if s["supercategory"] not in pr)
        out[tag] = {"fusion": run(Cfm, known, unseen, BY), "concat": run(Ccm, known, unseen, BY)}
        show(tag + " [fusion]", out[tag]["fusion"]); show(tag + " [concat]", out[tag]["concat"])
    for rep in ["fusion", "concat"]:
        sh = [out[t][rep]["shift"] for t in out]
        gain = [out[t][rep]["quantile"]["genus"]["nmi"] - out[t][rep]["absolute"]["genus"]["nmi"] for t in out]
        qn = [out[t][rep]["quantile"]["genus"]["nmi"] for t in out]
        r, p = pearsonr(sh, gain)
        print(f"[{rep}] quantile genus NMI {np.mean(qn):.3f} +/- {np.std(qn):.3f} | "
              f"gain vs shift r={r:.3f} p={p:.3f} | wins {sum(g > 0 for g in gain)}/5", flush=True)
    save("a3_construction_altsplits.json", out)

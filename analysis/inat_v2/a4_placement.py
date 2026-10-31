"""A4. Placement of the 180 novel-known species WITHOUT the evaluation-set rank prior.
Pseudo-training examples are built from known species only. The rank prior is varied:
natural (leakage-free), uniform, a deliberately mismatched order-heavy prior, and the old evaluation
prior (leaky, for comparison only). Trivial baselines are reported alongside."""
from common import *
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

PL = ["order", "family", "genus"]                     # placement ranks, coarse -> fine
DEPTH = {r: i for i, r in enumerate(PL)}
S, BY = load_split()
X, cat = load_feats(D / "features_trifusion_v2.npz")
ids, C = centroids(X, cat); Cm = dict(zip(ids.tolist(), C))
known = sorted(c for c, s in BY.items() if s["role"] == "known")
test  = sorted(c for c, s in BY.items() if s["role"] == "novel_known_branch")
recs = [BY[c] for c in known + test]
cd = codes(recs, PL)                                   # shared taxon codes for known + test
nk = len(known)
Ck = np.stack([Cm[c] for c in known]).astype(np.float64)
K = {r: cd[r][:nk] for r in PL}                        # known species' taxon codes
T = {r: cd[r][nk:] for r in PL}                        # test species' taxon codes
SUM = {r: np.zeros((cd[r].max() + 1, Ck.shape[1])) for r in PL}
CNT = {r: np.zeros(cd[r].max() + 1) for r in PL}
for r in PL:
    np.add.at(SUM[r], K[r], Ck); np.add.at(CNT[r], K[r], 1)
rep_of = {r: {} for r in PL}                           # taxon code -> lineage codes (order, family)
for i in range(len(recs)):
    for r in PL:
        rep_of[r].setdefault(cd[r][i], {x: cd[x][i] for x in PL[:DEPTH[r] + 1]})

def signals(q, excl):
    feats, tops = [], {}
    sims_sp = Ck @ q; sims_sp[excl] = -np.inf
    order_sp = np.argsort(-sims_sp)
    for r in PL:
        S_ = SUM[r].copy(); N_ = CNT[r].copy()
        if len(excl):
            np.subtract.at(S_, K[r][excl], Ck[excl]); np.subtract.at(N_, K[r][excl], 1)
        sims = (S_ @ q) / (np.linalg.norm(S_, axis=1) + 1e-12)
        sims[N_ <= 0] = -np.inf
        o = np.argsort(-sims); s1, s2 = sims[o[0]], sims[o[1]]
        valid = sims[np.isfinite(sims)]
        med = np.median(valid); mad = np.median(np.abs(valid - med)) * 1.4826 + 1e-6
        coh = np.mean(K[r][order_sp[:10]] == o[0])
        feats += [s1, s2, s1 - s2, (s1 - med) / mad, coh]
        tops[r] = o[0]
    n1, n2 = sims_sp[order_sp[0]], sims_sp[order_sp[1]]
    feats += [n1, n2, n1 - n2]
    return np.array(feats), tops, order_sp[0]

# ---------- pseudo-training examples from known species only ----------
Xtr, ytr = [], []
for i in range(nk):
    for r, grp in [("genus", "genus"), ("family", "genus"), ("order", "family")]:
        excl = np.array([i]) if r == "genus" else np.where(K[grp] == K[grp][i])[0]
        parent_rank = r
        others = np.setdiff1d(np.where(K[parent_rank] == K[parent_rank][i])[0], excl)
        if len(others) == 0:
            continue                                   # target rank not constructible for this species
        f, _, _ = signals(Ck[i], excl); Xtr.append(f); ytr.append(DEPTH[r])
Xtr, ytr = np.array(Xtr), np.array(ytr)
nat = Counter(ytr.tolist())
print("pseudo-training examples:", {PL[k]: v for k, v in sorted(nat.items())}, flush=True)

# ---------- test signals: reference = all known species ----------
Xte, tops_te, nn_te = [], [], []
for t in range(len(test)):
    f, tops, nn = signals(np.asarray(Cm[test[t]], dtype=np.float64), np.array([], dtype=int))
    Xte.append(f); tops_te.append(tops); nn_te.append(nn)
Xte = np.array(Xte)
true_rank = np.array([DEPTH[BY[c]["true_rank"]] for c in test])
true_code = {r: T[r] for r in PL}
ev = Counter(true_rank.tolist())
print("test true ranks:", {PL[k]: v for k, v in sorted(ev.items())}, flush=True)

def score(pred_rank, pred_code):
    rank_ok = pred_rank == true_rank
    par_ok = np.array([rank_ok[t] and pred_code[t] == true_code[PL[true_rank[t]]][t] for t in range(len(test))])
    tp = fp = fn = 0
    for t in range(len(test)):
        Tset = {(x, true_code[x][t]) for x in PL[:true_rank[t] + 1]}
        lin = rep_of[PL[pred_rank[t]]][pred_code[t]]
        Pset = {(x, lin[x]) for x in PL[:pred_rank[t] + 1]}
        tp += len(Tset & Pset); fp += len(Pset - Tset); fn += len(Tset - Pset)
    hp, hr = tp / (tp + fp), tp / (tp + fn)
    k = int(par_ok.sum()); lo, hi = wilson(k, len(test))
    return {"RankAcc": float(rank_ok.mean()), "ParentAcc": k / len(test), "ParentAcc_CI": [lo, hi],
            "hF1": 2 * hp * hr / (hp + hr), "RankErr": float(np.abs(pred_rank - true_rank).mean())}

def at_rank(pr):
    pr = np.asarray(pr); return pr, [tops_te[t][PL[pr[t]]] for t in range(len(test))]

res = {}
for r in PL:
    res[f"constant {r}"] = score(*at_rank([DEPTH[r]] * len(test)))
res["nearest species (genus of NN)"] = score(np.full(len(test), 2), [K["genus"][nn_te[t]] for t in range(len(test))])

# radius rule: training-free, prior-free, uses the paper's own rank radii
i, j = np.triu_indices(nk, 1); s_pair = (Ck @ Ck.T)[i, j]
lv = np.full(len(i), 3)
for k in [0, 1, 2]:                                    # coarse -> fine, so finer ranks overwrite
    lv[K[PL[k]][i] == K[PL[k]][j]] = 2 - k            # 0 genus, 1 family, 2 order
tau = {r: np.percentile(s_pair[lv == (2 - DEPTH[r])], 25) for r in PL}
nn_sim = Xte[:, -3]
rule = np.where(nn_sim >= tau["genus"], 2, np.where(nn_sim >= tau["family"], 1, 0))
res["radius rule (no training)"] = score(*at_rank(rule))

priors = {"natural (leakage-free)": None, "uniform": "balanced",
          "mismatched, order-heavy": {0: 0.6, 1: 0.25, 2: 0.15},
          "evaluation prior (LEAKY)": {k: ev[k] / len(test) for k in ev}}
for name, pr in priors.items():
    w = None; cw = None
    if pr == "balanced":
        cw = "balanced"
    elif isinstance(pr, dict):
        w = np.array([pr[y] / (nat[y] / len(ytr)) for y in ytr])
    clf = RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=SEED, class_weight=cw)
    clf.fit(Xtr, ytr, sample_weight=w)
    res[f"learned, {name}"] = score(*at_rank(clf.predict(Xte)))

print(f"\n{'method':<38}{'RankAcc':>8}{'ParentAcc':>10}{'95% CI':>16}{'hF1':>7}{'RankErr':>8}")
for k, v in res.items():
    ci = f"[{v['ParentAcc_CI'][0]:.2f},{v['ParentAcc_CI'][1]:.2f}]"
    print(f"{k:<38}{v['RankAcc']:>8.3f}{v['ParentAcc']:>10.3f}{ci:>16}{v['hF1']:>7.3f}{v['RankErr']:>8.3f}")
save("a4_placement.json", {"pseudo_prior": {PL[k]: v for k, v in nat.items()},
                           "test_prior": {PL[k]: v for k, v in ev.items()}, "tau": tau, "results": res})

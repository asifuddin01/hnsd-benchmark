"""B2. Placement of BIOSCAN's real undescribed species (2,914 novel_known_branch), same protocol as v2 A4:
pseudo-training examples from known species only, rank prior varied, trivial baselines alongside.
Reported overall and per source (bioscan_unseen / bioscan_other_heldout), for each representation.
Note: the BIOSCAN fusion was trained on 6,411 of the 6,531 known species, so its pseudo-training signals come
from fitted species while test signals come from unseen ones; concat and the iNat fusion have no such shift."""
import sys, time
from common_b import *
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

PL = ["order", "family", "genus"]
DEPTH = {r: i for i, r in enumerate(PL)}
S, BY = load_split()
known = sorted(c for c, s in BY.items() if s["role"] == "known")
test  = sorted(c for c, s in BY.items() if s["role"] == "novel_known_branch")
recs = [BY[c] for c in known + test]
cd = codes(recs, PL); nk = len(known)
K = {r: cd[r][:nk] for r in PL}; T = {r: cd[r][nk:] for r in PL}
src = np.array([BY[c]["holdout"] for c in test])
true_rank = np.array([DEPTH[BY[c]["true_rank"]] for c in test])
rep_of = {r: {} for r in PL}
for i in range(len(recs)):
    for r in PL:
        rep_of[r].setdefault(cd[r][i], {x: cd[x][i] for x in PL[:DEPTH[r] + 1]})

def run(name):
    t0 = time.time()
    X, cat = load_rep(name)
    ids, inv = np.unique(cat, return_inverse=True)
    C = np.zeros((len(ids), X.shape[1]), np.float64); np.add.at(C, inv, X); C = l2(C)
    del X
    pos = {c: i for i, c in enumerate(ids.tolist())}
    Ck = C[[pos[c] for c in known]]; Ct = C[[pos[c] for c in test]]
    SUM = {r: np.zeros((cd[r].max() + 1, Ck.shape[1])) for r in PL}
    CNT = {r: np.zeros(cd[r].max() + 1) for r in PL}
    for r in PL:
        np.add.at(SUM[r], K[r], Ck); np.add.at(CNT[r], K[r], 1)
    NRM = {r: np.linalg.norm(SUM[r], axis=1) + 1e-12 for r in PL}

    def signals(q, excl):
        feats, tops = [], {}
        sims_sp = Ck @ q
        if len(excl):
            sims_sp[excl] = -np.inf
        top10 = np.argpartition(-sims_sp, 10)[:10]; top10 = top10[np.argsort(-sims_sp[top10])]
        for r in PL:
            sims = (SUM[r] @ q) / NRM[r]
            sims[CNT[r] <= 0] = -np.inf
            if len(excl):
                for t in np.unique(K[r][excl]):
                    e = excl[K[r][excl] == t]; n = CNT[r][t] - len(e)
                    if n <= 0:
                        sims[t] = -np.inf
                    else:
                        v = SUM[r][t] - Ck[e].sum(0); sims[t] = v @ q / (np.linalg.norm(v) + 1e-12)
            o2 = np.argpartition(-sims, 1)[:2]; o2 = o2[np.argsort(-sims[o2])]
            s1, s2 = sims[o2[0]], sims[o2[1]]
            valid = sims[np.isfinite(sims)]
            med = np.median(valid); mad = np.median(np.abs(valid - med)) * 1.4826 + 1e-6
            coh = np.mean(K[r][top10] == o2[0])
            feats += [s1, s2, s1 - s2, (s1 - med) / mad, coh]
            tops[r] = o2[0]
        n1, n2 = sims_sp[top10[0]], sims_sp[top10[1]]
        feats += [n1, n2, n1 - n2]
        return np.array(feats), tops, top10[0]

    Xtr, ytr = [], []
    for i in range(nk):
        for r, grp in [("genus", "genus"), ("family", "genus"), ("order", "family")]:
            excl = np.array([i]) if r == "genus" else np.where(K[grp] == K[grp][i])[0]
            others = np.setdiff1d(np.where(K[r] == K[r][i])[0], excl)
            if len(others) == 0:
                continue
            f, _, _ = signals(Ck[i], excl); Xtr.append(f); ytr.append(DEPTH[r])
    Xtr, ytr = np.array(Xtr), np.array(ytr); nat = Counter(ytr.tolist())
    print(f"[{name}] pseudo-training {dict((PL[k], v) for k, v in sorted(nat.items()))} ({time.time()-t0:.0f}s)", flush=True)

    Xte, tops_te, nn_te = [], [], []
    for t in range(len(test)):
        f, tops, nn = signals(Ct[t], np.array([], dtype=int)); Xte.append(f); tops_te.append(tops); nn_te.append(nn)
    Xte = np.array(Xte); ev = Counter(true_rank.tolist())

    def score(pred_rank, pred_code, mask):
        idx = np.where(mask)[0]
        rank_ok = pred_rank == true_rank
        par_ok = np.array([rank_ok[t] and pred_code[t] == T[PL[true_rank[t]]][t] for t in idx])
        tp = fp = fn = 0
        for t in idx:
            Ts = {(x, T[x][t]) for x in PL[:true_rank[t] + 1]}
            lin = rep_of[PL[pred_rank[t]]][pred_code[t]]
            Ps = {(x, lin[x]) for x in PL[:pred_rank[t] + 1]}
            tp += len(Ts & Ps); fp += len(Ps - Ts); fn += len(Ts - Ps)
        hp, hr = tp / (tp + fp), tp / (tp + fn); k = int(par_ok.sum())
        return {"n": len(idx), "RankAcc": float(rank_ok[idx].mean()), "ParentAcc": k / len(idx),
                "ParentAcc_CI": list(wilson(k, len(idx))), "hF1": 2 * hp * hr / (hp + hr),
                "pred_rank_dist": {PL[a]: int(b) for a, b in sorted(Counter(pred_rank[idx].tolist()).items())}}

    subsets = {"all": np.ones(len(test), bool), "unseen": src == "bioscan_unseen", "other_heldout": src == "bioscan_other_heldout"}
    def evalm(pr, pc):
        return {s: score(np.asarray(pr), pc, m) for s, m in subsets.items()}
    def at_rank(pr):
        pr = np.asarray(pr); return pr, [tops_te[t][PL[pr[t]]] for t in range(len(test))]

    res = {}
    for r in PL:
        res[f"constant {r}"] = evalm(*at_rank([DEPTH[r]] * len(test)))
    res["nearest species (genus of NN)"] = evalm(np.full(len(test), 2), [K["genus"][nn_te[t]] for t in range(len(test))])
    # radius rule: 25th percentile of within-rank known centroid similarities, computed blockwise
    lv_s = {0: [], 1: [], 2: []}
    for i in range(0, nk, 512):
        sim = Ck[i:i + 512] @ Ck.T
        for a in range(sim.shape[0]):
            g = i + a; j = np.arange(g + 1, nk); s = sim[a, g + 1:]
            lv = np.full(len(j), 3)
            for kk in [0, 1, 2]:
                lv[K[PL[kk]][j] == K[PL[kk]][g]] = 2 - kk
            for L in (0, 1, 2):
                lv_s[L].append(s[lv == L].astype(np.float32))
    tau = {r: float(np.percentile(np.concatenate(lv_s[2 - DEPTH[r]]), 25)) for r in PL}
    del lv_s
    nn_sim = Xte[:, -3]
    rule = np.where(nn_sim >= tau["genus"], 2, np.where(nn_sim >= tau["family"], 1, 0))
    res["radius rule (no training)"] = evalm(*at_rank(rule))
    priors = {"natural (leakage-free)": None, "uniform": "balanced",
              "mismatched, order-heavy": {0: 0.6, 1: 0.25, 2: 0.15},
              "evaluation prior (LEAKY)": {k: ev[k] / len(test) for k in ev}}
    for pn, pr in priors.items():
        w = cw = None
        if pr == "balanced":
            cw = "balanced"
        elif isinstance(pr, dict):
            w = np.array([pr.get(y, 0.0) / (nat[y] / len(ytr)) for y in ytr])
        clf = RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=SEED, class_weight=cw)
        clf.fit(Xtr, ytr, sample_weight=w)
        res[f"learned, {pn}"] = evalm(*at_rank(clf.predict(Xte)))
    print(f"\n[{name}]  ({time.time()-t0:.0f}s)  test ranks {dict((PL[k], v) for k, v in sorted(ev.items()))}  tau {np.round(list(tau.values()),3).tolist()}")
    print(f"{'method':<34}" + "".join(f"{s+' Rank/Par/hF1':>28}" for s in subsets))
    for k, v in res.items():
        print(f"{k:<34}" + "".join(f"{v[s]['RankAcc']:>12.3f}{v[s]['ParentAcc']:>8.3f}{v[s]['hF1']:>8.3f}" for s in subsets), flush=True)
    save(f"b2_placement_{name}.json", {"pseudo_prior": {PL[k]: v for k, v in nat.items()},
         "test_prior": {PL[k]: v for k, v in ev.items()}, "tau": tau, "results": res})

for name in (sys.argv[1:] or REPS):
    run(name)

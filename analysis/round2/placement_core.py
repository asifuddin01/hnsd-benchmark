"""Placement core shared by the v2 and BIOSCAN drivers (same signals and baselines as A4 / B2), plus label-shift
estimation of the novel rank prior from the UNLABELLED novel species:
  EM   (Saerens, Latinne & Decaestecker 2002): reweight the classifier's posteriors until the prior is a fixed point;
  BBSE (Lipton, Wang & Smola 2018): solve C w = mu with the confusion matrix C from out-of-fold predictions on the
        pseudo-training set and mu the predicted-label distribution on the novel set.
Neither uses any test label. The estimated prior then reweights the pseudo-training examples (as in A4)."""
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

PL = ["order", "family", "genus"]; DEPTH = {r: i for i, r in enumerate(PL)}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]

class Placement:
    def __init__(self, Ck, Ct, K, T, rep_of, true_rank, seed=42):
        self.Ck, self.Ct, self.K, self.T, self.rep_of, self.true_rank, self.seed = Ck, Ct, K, T, rep_of, true_rank, seed
        self.SUM = {r: np.zeros((max(K[r].max(), T[r].max()) + 1, Ck.shape[1])) for r in PL}
        self.CNT = {r: np.zeros(len(self.SUM[r])) for r in PL}
        for r in PL:
            np.add.at(self.SUM[r], K[r], Ck); np.add.at(self.CNT[r], K[r], 1)
        self.NRM = {r: np.linalg.norm(self.SUM[r], axis=1) + 1e-12 for r in PL}
        self._build()

    def signals(self, q, excl):
        Ck, K = self.Ck, self.K; feats, tops = [], {}
        sims_sp = Ck @ q
        if len(excl):
            sims_sp[excl] = -np.inf
        top10 = np.argpartition(-sims_sp, 10)[:10]; top10 = top10[np.argsort(-sims_sp[top10])]
        for r in PL:
            sims = (self.SUM[r] @ q) / self.NRM[r]; sims[self.CNT[r] <= 0] = -np.inf
            if len(excl):
                for t in np.unique(K[r][excl]):
                    e = excl[K[r][excl] == t]; n = self.CNT[r][t] - len(e)
                    if n <= 0:
                        sims[t] = -np.inf
                    else:
                        v = self.SUM[r][t] - Ck[e].sum(0); sims[t] = v @ q / (np.linalg.norm(v) + 1e-12)
            o2 = np.argpartition(-sims, 1)[:2]; o2 = o2[np.argsort(-sims[o2])]
            s1, s2 = sims[o2[0]], sims[o2[1]]; valid = sims[np.isfinite(sims)]
            med = np.median(valid); mad = np.median(np.abs(valid - med)) * 1.4826 + 1e-6
            feats += [s1, s2, s1 - s2, (s1 - med) / mad, np.mean(K[r][top10] == o2[0])]; tops[r] = o2[0]
        feats += [sims_sp[top10[0]], sims_sp[top10[1]], sims_sp[top10[0]] - sims_sp[top10[1]]]
        return np.array(feats), tops, top10[0]

    def _build(self):
        K, nk = self.K, len(self.Ck); Xtr, ytr, grp = [], [], []
        for i in range(nk):
            for r, g in [("genus", "genus"), ("family", "genus"), ("order", "family")]:
                excl = np.array([i]) if r == "genus" else np.where(K[g] == K[g][i])[0]
                if len(np.setdiff1d(np.where(K[r] == K[r][i])[0], excl)) == 0:
                    continue
                f, _, _ = self.signals(self.Ck[i], excl); Xtr.append(f); ytr.append(DEPTH[r]); grp.append(K["family"][i])
        self.Xtr, self.ytr, self.grp = np.array(Xtr), np.array(ytr), np.array(grp)
        self.nat = np.bincount(self.ytr, minlength=3) / len(self.ytr)
        Xte, tops, nn = [], [], []
        for t in range(len(self.Ct)):
            f, tp, n = self.signals(self.Ct[t], np.array([], dtype=int)); Xte.append(f); tops.append(tp); nn.append(n)
        self.Xte, self.tops, self.nn = np.array(Xte), tops, nn

    def rf(self, w=None, cw=None):
        clf = RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=self.seed, class_weight=cw)
        return clf.fit(self.Xtr, self.ytr, sample_weight=w)

    def weights_for(self, prior):
        prior = np.asarray(prior, float)
        return np.array([prior[y] / self.nat[y] for y in self.ytr])

    # ---------------- label-shift estimators (no test labels) ----------------
    def oof(self, n_splits=5):
        P = np.zeros((len(self.ytr), 3))
        for tr, te in GroupKFold(n_splits).split(self.Xtr, self.ytr, self.grp):   # grouped by family: no leakage
            clf = RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=self.seed)
            clf.fit(self.Xtr[tr], self.ytr[tr]); P[np.ix_(te, clf.classes_)] = clf.predict_proba(self.Xtr[te])
        return P

    def estimate_priors(self):
        base = self.rf(); Pt = np.zeros((len(self.Xte), 3)); Pt[:, base.classes_] = base.predict_proba(self.Xte)
        Ps = self.oof(); yhat_s = Ps.argmax(1); yhat_t = Pt.argmax(1)
        # BBSE: C[i, j] = P(yhat = i, y = j) on source; mu[i] = P(yhat = i) on target; pi_t = pi_s * w
        C = np.zeros((3, 3)); np.add.at(C, (yhat_s, self.ytr), 1); C /= len(self.ytr)
        mu = np.bincount(yhat_t, minlength=3) / len(yhat_t)
        w = np.clip(np.linalg.lstsq(C, mu, rcond=None)[0], 0, None)
        bbse = w * self.nat; bbse = bbse / bbse.sum() if bbse.sum() > 0 else self.nat.copy()
        # EM (Saerens): with raw RF posteriors, and with posteriors calibrated by a power (temperature) fitted on OOF
        def em(P, it=1000):
            pi = self.nat.copy()
            for _ in range(it):
                post = P * (pi / self.nat); post /= post.sum(1, keepdims=True) + 1e-12
                new = post.mean(0)
                if np.abs(new - pi).max() < 1e-8:
                    break
                pi = new
            return pi
        def nll(T, P, y):
            Q = np.clip(P, 1e-6, 1) ** (1 / T); Q /= Q.sum(1, keepdims=True); return -np.mean(np.log(Q[np.arange(len(y)), y]))
        Ts = np.linspace(0.3, 3, 28); T = Ts[np.argmin([nll(t, Ps, self.ytr) for t in Ts])]
        Ptc = np.clip(Pt, 1e-6, 1) ** (1 / T); Ptc /= Ptc.sum(1, keepdims=True)
        return {"BBSE": bbse, "EM (raw RF)": em(Pt), f"EM (calibrated, T={T:.2f})": em(Ptc)}, \
               {"oof_acc": float(np.mean(yhat_s == self.ytr)), "C": C.tolist(), "mu": mu.tolist()}

    # ---------------- scoring ----------------
    def score(self, pred_rank, pred_code, mask=None):
        tr, T = self.true_rank, self.T
        idx = np.arange(len(tr)) if mask is None else np.where(mask)[0]
        rank_ok = pred_rank == tr
        par_ok = np.array([rank_ok[t] and pred_code[t] == T[PL[tr[t]]][t] for t in idx])
        tp = fp = fn = 0
        for t in idx:
            Ts = {(x, T[x][t]) for x in PL[:tr[t] + 1]}
            lin = self.rep_of[PL[pred_rank[t]]][pred_code[t]]
            Ps = {(x, lin[x]) for x in PL[:pred_rank[t] + 1]}
            tp += len(Ts & Ps); fp += len(Ps - Ts); fn += len(Ts - Ps)
        hp, hr = tp / (tp + fp), tp / (tp + fn); k = int(par_ok.sum())
        return {"n": len(idx), "RankAcc": float(rank_ok[idx].mean()), "ParentAcc": k / len(idx),
                "ParentAcc_CI": wilson(k, len(idx)), "hF1": 2 * hp * hr / (hp + hr),
                "pred_rank_dist": {PL[a]: int(b) for a, b in sorted(Counter(pred_rank[idx].tolist()).items())}}

    def at_rank(self, pr):
        pr = np.asarray(pr); return pr, [self.tops[t][PL[pr[t]]] for t in range(len(pr))]

    def evaluate_all(self, subsets=None):
        subsets = subsets or {"all": None}
        ev = lambda pr, pc: {s: self.score(np.asarray(pr), pc, m) for s, m in subsets.items()}
        n = len(self.Xte); res = {}
        for r in PL:
            res[f"constant {r}"] = ev(*self.at_rank([DEPTH[r]] * n))
        res["nearest species"] = ev(np.full(n, 2), [self.K["genus"][self.nn[t]] for t in range(n)])
        true_prior = np.bincount(self.true_rank, minlength=3) / n
        est, diag = self.estimate_priors()
        priors = {"natural (pseudo-training)": self.nat, "uniform": np.ones(3) / 3}
        priors.update({f"estimated: {k}": v for k, v in est.items()})
        priors["evaluation prior (LEAKY, reference)"] = true_prior
        prior_report = {}
        for name, pr in priors.items():
            clf = self.rf(w=self.weights_for(pr)) if name != "natural (pseudo-training)" else self.rf()
            res[f"learned, {name}"] = ev(*self.at_rank(clf.predict(self.Xte)))
            prior_report[name] = {"prior(order,family,genus)": np.round(pr, 4).tolist(),
                                  "L1_to_true": float(np.abs(np.asarray(pr) - true_prior).sum())}
        return res, prior_report, diag, true_prior

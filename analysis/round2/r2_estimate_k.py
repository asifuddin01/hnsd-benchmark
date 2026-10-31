"""R2a. Estimate the number of classes k without the oracle (GCD, Vaze et al. 2022, Sec. 4.2):
plain k-means on all 90,850 images for a candidate k, scored by clustering accuracy on the 29,425 labelled
images; Brent's bounded search over k in [n_old, 4000]. The implied number of novel species is k - 1,177
(true: 640, true k = 1,817).  usage: python analysis/round2/r2_estimate_k.py fusion concat"""
import sys, time
from scipy.optimize import minimize_scalar
from reps import *
from gcdlib import kmeans, cluster_acc

y_lab = np.array([CLASS_OF[c] for _, c in LAB])
out = {}
for rep in (sys.argv[1:] or ["fusion"]):
    X = load(rep); cache = {}; t0 = time.time()
    def f(x):
        k = int(round(x))
        if k not in cache:
            p = kmeans(X, k)
            cache[k] = cluster_acc(y_lab, p[:NL])
            print(f"[{rep}] k={k:>5}  labelled ACC {cache[k]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        return -cache[k]
    r = minimize_scalar(f, bounds=(N_OLD, 4000), method="bounded", options={"xatol": 10, "maxiter": 25})
    k_hat = max(cache, key=cache.get)
    out[rep] = {"k_hat": k_hat, "novel_hat": k_hat - N_OLD, "k_true": K_TRUE, "novel_true": K_TRUE - N_OLD,
                "evaluations": {str(k): v for k, v in sorted(cache.items())}}
    print(f"[{rep}] estimated k = {k_hat} (novel {k_hat - N_OLD}; true k {K_TRUE}, novel {K_TRUE - N_OLD})", flush=True)
    save("r2_estimate_k.json", out); del X

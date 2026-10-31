# Results

Reference outputs are in `results/<part>/*.json`; the command for each table is in [REPRODUCE.md](REPRODUCE.md).
Single training seed (42). "Fusion" is the tri-encoder fusion (SigLIP 2 + DINOv3 + DINOv2) trained on known
species only; "concat" is the training-free concatenation of the same three frozen encoders. 95% confidence
intervals for accuracies are Wilson intervals.

**A note on metrics.** With 50 images per species (iNaturalist) or at most 10 (BIOSCAN), a partition that puts
every image in its own cluster already reaches a high NMI. NMI is therefore always reported next to that
all-singletons value, and ARI and pair-F1 (which score that partition 0) are the main clustering metrics.

---

## 1. iNaturalist v2

### 1.1 Detection and oracle-k clustering (`results/inat_v2/a1_router_clusters.json`)

| representation | router AUROC (image) | router AUROC (cluster) | NMI, placement species | NMI, construction species |
|---|---|---|---|---|
| concat | 0.864 | 0.894 | 0.864 | 0.798 |
| fusion | 0.915 | 0.958 | 0.917 | 0.825 |

### 1.2 Construction, main split (`a2_construction_main.json`)

Genus-level NMI matches the all-singletons partition and is not evidence of anything:

| subset | genus NMI (quantile radii) | genus NMI (all singletons) |
|---|---|---|
| order hold-outs | 0.947 | 0.948 |
| Fungi | 0.967 | 0.963 |
| Reptiles | 0.943 | 0.941 |
| all unseen-branch | 0.953 | 0.958 |

What the rank radii do achieve, at order level on the order hold-outs (21 true orders):

| radii | orders found | NMI (singletons 0.634) | ARI | pair-F1 |
|---|---|---|---|---|
| quantile-transferred | 21 | 0.917 | 0.870 | 0.883 |
| absolute | 7 | 0.783 | 0.625 | 0.676 |

Genus pair-F1 stays low (0.19–0.34 across subsets). Failure case: the four Reptile orders are split into 17 with
quantile radii.

### 1.3 Transfer across five hold-out pairs (`a3_construction_altsplits.json`)

- With absolute radii the trained fusions collapse to 2–3 orders where 23–55 exist; quantile radii give 13–33.
- Quantile radii beat absolute radii on order-level NMI in 5 of 5 pairs, and beat the all-singletons baseline in
  4 of 5 (not on Mammals + Plants).
- Pair-F1 gains are small on average (+0.07 genus, +0.02 order) and reverse on Fungi + Reptiles.
- The distribution shift between known and novel similarities does not predict the gain on pair-F1
  (genus r = −0.02, p = 0.97).

### 1.4 Placement of 180 species (`a4_placement.json`)

| method | rank accuracy | parent accuracy [95% CI] | hierarchical F1 |
|---|---|---|---|
| constant genus | 0.433 | 0.306 [0.24, 0.38] | 0.695 |
| nearest known species | 0.433 | 0.333 [0.27, 0.41] | 0.709 |
| radius rule (no training) | 0.533 | 0.339 [0.27, 0.41] | 0.719 |
| learned, prior from the known taxonomy | 0.494 | 0.367 [0.30, 0.44] | 0.726 |
| learned, uniform prior | 0.544 | 0.389 [0.32, 0.46] | 0.733 |
| learned, evaluation-set prior (oracle, reference only) | 0.611 | 0.400 [0.33, 0.47] | 0.738 |

The learned placer's gain over the nearest-neighbour baseline is inside the confidence interval: placement is
limited by the representation. Section 4.3 replaces the oracle prior with one estimated from unlabelled data.

---

## 2. BIOSCAN-5M

### 2.1 Audit (`results/bioscan/b0_audit.json`)

No chosen species has a missing taxonomy label, so no ground truth is censored by a missing label. Novel species
are 66% Hymenoptera; known species are 37% Diptera.

### 2.2 Detection (`b1_*.json`, `b1b_router_by_order.json`)

| representation | router AUROC (cluster), all known species | leak-free (120 validation species) [95% CI] | within each order |
|---|---|---|---|
| concat | 0.369 | 0.357 [0.30, 0.41] | 0.497 |
| BIOSCAN fusion | 0.823* | **0.758** [0.71, 0.81] | 0.829* |
| iNaturalist fusion (transfer) | 0.402 | 0.385 [0.34, 0.43] | 0.512 |

\*98% of the known test clusters in this protocol were in the BIOSCAN fusion's training data; use the leak-free
column. Below-chance pooled values are a composition effect: Hymenoptera images get the lowest novelty scores and
dominate the novel set, while within each order frozen encoders are at chance.

### 2.3 Clustering of novel images, oracle k = 2,914 (`b1c_cluster_*.json`)

| representation | NMI | all-singletons NMI | ARI | pair-F1 |
|---|---|---|---|---|
| concat | 0.750 | 0.885 | 0.058 | 0.058 |
| BIOSCAN fusion | 0.802 | 0.885 | 0.160 | 0.160 |
| iNaturalist fusion | 0.752 | 0.885 | 0.063 | 0.063 |

### 2.4 Placement of 2,914 species (`b2_placement_*.json`; parent accuracy)

| method | concat | BIOSCAN fusion | iNaturalist fusion |
|---|---|---|---|
| constant family | 0.310 | 0.306 | 0.289 |
| nearest known species | 0.171 | 0.182 | 0.162 |
| learned, prior from the known taxonomy | 0.260 | 0.333 | 0.261 |
| learned, evaluation-set prior (oracle) | 0.345 | 0.359 | 0.325 |

The prior implied by the known taxonomy (order 38%) is far from that of real novelty (order 1%); Section 4.3
estimates it from unlabelled data instead.

---

## 3. GCD baselines

All runs use `data/gcd/simgcd_split.json` (1,177 old species, 50% of their images labelled; 640 new species;
k = 1,817 as in the GCD protocol), 100 epochs, final epoch reported (no model selection).

### 3.1 GCD protocol (`results/gcd_baselines/s1_gcd_protocol.json`, `results/round2/r1_gcd_protocol.json`)

| method | All | Old | New | novel_known ARI | novel_unseen ARI |
|---|---|---|---|---|---|
| SimGCD, DINO ViT-B/16 (head) | 0.374 | 0.534 | 0.228 | 0.335 | 0.172 |
| SimGCD, DINOv2 ViT-B/14 reg4 (head) | 0.479 | 0.668 | 0.306 | 0.589 | 0.185 |
| HypCD, DINOv2 ViT-B/14 reg4 (head) | 0.453 | 0.637 | 0.284 | 0.583 | 0.158 |
| SS-k-means on SimGCD-DINO features | 0.385 | 0.533 | 0.248 | 0.324 | 0.182 |
| SS-k-means on SimGCD-DINOv2 features | 0.519 | 0.683 | 0.369 | 0.581 | 0.249 |
| SS-k-means on HypCD-DINOv2 features | 0.488 | 0.672 | 0.319 | 0.541 | 0.203 |
| SS-k-means on frozen concat | 0.513 | 0.576 | **0.456** | 0.548 | 0.362 |
| SS-k-means on trained fusion† | 0.626 | 0.735 | 0.525 | 0.637 | 0.417 |

SS-k-means: semi-supervised k-means (Vaze et al., 2022), best of 3 restarts. †The fusion was trained with the
labels of all images of 1,057 known species, more supervision than the GCD protocol allows; reference only.
On `novel_unseen_branch` every GCD model's NMI (0.675–0.729) is below the all-singletons value (0.758).

### 3.2 GCD features in the HNSD pipeline (`s2_router_clusters.json`, `r1_router_clusters.json`, `*__a2_*`, `*__a4_*`)

| representation | router AUROC (cluster) | construction: orders found / 21 (order hold-outs) | order ARI, all unseen | genus pair-F1, all unseen | placement parent accuracy |
|---|---|---|---|---|---|
| trained fusion | 0.959 | 21 | 0.563 | 0.343 | 0.367 |
| SimGCD, DINO | 0.863 | 1 | 0.000 | 0.205 | 0.261 |
| SimGCD, DINOv2 | 0.852 | 1 | 0.000 | 0.204 | 0.294 |
| HypCD, DINOv2 | 0.854 | 15 | **0.650** | 0.179 | 0.278 |

SimGCD training flattens the hierarchy with either backbone; hyperbolic training (HypCD) keeps the order level but
loses genera. HypCD's clipping places every embedding at the same radius of the Poincaré ball (1.1455), so the
radius carries no rank information.

---

## 4. Removing the oracles (`results/round2/`)

### 4.1 Estimating the number of classes (`r2_estimate_k.json`)

GCD's method: k-means on all images, scored by clustering accuracy on the labelled images, Brent search over k.

| representation | estimated k | implied novel species (true 640) |
|---|---|---|
| trained fusion | 2,179 | 1,002 |
| concat | 2,513 | 1,336 |

The objective is flat between roughly 1,850 and 2,550, so k is weakly identified and overestimated.

### 4.2 End-to-end construction on discovered clusters (`r2_end_to_end.json`)

Semi-supervised k-means routes each unlabelled image to a known species or a new unit; quantile radii build the
hierarchy over the new units. Scored per image on `novel_unseen_branch` (not comparable with the per-species
numbers of Section 1.2).

| representation | setting | order ARI, order hold-outs | order ARI, all unseen | genus ARI, all unseen | unseen images routed to new units |
|---|---|---|---|---|---|
| fusion | true species units | 0.856 | 0.512 | 0.608 | 100% |
| fusion | discovered, true k | 0.732 | 0.607 | 0.329 | 89% |
| fusion | discovered, estimated k | 0.752 | 0.491 | 0.327 | 94% |
| concat | true species units | 0.892 | 0.674 | 0.582 | 100% |
| concat | discovered, true k | 0.674 | 0.444 | 0.302 | 84% |
| concat | discovered, estimated k | 0.726 | 0.495 | 0.280 | 92% |

Only 27–39% of `novel_known_branch` images reach new units (fusion); the rest are absorbed by a known relative.

### 4.3 Rank prior estimated from unlabelled novel species (`r3_prior_v2.json`, `r3_prior_bioscan.json`)

The rank prior is estimated without test labels by EM (Saerens et al., 2002) on the placer's posteriors, or by
BBSE (Lipton et al., 2018) from out-of-fold confusion on the pseudo-training set. Parent accuracy [95% CI]:

| benchmark / representation | prior from known taxonomy | BBSE | EM | evaluation-set prior (oracle) | constant family |
|---|---|---|---|---|---|
| iNat v2 / fusion | 0.367 | 0.378 | **0.383** | 0.400 | 0.267 |
| iNat v2 / concat | 0.378 | 0.394 | **0.394** | 0.411 | 0.289 |
| BIOSCAN / concat | 0.260 | 0.310 | **0.350** [0.33, 0.37] | 0.345 | 0.310 |
| BIOSCAN / BIOSCAN fusion | 0.333 | 0.306 | **0.364** [0.35, 0.38] | 0.359 | 0.306 |
| BIOSCAN / iNat fusion | 0.261 | 0.289 | **0.328** [0.31, 0.35] | 0.325 | 0.289 |

EM matches the oracle prior on BIOSCAN and closes most of the gap on iNaturalist. BBSE collapses to "all family"
on BIOSCAN (ill-conditioned confusion matrix). EM underestimates order-level novelty on iNaturalist (0% vs. 14%).

# Reproducing every table

## 1. Stage notebooks (CUDA GPU)

| notebook | produces | time on one RTX A6000 |
|---|---|---|
| `03_build_inat_benchmark.ipynb` | iNat v2 split, frozen features, fusion features, five hold-out fusions | several hours (feature extraction) |
| `03b_build_bioscan_benchmark.ipynb` | BIOSCAN-5M split, frozen features, BIOSCAN fusion, iNat-fusion transfer features | several hours (download + extraction) |
| `04_simgcd_dino.ipynb` | SimGCD (DINO ViT-B/16), 100 epochs | about 13 h |
| `05a_simgcd_dinov2.ipynb` | SimGCD (DINOv2 ViT-B/14 reg4), 100 epochs | about 14 h |
| `05b_hypcd_dinov2.ipynb` | HypCD (DINOv2 ViT-B/14 reg4), 100 epochs, FP32 | about 30–40 h |

Each notebook prints its projected run time in its smoke-test step, resumes from the last finished epoch after an
interruption, and lists the files to copy at the end.

## 2. Configure paths

```bash
cp paths.example.json paths.local.json   # then edit the folders
python hnsd_paths.py                     # prints ok / missing for each folder
```

## 3. Analyses

Run from the repository root. Each command writes JSON to `outputs/<part>/`; `python scripts/verify_results.py`
compares them with `results/<part>/`.

| table in docs/RESULTS.md | command | approximate time (Apple M-series) |
|---|---|---|
| 1.1 detection, oracle-k clustering | `python analysis/inat_v2/a1_router_clusters.py` | minutes |
| 1.2 construction, main split | `python analysis/inat_v2/a2_construction.py main` | < 1 min |
| 1.3 five hold-out pairs | `python analysis/inat_v2/a2_construction.py alt` | minutes |
| 1.4 placement | `python analysis/inat_v2/a4_placement.py` | < 1 min |
| 2.1 audit | `python analysis/bioscan/b0_audit.py` | seconds |
| 2.2 detection | `python analysis/bioscan/b1_router_clusters.py fusion_bioscan fusion_inat` and `SKIP_CLUSTER=1 python analysis/bioscan/b1_router_clusters.py concat`, then `python analysis/bioscan/b1b_router_by_order.py` | ~15 min |
| 2.3 clustering | `python analysis/bioscan/b1c_cluster_torch.py fusion_bioscan concat fusion_inat` (GPU k-means) | ~25 min |
| 2.4 placement | `python analysis/bioscan/b2_placement.py fusion_bioscan fusion_inat concat` | ~15 min |
| 3.1 GCD protocol (SimGCD-DINO, fusion, concat) | `python analysis/gcd_baselines/s1_gcd_protocol.py simgcd fusion concat` | ~15 min |
| 3.1 GCD protocol (DINOv2 runs) | `python analysis/round2/r1_gcd_protocol.py simgcd_dinov2 hypcd_dinov2` | ~5 min |
| 3.2 router and clustering | `python analysis/gcd_baselines/s2_router_clusters.py`, `python analysis/round2/r1_router_clusters.py simgcd_dinov2 hypcd_dinov2` | ~10 min |
| 3.2 construction and placement on GCD features | `scripts/run_analyses.sh gcd_baselines` and the `r1_run_v2_on.py` lines of `scripts/run_analyses.sh round2` | minutes |
| 4.1 estimated k | `python analysis/round2/r2_estimate_k.py fusion concat` | ~1 h |
| 4.2 end-to-end construction | `python analysis/round2/r2_end_to_end.py fusion concat` (needs 4.1) | ~25 min |
| 4.3 estimated rank prior | `python analysis/round2/r3_prior.py v2 fusion concat` and `python analysis/round2/r3_prior.py bioscan concat fusion_bioscan fusion_inat` | ~10 min |

Or everything at once: `scripts/run_analyses.sh all`.

## Determinism

CPU analyses (random forests, hierarchical clustering, scikit-learn k-means) are seeded and reproduce the reference
JSON exactly. The GPU k-means scripts (`b1c`, `s1`, `r1_gcd_protocol`, `r2_*`) are seeded but can differ in the
last digits on other hardware; compare them with a looser tolerance, e.g.
`python scripts/verify_results.py --rtol 1e-2`.

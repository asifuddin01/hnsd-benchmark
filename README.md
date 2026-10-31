# HNSD benchmark: taxonomic placement and sub-taxonomy construction for unseen species

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)](#installation)

Code, benchmark splits and reference results for **hierarchical novel species discovery (HNSD)**: given images of
species that are absent from a reference set, decide

1. **detection** – whether an image cluster belongs to a known species or to a novel one;
2. **placement** – for a novel species with known relatives, *where* it attaches in the known taxonomy
   (the rank: genus, family or order, and the parent taxon at that rank);
3. **construction** – for novel species with no known relative below class level, how they organise into a new
   sub-taxonomy (orders, families, genera).

Two benchmarks are provided, both with ground truth **computed** from each novel species' deepest ancestor in the
known set: an iNaturalist 2021 benchmark with controlled hold-outs, and a BIOSCAN-5M benchmark with real
undescribed insect species photographed in the lab. Generalized category discovery (GCD) baselines are included
(SimGCD and HypCD, with DINO and DINOv2 backbones).

---

## Contents

- [Repository layout](#repository-layout)
- [Benchmarks](#benchmarks)
- [Main results](#main-results)
- [Installation](#installation)
- [Data](#data)
- [Reproducing the results](#reproducing-the-results)
- [Third-party code, models and data](#third-party-code-models-and-data)
- [Citation](#citation)

## Repository layout

```
hnsd-benchmark/
├── data/                     benchmark definitions (splits only; no images, no features)
│   ├── inat_v2/              species pool, split with ground truth, image list, hold-out pairs, fusion folds
│   ├── bioscan/              BIOSCAN-5M split with ground truth, image list, fusion folds
│   └── gcd/                  labelled / unlabelled image split shared by all GCD runs
├── notebooks/                GPU stages: build benchmarks, extract features, train GCD baselines
│   ├── 03_build_inat_benchmark.ipynb
│   ├── 03b_build_bioscan_benchmark.ipynb
│   ├── 04_simgcd_dino.ipynb
│   ├── 05a_simgcd_dinov2.ipynb
│   └── 05b_hypcd_dinov2.ipynb
├── analysis/                 CPU analyses from cached features
│   ├── inat_v2/              detection, construction, placement on the iNaturalist benchmark
│   ├── bioscan/              the same on BIOSCAN-5M
│   ├── gcd_baselines/        GCD protocol and SimGCD (DINO) features through the pipeline
│   └── round2/               DINOv2 GCD baselines, estimated k, end-to-end construction, estimated rank prior
├── results/                  reference outputs (JSON) of every analysis and training log
├── scripts/                  run_analyses.sh (re-run everything), verify_results.py (compare with results/)
├── docs/                     BENCHMARK.md (data format), RESULTS.md (all tables), REPRODUCE.md (table → command)
├── hnsd_paths.py             where the cached features live (paths.example.json → paths.local.json)
├── requirements-gpu.txt      environment for the notebooks
└── requirements-analysis.txt environment for the analyses
```

The notebooks are numbered by pipeline stage (03 → 05b). Stages 01–02 were exploratory and are superseded by
stage 03, which builds the benchmark from scratch.

## Benchmarks

**iNaturalist v2** (`data/inat_v2/`): 2,017 species and 100,850 images (50 per species) from iNaturalist 2021
train-mini, drawn from 11 supercategories.

| role | species | how it is created | ground truth (deepest known ancestor) |
|---|---|---|---|
| `known` | 1,177 | reference set | — |
| `novel_known_branch` | 180 | single species withheld, a relative stays known | genus 78 · family 77 · order 25 |
| `novel_unseen_branch` | 460 | whole orders withheld (260); Fungi and Reptiles withheld in full (100 each) | class 256 · phylum 104 · none 100 |
| `pool_only` | 200 | extra Fungi/Reptiles species, not used | — |

**BIOSCAN-5M** (`data/bioscan/`): 9,445 species and 65,016 lab images (at most 10 per species). The 6,531
species of the `train` partition are known; 2,914 real undescribed species (914 from the unseen partitions,
2,000 sampled from `other_heldout`) are novel, with ground truth genus 1,302 · family 1,582 · order 30.

Details and file formats: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Main results

All tables: [docs/RESULTS.md](docs/RESULTS.md). Single training seed (42) throughout.

**GCD protocol** (iNaturalist v2; 1,177 old species with 50% of images labelled, 640 new; accuracy on the
61,425 unlabelled images)

| method | All | Old | New |
|---|---|---|---|
| SimGCD, DINO ViT-B/16 | 0.374 | 0.534 | 0.228 |
| SimGCD, DINOv2 ViT-B/14 | 0.479 | 0.668 | 0.306 |
| HypCD, DINOv2 ViT-B/14 | 0.453 | 0.637 | 0.284 |
| semi-supervised k-means on SimGCD-DINOv2 features | 0.519 | 0.683 | 0.369 |
| semi-supervised k-means on frozen SigLIP 2 + DINOv3 + DINOv2 (no training) | 0.513 | 0.576 | **0.456** |

**Construction is shaped by the training objective** (order hold-outs, 21 orders; quantile-transferred rank radii)

| representation | orders recovered | order ARI | genus pair-F1 (all unseen) |
|---|---|---|---|
| trained fusion (known species only) | 21 / 21 | 0.870 | **0.343** |
| SimGCD features (DINO or DINOv2) | 1 / 21 | 0.000 | 0.204–0.205 |
| HypCD features | 15 / 21 | 0.786 | 0.179 |

**Placement with a rank prior estimated from unlabelled data** (parent accuracy; EM prior estimation needs no
test labels)

| benchmark / representation | prior from known taxonomy | estimated prior (EM) | evaluation prior (oracle) | always "family" |
|---|---|---|---|---|
| iNaturalist v2 / trained fusion | 0.367 | 0.383 | 0.400 | 0.267 |
| BIOSCAN-5M / BIOSCAN fusion | 0.333 | **0.364** | 0.359 | 0.306 |

**Further findings**

- Genus-level NMI is not informative here: an all-singletons partition scores as high or higher, so ARI and
  pair-F1 are reported throughout.
- Without any oracle (estimated number of species, discovered clusters), construction keeps most of its
  order-level accuracy: order ARI on the order hold-outs falls from 0.856 to 0.752 (fusion, per-image scoring).
- Frozen encoders and the iNaturalist fusion do not transfer to BIOSCAN's microscope images (router AUROC at
  chance within each order); a fusion trained on BIOSCAN known species reaches 0.758.

## Installation

Two environments are used: a CUDA machine for the notebooks and any machine for the analyses.

```bash
git clone https://github.com/asifuddin01/hnsd-benchmark.git
cd hnsd-benchmark

# analyses (CPU; PyTorch is used for GPU k-means when CUDA or Apple MPS is available)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-analysis.txt

# notebooks (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```

Notebooks 04 and 05a import the official SimGCD code: clone it next to the data
(`git clone https://github.com/CVMI-Lab/SimGCD files/SimGCD`). Notebook 05b downloads the official HypCD code
at a pinned commit. DINOv3 weights are gated on Hugging Face; log in with `huggingface-cli login` first.

## Data

Images and extracted features are **not** redistributed.

- **iNaturalist 2021** train-mini images and `train_mini.json`:
  <https://github.com/visipedia/inat_comp/tree/master/2021>. Notebook 03 extracts only the selected species.
- **BIOSCAN-5M** metadata and 256-px cropped images (about 6 GB): Zenodo record
  [10.5281/zenodo.11973457](https://doi.org/10.5281/zenodo.11973457). Notebook 03b downloads and verifies them.

## Reproducing the results

1. **Stage notebooks** (CUDA GPU), in order: `03` → `03b` → `04` → `05a` → `05b`. Set `HNSD_BASE_DIR` (or edit
   `BASE_DIR` in the first cell) to your data folder. Each notebook has a smoke test, resumes after
   interruption, and ends with the list of files to copy to the analysis machine.
2. **Point the analyses at the features**: copy `paths.example.json` to `paths.local.json` and fill in the
   folders written by the notebooks; `python hnsd_paths.py` checks them.
3. **Run the analyses** and compare with the reference results:

```bash
scripts/run_analyses.sh all            # or: inat_v2 | bioscan | gcd_baselines | round2
python scripts/verify_results.py       # compares outputs/ with results/
```

Which command produces which table: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Third-party code, models and data

| component | source | licence |
|---|---|---|
| SimGCD (losses, head) | <https://github.com/CVMI-Lab/SimGCD> | MIT |
| HypCD (hyperbolic layers, losses) | <https://github.com/Visual-AI/HypCD> | MIT |
| DINO ViT-B/16, DINOv2 ViT-B/14 (reg4) | <https://github.com/facebookresearch/dino>, <https://github.com/facebookresearch/dinov2> | Apache-2.0 |
| DINOv3 ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 License |
| SigLIP 2 so400m | `google/siglip2-so400m-patch16-naflex` | Apache-2.0 |
| iNaturalist 2021 | iNat 2021 competition | per-image licences; not redistributed |
| BIOSCAN-5M | Gharaee et al., NeurIPS 2024 Datasets and Benchmarks | CC BY 3.0; not redistributed |

The benchmark split files in `data/` list species names, taxonomy and image file names only.

## Citation

If you use this benchmark or code, please cite it (see [CITATION.cff](CITATION.cff)):

```bibtex
@software{uddin_hnsd_benchmark_2026,
  author  = {Uddin, Md. Asif},
  title   = {{HNSD} benchmark: taxonomic placement and sub-taxonomy construction for unseen species},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/asifuddin01/hnsd-benchmark}
}
```

Please also cite iNaturalist 2021, BIOSCAN-5M, SimGCD and HypCD when you use the corresponding parts.

## License

Code and split files: [MIT](LICENSE). Third-party components keep their own licences (table above).

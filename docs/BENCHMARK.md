# Benchmark definition and file formats

## Tasks

| task | input | output | evaluated on |
|---|---|---|---|
| detection (router) | an image cluster | known vs. novel | known test clusters vs. novel species |
| placement | a novel species with known relatives | attachment rank (genus / family / order) and parent taxon | `novel_known_branch` |
| construction | novel species with no known relative below class | a sub-taxonomy (orders, families, genera) | `novel_unseen_branch` |

## Ground truth

For every novel species the lineage is walked from kingdom down, and the deepest rank whose **lineage prefix**
(for example `Animalia/Chordata/Aves/Passeriformes`) exists among the known species is recorded as
`true_rank`, with the taxon name as `true_parent`. Prefixes rather than bare names are compared, so homonyms in
different parents never match. Species whose deepest known ancestor is order, family or genus are
`novel_known_branch` (placement); coarser ones are `novel_unseen_branch` (construction).

## iNaturalist v2 (`data/inat_v2/`)

Source: iNaturalist 2021 train-mini (50 images per species). Seed 42.

| hold-out mechanism (`holdout`) | what is withheld | species |
|---|---|---|
| `species` | single species whose order keeps at least one known member | 180 |
| `order` | whole orders inside a supercategory (never the largest order) | 260 |
| `supercategory` | Fungi and Reptiles in full | 200 |
| `none` | known reference species | 1,177 |
| `pool_only` | extra Fungi/Reptiles species, outside the benchmark | 200 |

Five hold-out pairs for the transfer test (`alt_splits.json`): Fungi + Reptiles, Amphibians + Mollusks,
Arachnids + Ray-finned Fishes, Birds + Insects, Mammals + Plants.

| file | content |
|---|---|
| `species_pool.json` | list of 2,017 species: `category_id`, `name`, `common_name`, `supercategory`, `kingdom` … `genus`, `image_dir`, `num_images` |
| `species_split_v2.json` | the pool records plus `holdout`, `role`, `true_rank`, `true_parent` |
| `items.json` | list of `[relative image path, category_id]` (100,850 images; paths relative to the iNat 2021 root) |
| `alt_splits.json` | the five hold-out supercategory pairs |
| `fusion_main_species_folds.json` | known species used to train (1,057) and validate (120) the fusion |

## BIOSCAN-5M (`data/bioscan/`)

Source: BIOSCAN-5M (Gharaee et al., 2024), 256-px cropped images, at most 10 images per species, species with at
least 3 images. Known = all `train` species (6,531). Novel = all species of the `key_unseen` / `val_unseen` /
`test_unseen` partitions (914, images pooled) plus 2,000 species sampled from `other_heldout`. Kingdom is set to
Animalia. No chosen species has a missing taxonomy label.

| file | content |
|---|---|
| `bioscan_split.json` | species records: taxonomy (`phylum` … `genus`, `subfamily`), `holdout` (`none`, `bioscan_unseen`, `bioscan_other_heldout`), `role`, `true_rank`, `true_parent`, `num_images` |
| `items.json` | list of `[relative image path, category_id]` (65,016 images; `<split>/<chunk>/<processid>.jpg`) |
| `fusion_bioscan_species_folds.json` | known species used to train (6,411) and validate (120) the BIOSCAN fusion |

## GCD split (`data/gcd/simgcd_split.json`)

Shared by every GCD run (notebooks 04, 05a, 05b) and by the semi-supervised k-means analyses.

| key | content |
|---|---|
| `old_species` / `new_species` | 1,177 known species / 640 novel species (both novel roles); `pool_only` excluded |
| `labelled` | 29,425 `[path, category_id]` pairs: 50% of each known species' images (seed 42) |
| `unlabelled` | 61,425 pairs: the other known images and all novel images |

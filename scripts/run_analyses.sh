#!/usr/bin/env bash
# Re-run the analyses from cached features. Results go to outputs/<part>/ (reference copies: results/<part>/).
#
#   scripts/run_analyses.sh [inat_v2|bioscan|gcd_baselines|round2|all]
#
# Set PYTHON to choose the interpreter (default: python). Paths to the cached features come from hnsd_paths.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
run() { echo "+ $*"; "$@"; }
feat() { "$PY" -c "import hnsd_paths as p; print(p.$1 / '$2')"; }

inat_v2() {
  run "$PY" analysis/inat_v2/a1_router_clusters.py              # router AUROC, oracle-k NMI
  run "$PY" analysis/inat_v2/a2_construction.py main            # construction on the main split
  run "$PY" analysis/inat_v2/a2_construction.py alt             # five hold-out pairs (transfer test)
  run "$PY" analysis/inat_v2/a4_placement.py                    # placement of 180 novel species
}

bioscan() {
  run "$PY" analysis/bioscan/b0_audit.py
  run "$PY" analysis/bioscan/b1_router_clusters.py fusion_bioscan fusion_inat
  run env SKIP_CLUSTER=1 "$PY" analysis/bioscan/b1_router_clusters.py concat   # router only; clustering in b1c
  run "$PY" analysis/bioscan/b1b_router_by_order.py
  run "$PY" analysis/bioscan/b1c_cluster_torch.py fusion_bioscan concat fusion_inat
  run "$PY" analysis/bioscan/b2_placement.py fusion_bioscan fusion_inat concat
}

gcd_baselines() {
  local F; F="$(feat RUN_SIMGCD_DINO features_simgcd_v2.npz)"
  run "$PY" analysis/gcd_baselines/s1_gcd_protocol.py simgcd fusion concat
  run "$PY" analysis/gcd_baselines/s2_router_clusters.py
  run "$PY" analysis/gcd_baselines/run_v2_on.py a4_placement "$F" simgcd
  run "$PY" analysis/gcd_baselines/run_v2_on.py a2_construction "$F" simgcd main
}

round2() {
  run "$PY" analysis/round2/r1_gcd_protocol.py simgcd_dinov2 hypcd_dinov2
  run "$PY" analysis/round2/r1_router_clusters.py simgcd_dinov2 hypcd_dinov2
  for R in simgcd_dinov2 hypcd_dinov2; do
    local KEY F
    KEY="RUN_$(echo "$R" | tr '[:lower:]' '[:upper:]')"
    F="$(feat "$KEY" "features_${R}_v2.npz")"
    run "$PY" analysis/round2/r1_run_v2_on.py a4_placement "$F" "$R"
    run "$PY" analysis/round2/r1_run_v2_on.py a2_construction "$F" "$R" main
  done
  run "$PY" analysis/round2/r3_prior.py v2 fusion concat
  run "$PY" analysis/round2/r2_estimate_k.py fusion concat
  run "$PY" analysis/round2/r2_end_to_end.py fusion concat
  run "$PY" analysis/round2/r3_prior.py bioscan concat fusion_bioscan fusion_inat
}

case "${1:-all}" in
  inat_v2) inat_v2 ;;
  bioscan) bioscan ;;
  gcd_baselines) gcd_baselines ;;
  round2) round2 ;;
  all) inat_v2; bioscan; gcd_baselines; round2 ;;
  *) echo "unknown part: $1" >&2; exit 2 ;;
esac

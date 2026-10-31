"""Compare re-run analysis outputs (outputs/<part>/*.json) with the reference results (results/<part>/*.json).

    python scripts/verify_results.py [--rtol 1e-6] [--atol 1e-9] [part ...]

Timing fields are ignored. GPU k-means on another device can differ slightly; loosen --rtol for those files.
Exit code 0 when every compared file matches, 1 otherwise.
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from hnsd_paths import OUTPUTS  # noqa: E402

IGNORE = ("seconds", "minutes", "kmeans_seconds")


def diff(a, b, path, rtol, atol, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if any(s in str(k) for s in IGNORE):
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'reference' if k in a else 'output'}")
            else:
                diff(a[k], b[k], f"{path}/{k}", rtol, atol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", rtol, atol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=rtol, abs_tol=atol) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    args = ap.parse_args()
    parts = args.parts or sorted(p.name for p in (REPO / "results").iterdir() if p.is_dir() and p.name != "training")
    n_ok = n_bad = n_skip = 0
    for part in parts:
        for ref in sorted((REPO / "results" / part).glob("*.json")):
            new = OUTPUTS / part / ref.name
            if not new.exists():
                n_skip += 1
                continue
            problems = []
            diff(json.loads(ref.read_text()), json.loads(new.read_text()), "", args.rtol, args.atol, problems)
            status = "OK  " if not problems else "DIFF"
            n_ok += not problems
            n_bad += bool(problems)
            print(f"{status} {part}/{ref.name}" + (f"  ({len(problems)} differences)" if problems else ""))
            for p in problems[:5]:
                print("       ", p)
    print(f"\n{n_ok} match, {n_bad} differ, {n_skip} not re-run")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())

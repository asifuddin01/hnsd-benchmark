"""B0. Audit of the BIOSCAN split before any result is trusted.
(1) Is the ground truth censored by missing labels?  (2) Images per species (sets the trivial NMI).
(3) Were the router's 'known test' images in the BIOSCAN fusion's training data?"""
from common_b import *
from collections import Counter

S, BY = load_split()
items = json.load(open(SPLITS_BIOSCAN / "items.json"))
n_img = Counter(c for _, c in items)
nov = [s for s in S if s["role"] != "known"]
known = [s for s in S if s["role"] == "known"]
out = {}

print("== (1) ground truth vs missing labels ==")
tab = {}
for src in ["bioscan_unseen", "bioscan_other_heldout"]:
    sub = [s for s in nov if s["holdout"] == src]
    for r in ["genus", "family", "order"]:
        g = [s for s in sub if s["true_rank"] == r]
        c = sum(censored(s) for s in g)
        tab[f"{src}|{r}"] = {"n": len(g), "censored_by_missing_label": c}
        print(f"  {src:<24} truth={r:<7} n={len(g):>5}  censored(next label missing)={c:>5}")
miss = {r: sum(str(s[r]).startswith(MISSING) for s in known) for r in ["order", "family", "genus"]}
print("  known species with missing label:", miss)
missn = {r: sum(str(s[r]).startswith(MISSING) for s in nov) for r in ["order", "family", "genus"]}
print("  novel species with missing label:", missn)
out["truth_vs_missing"] = tab; out["known_missing"] = miss; out["novel_missing"] = missn
for src in ["bioscan_unseen", "bioscan_other_heldout"]:
    print(f"  example names {src}:", [s["name"] for s in nov if s["holdout"] == src][:8])
print("  example known names:", [s["name"] for s in known][:8])

print("\n== (2) images per species ==")
ipc = {}
for role in ["known", "novel_known_branch"]:
    v = np.array([n_img[s["category_id"]] for s in S if s["role"] == role])
    ipc[role] = {"species": len(v), "images": int(v.sum()), "mean": float(v.mean()),
                 "hist": {int(k): int(x) for k, x in sorted(Counter(v.tolist()).items())}}
    print(f"  {role:<20} species {len(v):>5} images {v.sum():>6} mean {v.mean():.2f}  hist {ipc[role]['hist']}")
out["images_per_species"] = ipc

print("\n== (3) router known-test images vs fusion training ==")
F = fusion_folds(); tr, va = set(F["train"]), set(F["val"])
kn = {s["category_id"] for s in known}
print(f"  known {len(kn)} | fusion train {len(tr & kn)} | fusion val {len(va & kn)} | novel in train/val {len((tr|va) - kn)}")
print("  The notebook's router test used 30% of images of ALL known species; the fusion trained on all images")
print(f"  of {len(tr & kn)} of them, so {len(tr & kn)/len(kn):.1%} of 'known test' clusters were seen in training.")
out["fusion_overlap"] = {"known": len(kn), "train": len(tr & kn), "val": len(va & kn)}
print("\n  orders of novel species:", Counter(s["order"] for s in nov).most_common(8))
print("  classes of known species:", Counter(s["class"] for s in known).most_common(5))
save("b0_audit.json", out)

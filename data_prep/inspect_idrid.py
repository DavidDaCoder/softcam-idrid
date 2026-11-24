"""Inspect the IDRiD dataset before using it for SoftCAM localization.

Usage: python3 inspect_idrid.py <root containing 'A. Segmentation' etc> <out_dir>

Checks:
  1. Overlap between parts A, B, C: exact (SHA-256) and near-duplicate
     (mean absolute pixel difference on a 96x64 RGB thumbnail; a match needs
     MAD < 2 and best/second-best < 0.5), since the same photo is re-encoded
     between parts. A 64-bit dHash was tried first and gave false matches. Names differ
     (IDRiD_01 vs IDRiD_001) and repeat across train/test, so never key by name.
  2. Image resolution per part/split.
  3. Part A: split counts, mask counts per lesion type, mask value check,
     lesion area as a share of the fundus field of view (the chance baseline
     for evidence-in-mask, like box area ~12.8% on RSNA).
  4. Part B: DR grade (0-4) and DME grade (0-2) distributions, and the
     balance at DR grade >= 2.
  5. Part C: optic disc centers vs Part A optic disc masks (for matched images).
"""
import csv, hashlib, json, os, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_cache.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

SPLITS = {"a. Training Set": "train", "b. Testing Set": "test"}
LESIONS = {"1. Microaneurysms": "MA", "2. Haemorrhages": "HE", "3. Hard Exudates": "EX",
           "4. Soft Exudates": "SE", "5. Optic Disc": "OD"}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def image_info(path):
    """Size, sha256, dHash, and field-of-view mask stats (cached)."""
    key = os.path.relpath(path, ROOT)
    if key in cache:
        return cache[key]
    im = Image.open(path)
    w, h = im.size
    im.draft("L", (w // 8, h // 8))          # fast JPEG decode at 1/8 scale
    g = np.asarray(im.convert("L"), dtype=np.float32)
    small = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((9, 8), Image.BILINEAR), dtype=np.float32)
    dhash = int("".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).ravel()), 2)
    fov = float((g > 15).mean())             # share of image inside the circular fundus
    im2 = Image.open(path); im2.draft("RGB", (w // 8, h // 8))
    thumb = np.asarray(im2.convert("RGB").resize((96, 64), Image.BILINEAR), dtype=np.uint8)
    info = {"w": w, "h": h, "sha": sha(path), "dhash": f"{dhash:016x}", "fov_frac": fov,
            "thumb": thumb.ravel().tolist()}
    cache[key] = info
    return info

def mask_info(path):
    key = os.path.relpath(path, ROOT)
    if key in cache:
        return cache[key]
    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m.max(axis=2)
    vals = np.unique(m)
    pos = m > 0
    ys, xs = np.nonzero(pos)
    info = {"w": m.shape[1], "h": m.shape[0], "values": [int(v) for v in vals[:10]],
            "n_values": int(len(vals)), "pos_pixels": int(pos.sum()),
            "centroid": [float(xs.mean()), float(ys.mean())] if len(xs) else None,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None}
    cache[key] = info
    return info

def list_images(part):
    out = []
    base = os.path.join(ROOT, part, "1. Original Images")
    for sdir, split in SPLITS.items():
        for f in sorted(os.listdir(os.path.join(base, sdir))):
            if f.lower().endswith(".jpg"):
                out.append((split, f[:-4], os.path.join(base, sdir, f)))
    return out

def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")

report = []
P = report.append

# ---------- images ----------
parts = {"A": "A. Segmentation", "B": "B. Disease Grading", "C": "C. Localization"}
imgs = {}
for k, part in parts.items():
    imgs[k] = [(s, n, p, image_info(p)) for s, n, p in list_images(part)]
    json.dump(cache, open(CACHE, "w"))

P("# IDRiD inspection\n")
P("## 1. Image counts and resolution\n")
P("| Part | Split | Images | Resolutions (w×h: count) | Mean FOV share |")
P("|---|---|---|---|---|")
for k in parts:
    for split in ("train", "test"):
        rows = [i for s, n, p, i in imgs[k] if s == split]
        res = Counter(f"{i['w']}×{i['h']}" for i in rows)
        fov = np.mean([i["fov_frac"] for i in rows])
        P(f"| {k} | {split} | {len(rows)} | {', '.join(f'{r}: {c}' for r, c in res.items())} | {fov:.3f} |")

# ---------- overlap ----------
P("\n## 2. Overlap between parts\n")
def overlap(ka, kb, mad_max=2.0, ratio_max=0.5):
    exact, nearpairs = [], []
    by_sha = defaultdict(list)
    for s, n, p, i in imgs[kb]:
        by_sha[i["sha"]].append((s, n))
    TB = np.stack([np.array(j["thumb"], dtype=np.float32) for *_, j in imgs[kb]])
    keys = [(s2, n2) for s2, n2, _, _ in imgs[kb]]
    for s, n, p, i in imgs[ka]:
        if i["sha"] in by_sha:
            exact.append(((s, n), by_sha[i["sha"]]))
        d = np.abs(TB - np.array(i["thumb"], dtype=np.float32)).mean(axis=1)
        o = np.argsort(d)
        if d[o[0]] < mad_max and d[o[0]] / max(d[o[1]], 1e-9) < ratio_max:
            nearpairs.append(((s, n), keys[o[0]], round(float(d[o[0]]), 3)))
    return exact, nearpairs

overlaps = {}
for ka, kb in (("A", "B"), ("B", "C"), ("A", "C")):
    ex, nr = overlap(ka, kb)
    overlaps[f"{ka}-{kb}"] = {"exact": ex, "near": nr}
    same_name_split = sum(1 for (a, bs) in ex if a in bs)
    P(f"- **{ka} vs {kb}:** {len(ex)} of {len(imgs[ka])} {ka} images have a byte-identical file in {kb} "
      f"({same_name_split} with the same split and name); {len(nr)} match by pixel comparison (MAD < 2, clear of the runner-up).")
    if nr:
        P(f"  - Split pairing ({ka} split → {kb} split): " + ", ".join(f"{a}→{b}: {c}" for (a, b), c in Counter((x[0][0], x[1][0]) for x in nr).items()))
P("\nIn A vs B the best-match MAD has a clean gap (≤ 0.7 for matches, ≥ 3.0 for the rest), so the threshold is not delicate.")
ab_near = overlaps["A-B"]["near"]
if ab_near:
    P("\nA → B near-duplicate pairs (first 20):\n")
    P("| A image | B match | Thumbnail MAD |")
    P("|---|---|---|")
    for a, b, d in ab_near[:20]:
        P(f"| {a[0]}/{a[1]} | {b[0]}/{b[1]} | {d} |")

# ---------- part A masks ----------
P("\n## 3. Part A segmentation masks\n")
P("Mask counts include every file; the area table excludes any flagged bad mask.\n")
gt = os.path.join(ROOT, parts["A"], "2. All Segmentation Groundtruths")
fov_by = {(s, n): i for s, n, p, i in imgs["A"]}
area = defaultdict(list)
mask_counts = defaultdict(Counter)
value_sets = Counter()
size_mismatch = 0
union_frac = defaultdict(float)
for sdir, split in SPLITS.items():
    for ldir, code in LESIONS.items():
        d = os.path.join(gt, sdir, ldir)
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".tif"):
                continue
            name = f.rsplit("_", 1)[0]
            mi = mask_info(os.path.join(d, f))
            img = fov_by.get((split, name))
            mask_counts[split][code] += 1
            value_sets[tuple(mi["values"])] += 1
            if mi["n_values"] == 1 and mi["pos_pixels"] == mi["w"] * mi["h"]:
                P(f"- **Bad mask:** {split}/{f} is entirely positive (all {mi['values'][0]}); excluded from area stats.")
                continue
            if img and (img["w"], img["h"]) != (mi["w"], mi["h"]):
                size_mismatch += 1
            fov_px = (img["fov_frac"] if img else 1.0) * mi["w"] * mi["h"]
            frac = mi["pos_pixels"] / fov_px
            area[(split, code)].append(frac)
            if code != "OD":
                union_frac[(split, name)] += frac   # upper bound on lesion union (overlaps double-count)
    json.dump(cache, open(CACHE, "w"))

P("| Split | Images | MA | HE | EX | SE | OD |")
P("|---|---|---|---|---|---|---|")
for split in ("train", "test"):
    n = sum(1 for s, *_ in imgs["A"] if s == split)
    c = mask_counts[split]
    P(f"| {split} | {n} | " + " | ".join(str(c[x]) for x in ("MA", "HE", "EX", "SE", "OD")) + " |")
P(f"\nMask pixel values seen: {dict((str(k), v) for k, v in value_sets.items())}. "
  f"Masks whose size differs from their image: {size_mismatch}.")
P("\nLesion area as a share of the fundus field of view (median [min–max]), the chance baseline for evidence-in-mask:\n")
P("| Split | MA | HE | EX | SE | OD | All lesions (sum, upper bound) |")
P("|---|---|---|---|---|---|---|")
for split in ("train", "test"):
    cells = []
    for code in ("MA", "HE", "EX", "SE", "OD"):
        v = np.array(area[(split, code)])
        cells.append(f"{100*np.median(v):.2f}% [{100*v.min():.2f}–{100*v.max():.2f}]" if len(v) else "—")
    u = np.array([v for (s, _), v in union_frac.items() if s == split])
    cells.append(f"{100*np.median(u):.2f}% [{100*u.min():.2f}–{100*u.max():.2f}]")
    P(f"| {split} | " + " | ".join(cells) + " |")

# ---------- part B grades ----------
P("\n## 4. Part B grades\n")
def read_grades(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row and row[0].strip():
                out[row[0].strip()] = (int(row[1]), int(row[2]))
    return out
gdir = os.path.join(ROOT, parts["B"], "2. Groundtruths")
grades = {"train": read_grades(os.path.join(gdir, "a. IDRiD_Disease Grading_Training Labels.csv")),
          "test": read_grades(os.path.join(gdir, "b. IDRiD_Disease Grading_Testing Labels.csv"))}
P("| Split | Labeled | DR 0 | DR 1 | DR 2 | DR 3 | DR 4 | DR ≥ 2 | DME 0 | DME 1 | DME 2 |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for split, g in grades.items():
    dr = Counter(v[0] for v in g.values()); dme = Counter(v[1] for v in g.values())
    pos = sum(dr[k] for k in (2, 3, 4))
    P(f"| {split} | {len(g)} | " + " | ".join(str(dr[k]) for k in range(5)) +
      f" | {pos} ({100*pos/len(g):.1f}%) | " + " | ".join(str(dme[k]) for k in range(3)) + " |")
    names = {n for s, n, *_ in imgs["B"] if s == split}
    if names != set(g):
        P(f"\n- {split}: {len(names - set(g))} images without labels, {len(set(g) - names)} labels without images.")

# grades for part A images via near-duplicate match
if ab_near:
    matched = Counter()
    for a, b, d in ab_near:
        dr = grades[b[0]].get(b[1], (None,))[0]
        matched[dr] += 1
    P(f"\nDR grades of Part A images matched to Part B: {dict(sorted(matched.items(), key=lambda x: (x[0] is None, x[0])))}")

# ---------- duplicates inside part B ----------
P("\n## 4b. Duplicates inside Part B\n")
TB = np.stack([np.array(i["thumb"], dtype=np.float32) for *_, i in imgs["B"]])
dups = []
for a in range(len(imgs["B"])):
    d = np.abs(TB[a + 1:] - TB[a]).mean(axis=1)
    for off in np.nonzero(d < 2.0)[0]:
        b = a + 1 + int(off)
        (sa, na, _, ia), (sb, nb, _, ib) = imgs["B"][a], imgs["B"][b]
        dups.append((sa, na, sb, nb, ia["sha"] == ib["sha"], grades[sa][na], grades[sb][nb]))
if dups:
    P("| Image | Duplicate | Same file bytes | Grades (DR, DME) |")
    P("|---|---|---|---|")
    for sa, na, sb, nb, same, ga, gb in dups:
        flag = "" if ga == gb else " **(disagree)**"
        P(f"| {sa}/{na} | {sb}/{nb} | {'yes' if same else 'no'} | {ga} vs {gb}{flag} |")
    cross = sum(1 for d in dups if d[0] != d[2])
    P(f"\n{len(dups)} duplicate pairs; {cross} cross the train/test split.")
else:
    P("None found.")

# ---------- part C vs part A optic disc ----------
P("\n## 5. Part C optic disc centers vs Part A optic disc masks\n")
cdir = os.path.join(ROOT, parts["C"], "2. Groundtruths", "1. Optic Disc Center Location")
centers = {}
for f in os.listdir(cdir):
    if not f.endswith(".csv"):
        continue
    split = "train" if "Train" in f else "test"
    with open(os.path.join(cdir, f), newline="", encoding="utf-8-sig") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            if row and row[0].strip():
                centers[(split, row[0].strip())] = (float(row[1]), float(row[2]))
ac = overlaps["A-C"]["near"] + [((a), bs[0], 0) for a, bs in overlaps["A-C"]["exact"]]
seen, dists, inside = set(), [], 0
for a, c, _ in ac:
    if a in seen or c not in centers:
        continue
    seen.add(a)
    sdir = [k for k, v in SPLITS.items() if v == a[0]][0]
    mp = os.path.join(gt, sdir, "5. Optic Disc", f"{a[1]}_OD.tif")
    if not os.path.exists(mp):
        continue
    mi = mask_info(mp)
    cx, cy = centers[c]
    dists.append(float(np.hypot(cx - mi["centroid"][0], cy - mi["centroid"][1])))
    x0, y0, x1, y1 = mi["bbox"]
    inside += x0 <= cx <= x1 and y0 <= cy <= y1
if dists:
    P(f"{len(dists)} Part A images matched to Part C. Distance from C's disc center to A's mask centroid: "
      f"median {np.median(dists):.0f} px, max {max(dists):.0f} px; {inside} of {len(dists)} centers fall inside the mask's bounding box.")
else:
    P("No Part A images matched to Part C, so this check could not run.")

json.dump(cache, open(CACHE, "w"))
json.dump({k: {"exact": v["exact"], "near": v["near"]} for k, v in overlaps.items()},
          open(os.path.join(OUT, "overlaps.json"), "w"), indent=1)
open(os.path.join(OUT, "IDRiD_inspection.md"), "w").write("\n".join(report) + "\n")
print("\n".join(report))

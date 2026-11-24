"""Build IDRiD splits and 512-px images/masks for the SoftCAM Fundus pipeline.

Usage:
  python3 prepare_idrid.py <idrid_root> <out_dir> [--size 512] [--val-frac 0.15] [--seed 0] [--workers N]

<idrid_root> holds 'A. Segmentation', 'B. Disease Grading' (Part C is not needed:
it is byte-identical to B). Re-running skips images already written.

Output (<out_dir>):
  images/<name>.png         fundus cropped to its field of view, padded to square
                            (black), resized to --size. Names are unique:
                            B_train_IDRiD_001, B_test_IDRiD_001, A_train_IDRiD_01 ...
  masks/<name>_<LES>.png    Part A masks, same crop/pad, resized with box
                            averaging: value = share of the cell that is lesion
                            (0-255). Use > 0 for "any lesion", or threshold.
                            LES in MA, HE, EX, SE, OD.
  csv/train.csv, val.csv, test.csv   SoftCAM Fundus format (filename, level, onset1, onset2, ...)
  csv/cv_folds.csv          train + val pooled with a 5-fold column, for CV
  csv/localization.csv      all 81 Part A images with mask filenames and matched B grade
  csv/excluded.csv          every B image left out of training, with the reason
  csv/crop_params.csv       crop box and scale per image (map back to full resolution)
  paths_snippet.yaml        values for the 'Fundus' entry in configs/paths.yaml
  SPLIT_SUMMARY.md          counts and class balance per split

Split rules (see '20 Next Phase' -> Inspection results):
  - Images are matched across parts by 96x64 thumbnail mean absolute pixel
    difference (match: MAD < 2 and < 0.5x the runner-up). Names are not reliable.
  - B-train images that are also in Part A are removed from train/val, so no
    localization image is seen in training.
  - Byte-identical duplicates inside B: if grades agree, keep the first; if they
    disagree, drop both from train/val. A train copy of a test image is dropped;
    the test set itself is left as published (conflicts flagged in label_conflict).
  - Val = stratified (by DR grade) fraction of the remaining B-train images.
  - The all-white mask A test IDRiD_81_EX is marked broken and not written.
"""
import argparse, csv, hashlib, os, random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SPLITS = {"a. Training Set": "train", "b. Testing Set": "test"}
LESIONS = {"1. Microaneurysms": "MA", "2. Haemorrhages": "HE", "3. Hard Exudates": "EX",
           "4. Soft Exudates": "SE", "5. Optic Disc": "OD"}
BROKEN_MASKS = {("test", "IDRiD_81", "EX")}
FOV_THRESH = 15


# ---------------------------------------------------------------- image helpers
def scan(path):
    """sha256, 96x64 thumbnail, and field-of-view bbox (full-res coords)."""
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    im = Image.open(path)
    W, H = im.size
    im.draft("RGB", (W // 8, H // 8))
    small = im.convert("RGB")
    sw, sh = small.size
    thumb = np.asarray(small.resize((96, 64), Image.BILINEAR), dtype=np.float32).ravel()
    g = np.asarray(small.convert("L"))
    ys, xs = np.nonzero(g > FOV_THRESH)
    fx, fy = W / sw, H / sh
    box = (int(xs.min() * fx), int(ys.min() * fy),
           min(W, int((xs.max() + 1) * fx)), min(H, int((ys.max() + 1) * fy)))
    return h, thumb, (W, H), box


def square_box(box, W, H):
    """Square crop box around the FOV bbox (may extend past the image: padded black)."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (int(round(cx - side / 2)), int(round(cy - side / 2)),
            int(round(cx - side / 2)) + side, int(round(cy - side / 2)) + side)


def crop_pad(im, sq):
    # PIL crop pads out-of-bounds areas with 0 (black)
    return im.crop(sq)


def process_image(job):
    src, dst, sq, size = job
    if os.path.exists(dst):
        return dst
    im = Image.open(src).convert("RGB")
    crop_pad(im, sq).resize((size, size), Image.LANCZOS).save(dst)
    return dst


def process_mask(job):
    src, dst, sq, size = job
    if os.path.exists(dst):
        return dst
    m = np.asarray(Image.open(src))
    if m.ndim == 3:
        m = m.max(axis=2)
    m = Image.fromarray(((m > 0) * 255).astype(np.uint8))
    crop_pad(m, sq).resize((size, size), Image.BOX).save(dst)   # box = area coverage
    return dst


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--splits-only", action="store_true", help="write CSVs, skip image/mask export")
    a = ap.parse_args()
    for d in ("images", "masks", "csv"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    # --- list images
    recs = []   # dicts: part, split, id, path
    for part, pdir in (("A", "A. Segmentation"), ("B", "B. Disease Grading")):
        base = os.path.join(a.root, pdir, "1. Original Images")
        for sdir, split in SPLITS.items():
            for f in sorted(os.listdir(os.path.join(base, sdir))):
                if f.lower().endswith(".jpg"):
                    recs.append({"part": part, "split": split, "id": f[:-4],
                                 "path": os.path.join(base, sdir, f),
                                 "name": f"{part}_{split}_{f[:-4]}"})
    with ProcessPoolExecutor(a.workers) as ex:
        for r, (h, t, wh, box) in zip(recs, ex.map(scan, [r["path"] for r in recs], chunksize=8)):
            r.update(sha=h, thumb=t, W=wh[0], H=wh[1], fov=box, sq=square_box(box, *wh))
    A = [r for r in recs if r["part"] == "A"]
    B = [r for r in recs if r["part"] == "B"]

    # --- grades
    gdir = os.path.join(a.root, "B. Disease Grading", "2. Groundtruths")
    grades = {}
    for split, f in (("train", "a. IDRiD_Disease Grading_Training Labels.csv"),
                     ("test", "b. IDRiD_Disease Grading_Testing Labels.csv")):
        with open(os.path.join(gdir, f), newline="", encoding="utf-8-sig") as fh:
            rd = csv.reader(fh); next(rd)
            for row in rd:
                if row and row[0].strip():
                    grades[(split, row[0].strip())] = (int(row[1]), int(row[2]))
    for r in B:
        r["level"], r["dme"] = grades[(r["split"], r["id"])]

    # --- A <-> B matching
    TB = np.stack([r["thumb"] for r in B])
    for r in A:
        d = np.abs(TB - r["thumb"]).mean(axis=1)
        o = np.argsort(d)
        r["b_match"] = B[o[0]] if d[o[0]] < 2.0 and d[o[0]] / max(d[o[1]], 1e-9) < 0.5 else None
        r["match_mad"] = float(d[o[0]])
    in_A = {id(r["b_match"]): r for r in A if r["b_match"] is not None}

    # --- duplicates inside B (byte-identical)
    by_sha = defaultdict(list)
    for r in B:
        by_sha[r["sha"]].append(r)
    excluded = {}   # id(rec) -> reason
    conflict = set()
    for group in by_sha.values():
        if len(group) < 2:
            continue
        labels = {(g["level"], g["dme"]) for g in group}
        names = ", ".join(g["name"] for g in group)
        if len(labels) > 1:
            for g in group:
                conflict.add(id(g))
        has_test = any(g["split"] == "test" for g in group)
        for i, g in enumerate(group):
            if g["split"] == "test":
                continue
            if has_test:
                excluded[id(g)] = f"duplicate of a test image ({names})"
            elif len(labels) > 1:
                excluded[id(g)] = f"duplicate with conflicting grades ({names})"
            elif i > 0:
                excluded[id(g)] = f"duplicate, kept first copy ({names})"
    for r in B:
        if r["split"] == "train" and id(r) in in_A and id(r) not in excluded:
            excluded[id(r)] = f"also in Part A ({in_A[id(r)]['name']}); held out for localization"

    # --- train / val split (stratified by DR grade)
    pool = [r for r in B if r["split"] == "train" and id(r) not in excluded]
    rng = random.Random(a.seed)
    by_grade = defaultdict(list)
    for r in pool:
        by_grade[r["level"]].append(r)
    val, train, fold_of = [], [], {}
    for g, rs in sorted(by_grade.items()):
        rs = sorted(rs, key=lambda r: r["name"]); rng.shuffle(rs)
        k = round(len(rs) * a.val_frac)
        val += rs[:k]; train += rs[k:]
        for i, r in enumerate(rs):
            fold_of[id(r)] = i % a.folds
    test = [r for r in B if r["split"] == "test"]

    # --- write CSVs
    cols = ["image", "level", "dme", "onset1", "onset2", "source_split", "label_conflict", "filename"]
    def row(r):
        return [r["name"], r["level"], r["dme"], int(r["level"] >= 1), int(r["level"] >= 2),
                r["split"], int(id(r) in conflict), r["name"] + ".png"]
    def write(fname, rows, header):
        with open(os.path.join(a.out, "csv", fname), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    for fname, rs in (("train.csv", train), ("val.csv", val), ("test.csv", test)):
        write(fname, [row(r) for r in sorted(rs, key=lambda r: r["name"])], cols)
    write("cv_folds.csv", [row(r) + [fold_of[id(r)]] for r in sorted(pool, key=lambda r: r["name"])], cols + ["fold"])
    write("excluded.csv", [[r["name"], r["level"], r["dme"], excluded[id(r)]]
                           for r in B if id(r) in excluded], ["image", "level", "dme", "reason"])

    gt = os.path.join(a.root, "A. Segmentation", "2. All Segmentation Groundtruths")
    sdir_of = {v: k for k, v in SPLITS.items()}
    loc_rows, mask_jobs = [], []
    for r in A:
        m = r["b_match"]
        mrow = {}
        for ldir, code in LESIONS.items():
            src = os.path.join(gt, sdir_of[r["split"]], ldir, f"{r['id']}_{code}.tif")
            if (r["split"], r["id"], code) in BROKEN_MASKS:
                mrow[code] = "BROKEN"
            elif os.path.exists(src):
                dst = f"{r['name']}_{code}.png"
                mrow[code] = dst
                mask_jobs.append((src, os.path.join(a.out, "masks", dst), r["sq"], a.size))
            else:
                mrow[code] = ""
        loc_rows.append([r["name"], r["name"] + ".png", r["split"],
                         m["name"] if m else "", m["level"] if m else "", m["dme"] if m else "",
                         round(r["match_mad"], 3)] + [mrow[c] for c in ("MA", "HE", "EX", "SE", "OD")])
    write("localization.csv", loc_rows, ["image", "filename", "a_split", "b_match", "b_level", "b_dme",
                                         "match_mad", "mask_MA", "mask_HE", "mask_EX", "mask_SE", "mask_OD"])
    write("crop_params.csv", [[r["name"], r["W"], r["H"], *r["fov"], *r["sq"], round((r["sq"][2] - r["sq"][0]) / a.size, 4)]
                              for r in recs],
          ["image", "orig_w", "orig_h", "fov_x0", "fov_y0", "fov_x1", "fov_y1",
           "crop_x0", "crop_y0", "crop_x1", "crop_y1", "orig_px_per_out_px"])

    # --- summary
    def bal(rs):
        n = len(rs); p = sum(r["level"] >= 2 for r in rs)
        c = Counter(r["level"] for r in rs)
        return f"| {n} | {p} ({100*p/n:.1f}%) | {n-p} | " + " | ".join(str(c[k]) for k in range(5)) + " |"
    reasons = Counter(v.split(" (")[0] for v in excluded.values())
    S = ["# IDRiD split summary", "",
         f"Seed {a.seed}, val fraction {a.val_frac}, {a.folds} CV folds, output size {a.size}px.", "",
         "| Split | Images | DR ≥ 2 | DR < 2 | DR 0 | DR 1 | DR 2 | DR 3 | DR 4 |", "|---|---|---|---|---|---|---|---|---|",
         "| train " + bal(train), "| val " + bal(val), "| test " + bal(test), "| train+val (CV pool) " + bal(pool), "",
         f"Excluded from B-train ({len(excluded)}): " + ", ".join(f"{k}: {v}" for k, v in reasons.items()) + ".",
         f"Test images with conflicting duplicate labels (kept, flagged): "
         + (", ".join(r["name"] for r in test if id(r) in conflict) or "none") + ".",
         f"Localization set: {len(A)} Part A images, {sum(r['b_match'] is not None for r in A)} matched to a B grade."]
    open(os.path.join(a.out, "SPLIT_SUMMARY.md"), "w").write("\n".join(S) + "\n")
    print("\n".join(S))

    # --- normalization stats + images
    if not a.splits_only:
        jobs = [(r["path"], os.path.join(a.out, "images", r["name"] + ".png"), r["sq"], a.size) for r in recs]
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(process_image, jobs, chunksize=4))
            list(ex.map(process_mask, mask_jobs, chunksize=4))
        s1 = np.zeros(3); s2 = np.zeros(3); n = 0
        for r in train:
            x = np.asarray(Image.open(os.path.join(a.out, "images", r["name"] + ".png")), dtype=np.float64) / 255
            s1 += x.reshape(-1, 3).sum(0); s2 += (x.reshape(-1, 3) ** 2).sum(0); n += x.shape[0] * x.shape[1]
        mean = s1 / n; std = np.sqrt(s2 / n - mean ** 2)
        snippet = f"""# Paste over the 'Fundus' entry in configs/paths.yaml (then set base.dataset: Fundus)
# Copy {a.out}/csv/* to <repo>/csv_files/IDRiD/ and the images folder to <root>/<data_dir>.
Fundus:
  dset_name: IDRiD_{a.size}
  root: /home/jovyan/data/IDRiD
  data_dir: images
  data_dir_slurm: images
  target: "onset2"   # referable DR (grade >= 2)
  fname: "filename"
  train_csv: './csv_files/IDRiD/train.csv'
  val_csv: './csv_files/IDRiD/val.csv'
  test_csv: './csv_files/IDRiD/test.csv'
  mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]   # train split, includes black padding
  std: [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]
  num_classes: 2
  binary: True
  threshold: False
  input_size: {a.size}
"""
        open(os.path.join(a.out, "paths_snippet.yaml"), "w").write(snippet)
        print(f"\nimages: {len(jobs)}  masks: {len(mask_jobs)}  mean {mean.round(4)}  std {std.round(4)}")


if __name__ == "__main__":
    main()

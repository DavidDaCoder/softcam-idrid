"""Score a finished IDRiD run on train, val and test (classification only).

Why: the deck needs train-set results (rubric) and validation numbers to pick
hyperparameters without looking at the test set. eval_idrid_localization.py only scores test.

Usage (TIDE, softcam env; CPU is fine, ~1-2 min per run):
  python eval_idrid_splits.py --run <timestamp> [--ckpt acc] [--repo ~/SoftCAM]
        [--data ~/data/IDRiD] [--out ~/analysis/idrid]
  # every run in a manifest that has no splits.json yet:
  python eval_idrid_splits.py --manifest ~/analysis/idrid/seeds_<stamp>/manifest.csv

Writes <out>/<timestamp>/splits.json:
  {"run": ts, "ckpt": "acc", "train": {...}, "val": {...}, "test": {...}}
  each split: n, acc, auc, sens, spec, confusion [[tn, fp], [fn, tp]] (rows = true class)

Notes
- Images are scored without augmentation, exactly like eval_idrid_localization.py
  (same resize + the run's own mean/std from paths.yaml).
- "val" is the set the checkpoint was chosen on, so it is slightly optimistic; it is still the
  right set for choosing hyperparameters. Report test once, for the final configs.
- test numbers here must match test_predictions.csv / summary.json from the localization eval.
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import torch
import yaml
from PIL import Image


def find_run(run, repo_out):
    if os.path.isdir(run):
        return os.path.abspath(run)
    hits = glob.glob(os.path.join(repo_out, "*", run))
    if len(hits) != 1:
        sys.exit(f"Run '{run}' not found uniquely under {repo_out}/*/ (found {hits})")
    return hits[0]


def auc_score(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def score_run(run, a):
    run_dir = find_run(run, os.path.expanduser("~/Outputs/FullyConvNet"))
    ts = os.path.basename(run_dir.rstrip("/"))
    cfg = yaml.safe_load(open(os.path.join(run_dir, "configs", "default.yaml")))
    dset = yaml.safe_load(open(os.path.join(run_dir, "configs", "paths.yaml")))[cfg["base"]["dataset"]]
    sys.path.insert(0, a.repo)
    model = torch.load(os.path.join(run_dir, f"best_validation_weights_{a.ckpt}_m.pt"),
                       map_location=a.device, weights_only=False).eval()
    if hasattr(model, "module"):
        model = model.module
    mean = torch.tensor(dset["mean"]).view(3, 1, 1); std = torch.tensor(dset["std"]).view(3, 1, 1)
    size = int(dset["input_size"])

    def logit_diff(names):
        out = []
        with torch.no_grad():
            for i in range(0, len(names), 16):
                xs = []
                for n in names[i:i + 16]:
                    im = Image.open(os.path.join(a.data, "images", n)).convert("RGB")
                    if im.size != (size, size):
                        im = im.resize((size, size), Image.BILINEAR)
                    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)
                    xs.append((x - mean) / std)
                o = model(torch.stack(xs).to(a.device))
                L = (o[0] if isinstance(o, tuple) else o).cpu().numpy()
                out.append(L[:, 1] - L[:, 0])
        return np.concatenate(out)

    res = {"run": ts, "ckpt": a.ckpt, "lr": cfg["solver"]["learning_rate"],
           "weight_decay": cfg["solver"]["weight_decay"]}
    csvdir = os.path.join(a.repo, "csv_files", "IDRiD")
    for split in ("train", "val", "test"):
        rows = list(csv.DictReader(open(os.path.join(csvdir, f"{split}.csv"))))
        y = np.array([int(r["onset2"]) for r in rows]); s = logit_diff([r["filename"] for r in rows]); p = (s > 0).astype(int)
        tp, tn = int(((p == 1) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
        fp, fn = int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
        res[split] = {"n": len(y), "acc": (tp + tn) / len(y), "auc": auc_score(y, s),
                      "sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1), "confusion": [[tn, fp], [fn, tp]]}
        print(f"{ts} {split:5s} n={len(y):3d} acc {res[split]['acc']:.4f} auc {res[split]['auc']:.4f} "
              f"sens {res[split]['sens']:.4f} spec {res[split]['spec']:.4f} confusion {res[split]['confusion']}")
    od = os.path.join(a.out, ts); os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "splits.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--manifest")
    ap.add_argument("--ckpt", default="acc")
    ap.add_argument("--repo", default=os.path.expanduser("~/SoftCAM"))
    ap.add_argument("--data", default=os.path.expanduser("~/data/IDRiD"))
    ap.add_argument("--out", default=os.path.expanduser("~/analysis/idrid"))
    ap.add_argument("--redo", action="store_true", help="re-score runs that already have splits.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.run:
        score_run(a.run, a)
    elif a.manifest:
        for r in csv.DictReader(open(a.manifest)):
            if not a.redo and os.path.exists(os.path.join(a.out, r["timestamp"], "splits.json")):
                continue
            try:
                score_run(r["timestamp"], a)
            except (SystemExit, Exception) as e:   # keep going; one missing checkpoint shouldn't stop the rest
                print(f"FAILED {r['label']} s{r['seed']} {r['timestamp']}: {e}")
    else:
        sys.exit("give --run or --manifest")


if __name__ == "__main__":
    main()

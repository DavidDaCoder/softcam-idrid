"""Evaluate a SoftCAM run trained on IDRiD: test-set check + lesion localization on Part A.

Usage (on TIDE, softcam env, any directory; CPU is fine, ~1-3 min):
  python eval_idrid_localization.py --run <timestamp or run folder> [--ckpt acc]
        [--repo ~/SoftCAM] [--data ~/data/IDRiD] [--out ~/analysis/idrid]
        [--score map|contribution]

--score contribution (fair comparison for top-k runs): score the map of what each cell
  actually adds to the logit. For pooling: topk that is A_c * topk_mask_c * N/k per class
  (zero outside each class's top-k cells), so mean(contribution) = logit exactly. For mean
  pooling the contribution map equals the evidence map, so nothing changes. Multi-layer runs
  with per-stage top-k already return their contribution map. Writes to <out>/<timestamp>_contrib/.

Needs on TIDE:  <data>/images/*.png   (from prepare_idrid.py)
                <data>/masks/*.png    (from prepare_idrid.py; upload IDRiD_512/masks)
                <repo>/csv_files/IDRiD/{test,localization}.csv

What it computes
  1. Test set (103): accuracy/AUC/sens/spec from the loaded checkpoint. Must match the
     training log's numbers for the same checkpoint; if not, loading/preprocessing is wrong.
  2. Faithfulness: each class logit recomputed from its own evidence map with the run's
     pooling (mean, or mean of the top-k cells if the run's config says pooling: topk);
     max |difference| over all images should be ~1e-5.
  3. Localization on the 81 Part A images (all have lesion masks, so all are DR-positive
     in practice; 58 are matched to a B grade, all DR >= 2):
     - decision map D = class-1 map - class-0 map (16x16 for ResNet-50 at 512),
       upsampled to 512 by nearest neighbor (each cell = 32x32 px; keeps the mean).
     - evidence in lesions = sum(max(D,0) inside lesion union) / sum(max(D,0))
       lesion union = MA | HE | EX | SE (any coverage > 0 in the 512 masks).
     - chance = lesion union area / image area (what evenly spread evidence would get).
     - same per lesion type (only images that have that mask), and for the optic disc
       (shortcut check), and for the fundus field of view vs. black padding.
     - pointing game: does the single strongest positive cell contain any lesion pixel?
       chance = share of cells inside the field of view that contain lesion.
     Reported for all 81 and for true positives only (the RSNA protocol). 95% bootstrap CIs.
Outputs to <out>/<timestamp>/: test_predictions.csv, per_image.csv, summary.json, SUMMARY.md, maps.npz, cases.png
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import torch
import yaml
from PIL import Image

LES = ["MA", "HE", "EX", "SE"]


def find_run(run, repo_out):
    if os.path.isdir(run):
        return os.path.abspath(run)
    hits = glob.glob(os.path.join(repo_out, "*", run))
    if len(hits) != 1:
        sys.exit(f"Run '{run}' not found uniquely under {repo_out}/*/ (found {hits})")
    return hits[0]


def load_cfg(run_dir):
    with open(os.path.join(run_dir, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(run_dir, "configs", "paths.yaml")) as f:
        paths = yaml.safe_load(f)
    ds = cfg["base"]["dataset"]
    return cfg, paths[ds]


def auc_score(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v == v])
    if len(x) == 0:
        return [float("nan")] * 3
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="acc", help="acc, auc, sens, spec, prec, loss")
    ap.add_argument("--repo", default=os.path.expanduser("~/SoftCAM"))
    ap.add_argument("--data", default=os.path.expanduser("~/data/IDRiD"))
    ap.add_argument("--out", default=os.path.expanduser("~/analysis/idrid"))
    ap.add_argument("--score", default="map", choices=["map", "contribution"],
                    help="map: score the raw evidence map; contribution: score what each cell adds to the logit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    run_dir = find_run(a.run, os.path.expanduser("~/Outputs/FullyConvNet"))
    ts = os.path.basename(run_dir.rstrip("/"))
    cfg, dset = load_cfg(run_dir)
    out = os.path.join(a.out, ts + ("_contrib" if a.score == "contribution" else "")); os.makedirs(out, exist_ok=True)
    csvdir = os.path.join(a.repo, "csv_files", "IDRiD")

    # model: whole pickled model; class definitions come from the repo
    sys.path.insert(0, a.repo)
    ck = os.path.join(run_dir, f"best_validation_weights_{a.ckpt}_m.pt")
    model = torch.load(ck, map_location=a.device, weights_only=False).eval()
    if hasattr(model, "module"):
        model = model.module
    mean = torch.tensor(dset["mean"]).view(3, 1, 1); std = torch.tensor(dset["std"]).view(3, 1, 1)
    size = int(dset["input_size"])
    multilayer = str(cfg["train"].get("multilayer", "none") or "none")
    stage_pooling = str(cfg["train"].get("stage_pooling", "none") or "none")
    stage_model = hasattr(model, "keep_stage_maps")
    if stage_model:
        model.keep_stage_maps = True
    pooling = cfg["train"].get("pooling", "mean") or "mean"
    topk_frac = float(cfg["train"].get("topk_frac", 0.05))
    print(f"run {ts}  network {cfg['train']['network']}  conv_cls {cfg['train']['conv_cls']}  "
          f"pooling {pooling}{f' (top {topk_frac:.0%})' if pooling == 'topk' else ''}  "
          f"{'multilayer stages ' + multilayer + '  ' if stage_model else ''}"
          f"{'stage pooling ' + stage_pooling + '  ' if stage_model and stage_pooling != 'none' else ''}ckpt {os.path.basename(ck)}")

    def run_images(names):
        logits, maps, stages = [], [], []
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
                if isinstance(o, tuple):
                    logits.append(o[0].cpu()); maps.append(o[1].cpu())
                    if getattr(model, "stage_maps", None):
                        stages.append({k: v.cpu() for k, v in model.stage_maps.items()})
                else:
                    logits.append(o.cpu())
        L = torch.cat(logits).numpy()
        M = torch.cat(maps).numpy() if maps else None
        run_images.stage_maps = {k: torch.cat([b[k] for b in stages]).numpy() for k in stages[0]} if stages else None
        return L, M

    # ---------------- 1. test set check
    test = list(csv.DictReader(open(os.path.join(csvdir, "test.csv"))))
    Lt, Mt = run_images([r["filename"] for r in test])
    y = np.array([int(r["onset2"]) for r in test]); s = Lt[:, 1] - Lt[:, 0]; p = (s > 0).astype(int)
    tp, tn = int(((p == 1) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    fp, fn = int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    test_metrics = {"n": len(y), "acc": (tp + tn) / len(y), "auc": auc_score(y, s),
                    "sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1),
                    "confusion": [[tn, fp], [fn, tp]]}
    print(f"test: acc {test_metrics['acc']:.4f}  auc {test_metrics['auc']:.4f}  "
          f"sens {test_metrics['sens']:.4f}  spec {test_metrics['spec']:.4f}  confusion {test_metrics['confusion']}")
    with open(os.path.join(out, "test_predictions.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["image", "label", "level", "logit_diff", "prob_dr", "pred"])
        for r, yy, ss in zip(test, y, s):
            wr.writerow([r["image"], int(yy), r["level"], float(ss), float(1 / (1 + np.exp(-ss))), int(ss > 0)])
    if Mt is None:
        print("Model returns no evidence map (black-box run): localization skipped.")
        json.dump({"run": ts, "test": test_metrics}, open(os.path.join(out, "summary.json"), "w"), indent=1)
        return

    # ---------------- 2/3. Part A localization
    loc = list(csv.DictReader(open(os.path.join(csvdir, "localization.csv"))))
    La, Ma = run_images([r["filename"] for r in loc])
    Sa = run_images.stage_maps   # {stage: (n, 2, h_s, w_s)} weighted contributions w_s * A_s, or None
    # faithfulness: recompute each class logit from its own map with the run's pooling
    def pool(M):
        flat = M.reshape(M.shape[0], M.shape[1], -1)
        if pooling == "topk":
            k = max(1, round(topk_frac * flat.shape[2]))
            return -np.sort(-flat, axis=2)[:, :, :k].mean(axis=2)
        return flat.mean(axis=2)
    M_all = np.concatenate([Mt, Ma]); L_all = np.concatenate([Lt, La])
    faith = float(np.abs(pool(M_all) - L_all).max())
    print(f"faithfulness gap ({pooling} pooling, max over {len(L_all)} images): {faith:.2e}")
    contrib_applied = False
    if a.score == "contribution" and pooling == "topk":
        def contrib(M):   # A_c * topk_mask_c * N/k: zero outside each class's top-k cells; mean = logit
            n, c = M.shape[:2]; flat = M.reshape(n, c, -1); N = flat.shape[2]
            k = max(1, round(topk_frac * N))
            mask = np.zeros_like(flat)
            np.put_along_axis(mask, np.argsort(-flat, axis=2)[:, :, :k], 1.0, axis=2)
            return (flat * mask * (N / k)).reshape(M.shape).astype(np.float32)
        Mt, Ma = contrib(Mt), contrib(Ma); contrib_applied = True
        cf = float(np.abs(np.concatenate([Mt, Ma]).reshape(len(L_all), 2, -1).mean(2) - L_all).max())
        print(f"contribution maps: mean(contribution) vs logit, max gap {cf:.2e} (should be ~1e-5)")
        if cf > 1e-3:
            print("WARNING: contribution maps don't reproduce the logits.")
    elif a.score == "contribution":
        print(f"--score contribution: {pooling} pooling, so the contribution map is the evidence map (no change)")
    if faith > 1e-3:
        print("WARNING: logits don't match the maps under this pooling; check the run's model/config.")

    h, w = Ma.shape[2:]
    cell = size // h
    rows, maps_out = [], {}
    for idx, (r, lg, mp) in enumerate(zip(loc, La, Ma)):
        D = mp[1] - mp[0]
        Dup = np.kron(D, np.ones((cell, cell), dtype=np.float32))
        P = np.maximum(Dup, 0); Ptot = P.sum()
        img = np.asarray(Image.open(os.path.join(a.data, "images", r["filename"])).convert("L"))
        fov = img > 15
        masks = {}
        for c in LES + ["OD"]:
            f = r[f"mask_{c}"]
            masks[c] = (np.asarray(Image.open(os.path.join(a.data, "masks", f))) > 0) if f and f != "BROKEN" else None
        union = np.zeros((size, size), bool)
        for c in LES:
            if masks[c] is not None:
                union |= masks[c]
        share = lambda m: float((P * m).sum() / Ptot) if (m is not None and Ptot > 0) else float("nan")
        # pointing game on the cell grid
        cells_les = union.reshape(h, cell, w, cell).any(axis=(1, 3))
        cells_fov = fov.reshape(h, cell, w, cell).any(axis=(1, 3))
        imax = np.unravel_index(np.argmax(D), D.shape)
        row = {"image": r["image"], "b_match": r["b_match"], "b_level": r["b_level"],
               "logit_diff": float(lg[1] - lg[0]), "pred_pos": int(lg[1] > lg[0]),
               "pos_evidence_total": float(Ptot),
               "ev_lesions": share(union), "chance_lesions": float(union.mean()),
               "ev_fov": share(fov), "chance_fov": float(fov.mean()),
               "ev_OD": share(masks["OD"]), "chance_OD": float(masks["OD"].mean()) if masks["OD"] is not None else float("nan"),
               "point_hit": int(cells_les[imax] and D[imax] > 0),
               "point_chance": float(cells_les[cells_fov].mean()) if cells_fov.any() else float("nan")}
        for c in LES:
            row[f"ev_{c}"] = share(masks[c])
            row[f"chance_{c}"] = float(masks[c].mean()) if masks[c] is not None else float("nan")
        if Sa is not None:   # per stage: positive evidence of w_s * A_s on lesions / MA, at that stage's own cell size
            for st, arr in Sa.items():
                Ds = arr[idx, 1] - arr[idx, 0]
                Ps = np.maximum(np.kron(Ds, np.ones((size // Ds.shape[0], size // Ds.shape[1]), dtype=np.float32)), 0)
                tot = Ps.sum()
                row[f"s{st}_pos_share"] = float(tot / max(Ptot, 1e-12))
                for nm, m in (("lesions", union), ("MA", masks["MA"])):
                    row[f"s{st}_ev_{nm}"] = float((Ps * m).sum() / tot) if (m is not None and tot > 0) else float("nan")
        rows.append(row)
        maps_out[r["image"]] = D

    with open(os.path.join(out, "per_image.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    np.savez_compressed(os.path.join(out, "maps.npz"), **maps_out)

    def summarize(sub):
        o = {"n": len(sub)}
        for k in ["lesions", "OD", "fov"] + LES:
            ev = [x[f"ev_{k}"] for x in sub]; ch = [x[f"chance_{k}"] for x in sub]
            ratio = [e / c for e, c in zip(ev, ch) if e == e and c == c and c > 0]
            o[k] = {"evidence": boot_ci(ev), "chance": boot_ci(ch), "ratio_to_chance": boot_ci(ratio),
                    "n_images": int(sum(1 for e in ev if e == e))}
        o["pointing"] = {"hit_rate": boot_ci([x["point_hit"] for x in sub]),
                         "chance": boot_ci([x["point_chance"] for x in sub])}
        return o

    stage_summary = {}
    if Sa is not None:
        for st in Sa:
            stage_summary[str(st)] = {}
            for nm in ("lesions", "MA"):
                rr = [x[f"s{st}_ev_{nm}"] / x[f"chance_{nm}"] for x in rows
                      if x[f"s{st}_ev_{nm}"] == x[f"s{st}_ev_{nm}"] and x[f"chance_{nm}"] == x[f"chance_{nm}"] and x[f"chance_{nm}"] > 0]
                stage_summary[str(st)][nm] = boot_ci(rr)
            stage_summary[str(st)]["pos_share"] = boot_ci([x[f"s{st}_pos_share"] for x in rows])
    summ = {"run": ts, "ckpt": a.ckpt, "score": a.score, "contribution_applied": contrib_applied, "network": cfg["train"]["network"], "pooling": pooling,
            "multilayer": multilayer if stage_model else None, "stages": stage_summary or None,
            "stage_pooling": stage_pooling if stage_model else None,
            "topk_frac": topk_frac if pooling == "topk" else None, "map_size": [int(h), int(w)],
            "test": test_metrics, "faithfulness_gap_max": faith,
            "partA_detected": f"{sum(x['pred_pos'] for x in rows)}/{len(rows)}",
            "all_partA": summarize(rows),
            "true_positives": summarize([x for x in rows if x["pred_pos"] == 1])}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)

    # ---------------- markdown summary
    pct = lambda t: f"{100*t[0]:.1f}% [{100*t[1]:.1f}–{100*t[2]:.1f}]"
    md = [f"# IDRiD localization: {ts}", "",
          f"Network {summ['network']}, **{pooling} pooling**{f' (top {topk_frac:.0%} of cells)' if pooling == 'topk' else ''}"
          f"{f', stages {multilayer}' if stage_model else ''}{f', per-stage pooling {stage_pooling}' if stage_model and stage_pooling != 'none' else ''}, "
          f"checkpoint `{a.ckpt}`, evidence map {h}×{w} (one cell = {cell}×{cell} px at {size}).", "",
          *([f"**Scored on contribution maps** (each class map kept only on its top-k cells, ×N/k; mean = logit).", ""] if contrib_applied else []),
          f"**Test set:** acc {test_metrics['acc']:.3f}, AUC {test_metrics['auc']:.3f}, sens {test_metrics['sens']:.3f}, "
          f"spec {test_metrics['spec']:.3f}, confusion {test_metrics['confusion']} (rows true 0/1). Check these match the training log.", "",
          f"**Faithfulness gap:** {faith:.1e} (max over all test + Part A images).", "",
          f"**Part A detected as DR:** {summ['partA_detected']}.", ""]
    for name, key in (("All 81 Part A images", "all_partA"), ("True positives only (RSNA protocol)", "true_positives")):
        S = summ[key]
        md += [f"## {name} (n = {S['n']})", "",
               "| Region | Images | Evidence share (95% CI) | Area (chance) | Ratio to chance |", "|---|---|---|---|---|"]
        for k, label in (("lesions", "All lesions"), ("MA", "Microaneurysms"), ("HE", "Haemorrhages"),
                         ("EX", "Hard exudates"), ("SE", "Soft exudates"), ("OD", "Optic disc (shortcut check)"),
                         ("fov", "Fundus field of view")):
            e = S[k]
            md.append(f"| {label} | {e['n_images']} | {pct(e['evidence'])} | {100*e['chance'][0]:.2f}% | "
                      f"{e['ratio_to_chance'][0]:.1f}× [{e['ratio_to_chance'][1]:.1f}–{e['ratio_to_chance'][2]:.1f}] |")
        pg = S["pointing"]
        md += ["", f"Pointing game (strongest cell contains lesion): {pct(pg['hit_rate'])}, chance {100*pg['chance'][0]:.1f}%.", ""]
    if stage_summary:
        md += ["## Per stage (multi-layer): each stage's weighted map at its own resolution", "",
               "| Stage | Map | Lesions ÷ chance | Microaneurysms ÷ chance | Positive evidence vs fused map |", "|---|---|---|---|---|"]
        for st, v in stage_summary.items():
            hs = Sa[int(st)].shape[2]
            md.append(f"| {st} | {hs}×{hs} ({size // hs} px cells) | {v['lesions'][0]:.2f}× [{v['lesions'][1]:.2f}–{v['lesions'][2]:.2f}] | "
                      f"{v['MA'][0]:.2f}× [{v['MA'][1]:.2f}–{v['MA'][2]:.2f}] | {100*v['pos_share'][0]:.0f}% |")
        md += ["", "Last column: this stage's positive evidence relative to the fused map's (approximate: stages can partly cancel, so the column need not sum to 100%).", ""]
    md += ["Evidence share = positive decision-map evidence inside the region ÷ all positive evidence. "
           "Ratio to chance is computed per image, then averaged."]
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---------------- case figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        tps = sorted([x for x in rows if x["pred_pos"]], key=lambda x: -x["logit_diff"])
        pick = (tps[:3] + tps[len(tps) // 2:len(tps) // 2 + 1] + tps[-2:])[:6] if tps else []
        if pick:
            fig, ax = plt.subplots(2, len(pick), figsize=(3 * len(pick), 6))
            ax = np.atleast_2d(ax).reshape(2, -1)
            for j, x in enumerate(pick):
                r = next(q for q in loc if q["image"] == x["image"])
                img = np.asarray(Image.open(os.path.join(a.data, "images", r["filename"])).convert("RGB"))
                un = np.zeros((size, size), bool)
                for c in LES:
                    f = r[f"mask_{c}"]
                    if f and f != "BROKEN":
                        un |= np.asarray(Image.open(os.path.join(a.data, "masks", f))) > 0
                ov = img.copy(); ov[un] = [255, 255, 0]
                ax[0, j].imshow(ov); ax[0, j].set_title(f"{x['image']}\nlesions (yellow)", fontsize=8)
                D = maps_out[x["image"]]; v = np.abs(D).max()
                Dk = np.kron(D, np.ones((cell, cell)))
                rgba = plt.get_cmap("bwr")((Dk / max(v, 1e-12) + 1) / 2)
                rgba[..., 3] = 0.85 * np.abs(Dk) / max(v, 1e-12)      # near-zero cells see-through
                ax[1, j].imshow(img); ax[1, j].imshow(rgba)
                ax[1, j].set_title(f"in lesions {100*x['ev_lesions']:.0f}% (chance {100*x['chance_lesions']:.1f}%)", fontsize=8)
                for k in range(2):
                    ax[k, j].axis("off")
            plt.tight_layout(); plt.savefig(os.path.join(out, "cases.png"), dpi=110); plt.close()
    except Exception as e:
        print("figure skipped:", e)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

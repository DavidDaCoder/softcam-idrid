"""Figures, cases and paired comparisons for IDRiD SoftCAM runs.

Run AFTER eval_idrid_localization.py has been run on each model (CPU is fine):
  python make_idrid_figures.py \
      --run "Mean SC=26-10-2026_01:36:04" --run "Top-k SC=24-10-2026_06:27:36" \
      --run "Black-box=24-10-2026_06:32:01" [--out ~/analysis/idrid/figures]

The first two SoftCAM runs (runs with maps.npz) are compared pairwise (first = A, second = B).
Reads ~/analysis/idrid/<timestamp>/ (eval outputs), ~/data/IDRiD/{images,masks},
~/SoftCAM/csv_files/IDRiD/localization.csv.

Writes to --out:
  test_performance.png     acc / AUC / sensitivity / specificity per model, 95% bootstrap CIs
  roc_curves.png           ROC per model on the 103 test images
  localization_by_region.png  evidence-to-chance ratio per lesion type (SoftCAM runs)
  paired_lesion_evidence.png  per-image ratio, model A vs B (same 81 images)
  ratio_vs_lesion_area.png    does localization depend on how much lesion there is?
  cases_grid.png + cases/<image>.png/.txt   image | lesions | each model's evidence map
  ma_zoom.png              microaneurysms vs the 32-px evidence cells (resolution limit)
  RESULTS.md               all numbers, incl. paired tests
"""
import argparse, csv, json, math, os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# fixed colour per model slot (validated categorical palette, first three slots)
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LES_COL = {"MA": "#ff4fd8", "HE": "#4dd6ff", "EX": "#ffe14d", "SE": "#ffffff"}
LES_NAME = {"MA": "Microaneurysms", "HE": "Haemorrhages", "EX": "Hard exudates", "SE": "Soft exudates"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})


def rcsv(p):
    return list(csv.DictReader(open(p)))


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def metrics(y, s):
    y = np.asarray(y); pr = (np.asarray(s) > 0).astype(int)
    return {"Accuracy": (pr == y).mean(), "AUC": auc(y, s),
            "Sensitivity": pr[y == 1].mean(), "Specificity": 1 - pr[y == 0].mean()}


def boot(fn, n_items, n=2000, seed=0):
    r = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = r.integers(0, n_items, n_items)
        try:
            out.append(fn(idx))
        except Exception:
            pass
    out = np.array(out)
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def sign_test(d):
    d = [x for x in d if x != 0]; n = len(d); k = sum(x > 0 for x in d)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def outline(m):
    p = np.pad(m, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def overlay_lesions(ax, img, masks, od=None):
    o = img.copy()
    for c in ["SE", "HE", "EX", "MA"]:
        if masks.get(c) is not None:
            m = masks[c]
            o[m] = (0.45 * o[m] + 0.55 * np.array(Image.new("RGB", (1, 1), LES_COL[c]).getpixel((0, 0)))).astype(np.uint8)
            o[outline(m)] = Image.new("RGB", (1, 1), LES_COL[c]).getpixel((0, 0))
    ax.imshow(o)
    if od is not None and od.any():
        ax.contour(od, levels=[0.5], colors="white", linewidths=0.8, linestyles="--")


def evidence_rgba(D, cell, vmax):
    Dk = np.kron(D, np.ones((cell, cell)))
    rgba = plt.get_cmap("bwr")((Dk / vmax + 1) / 2)
    rgba[..., 3] = np.clip(0.85 * np.abs(Dk) / vmax, 0, 0.85)
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help='"Label=timestamp"')
    ap.add_argument("--analysis", default=os.path.expanduser("~/analysis/idrid"))
    ap.add_argument("--data", default=os.path.expanduser("~/data/IDRiD"))
    ap.add_argument("--csv", default=os.path.expanduser("~/SoftCAM/csv_files/IDRiD"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", type=int, default=6)
    a = ap.parse_args()
    out = a.out or os.path.join(a.analysis, "figures"); os.makedirs(os.path.join(out, "cases"), exist_ok=True)

    runs = []
    for i, spec in enumerate(a.run):
        label, ts = spec.split("=", 1)
        d = os.path.join(a.analysis, ts)
        R = {"label": label, "ts": ts, "dir": d, "color": SLOT[i % 3],
             "summary": json.load(open(os.path.join(d, "summary.json")))}
        tp = os.path.join(d, "test_predictions.csv")
        if not os.path.exists(tp):
            raise SystemExit(f"{tp} missing: re-run the updated eval_idrid_localization.py on {ts}")
        R["test"] = rcsv(tp)
        if os.path.exists(os.path.join(d, "maps.npz")):
            R["per"] = {r["image"]: r for r in rcsv(os.path.join(d, "per_image.csv"))}
            R["maps"] = dict(np.load(os.path.join(d, "maps.npz")))
        runs.append(R)
    sc = [R for R in runs if "maps" in R]
    md = ["# IDRiD results: figures and comparisons", "",
          "Runs: " + "; ".join(f"**{R['label']}** `{R['ts']}`" + (f" ({R['summary'].get('pooling')} pooling)" if 'maps' in R else "") for R in runs), ""]

    # ---------------- 1. test performance with CIs
    names = ["Accuracy", "AUC", "Sensitivity", "Specificity"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    md += ["## Test set (n = 103)", "", "| Model | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, R in enumerate(runs):
        y = np.array([int(r["label"]) for r in R["test"]]); s = np.array([float(r["logit_diff"]) for r in R["test"]])
        m = metrics(y, s)
        lo, hi = boot(lambda idx: [metrics(y[idx], s[idx])[k] for k in names], len(y))
        R["y"], R["s"] = y, s
        xs = np.arange(len(names)) + (i - (len(runs) - 1) / 2) * 0.22
        v = np.array([m[k] for k in names])
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="o", ms=6, color=R["color"], ecolor=R["color"],
                    elinewidth=1.5, capsize=0, label=R["label"], mec="white", mew=1)
        md.append(f"| {R['label']} | " + " | ".join(f"{m[k]:.3f} [{lo[j]:.2f}–{hi[j]:.2f}]" for j, k in enumerate(names)) + " |")
    lows = [l.get_ydata().min() for l in ax.lines if len(l.get_ydata())] or [0.5]
    ax.set_xticks(range(len(names)), names); ax.set_ylim(max(0, min(0.5, np.floor(min(lows) * 10) / 10 - 0.05)), 1.0)
    ax.set_ylabel("Score (95% CI)")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(runs), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(out, "test_performance.png")); plt.close(fig)

    # paired AUC differences vs first run
    if len(runs) > 1:
        md += ["", "Paired AUC difference vs first model (bootstrap over the same test images):", ""]
        y0 = runs[0]["y"]
        for R in runs[1:]:
            assert [r["image"] for r in R["test"]] == [r["image"] for r in runs[0]["test"]]
            dlt = auc(y0, R["s"]) - auc(y0, runs[0]["s"])
            lo, hi = boot(lambda idx: auc(y0[idx], R["s"][idx]) - auc(y0[idx], runs[0]["s"][idx]), len(y0))
            agree = int(((R["s"] > 0) == (runs[0]["s"] > 0)).sum())
            md.append(f"- {R['label']} − {runs[0]['label']}: ΔAUC {dlt:+.3f} [{lo:+.3f}, {hi:+.3f}]; "
                      f"same prediction on {agree}/{len(y0)} images.")

    # ---------------- 2. ROC
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    for R in runs:
        y, s = R["y"], R["s"]; th = np.r_[np.inf, np.sort(np.unique(s))[::-1]]
        tpr = [((s >= t) & (y == 1)).sum() / (y == 1).sum() for t in th]
        fpr = [((s >= t) & (y == 0)).sum() / (y == 0).sum() for t in th]
        ax.plot(fpr, tpr, color=R["color"], lw=2, label=f"{R['label']} (AUC {auc(y, s):.3f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(os.path.join(out, "roc_curves.png")); plt.close(fig)

    if not sc:
        open(os.path.join(out, "RESULTS.md"), "w").write("\n".join(md) + "\n"); return

    # ---------------- 3. localization by region
    regions = [("lesions", "All lesions"), ("MA", "Microaneurysms"), ("HE", "Haemorrhages"),
               ("EX", "Hard exudates"), ("SE", "Soft exudates"), ("OD", "Optic disc")]
    tick = {"lesions": "All\nlesions", "MA": "Micro-\naneurysms", "HE": "Haemor-\nrhages",
            "EX": "Hard\nexudates", "SE": "Soft\nexudates", "OD": "Optic disc\n(shortcut)"}
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    md += ["", "## Localization on Part A (all 81 images; ratio = evidence share ÷ area, per image)", "",
           "| Region | " + " | ".join(R["label"] for R in sc) + " |", "|---|" + "---|" * len(sc)]
    rows_md = {k: [] for k, _ in regions}
    for i, R in enumerate(sc):
        S = R["summary"]["all_partA"]
        xs = np.arange(len(regions)) + (i - (len(sc) - 1) / 2) * 0.25
        v = np.array([S[k]["ratio_to_chance"][0] for k, _ in regions])
        lo = np.array([S[k]["ratio_to_chance"][1] for k, _ in regions]); hi = np.array([S[k]["ratio_to_chance"][2] for k, _ in regions])
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="o", ms=6, color=R["color"], elinewidth=1.5, capsize=0,
                    label=R["label"], mec="white", mew=1)
        for k, _ in regions:
            rows_md[k].append(f"{S[k]['ratio_to_chance'][0]:.1f}× [{S[k]['ratio_to_chance'][1]:.1f}–{S[k]['ratio_to_chance'][2]:.1f}]")
    ax.axhline(1, color=INK2, lw=1, ls="--"); ax.text(-0.45, 1.05, "chance", color=INK2, ha="left", va="bottom", fontsize=8)
    ax.set_xticks(range(len(regions)), [tick[k] for k, _ in regions]); ax.set_ylabel("Evidence ÷ area (95% CI)")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=len(sc), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(out, "localization_by_region.png")); plt.close(fig)
    for k, n in regions:
        md.append(f"| {n} | " + " | ".join(rows_md[k]) + " |")
    md.append("| Pointing game (chance ~37%) | " + " | ".join(
        f"{100*R['summary']['all_partA']['pointing']['hit_rate'][0]:.0f}%" for R in sc) + " |")

    # ---------------- 4. paired comparison A vs B
    imgs = sorted(sc[0]["per"])
    if len(sc) >= 2:
        A, B = sc[0], sc[1]
        md += ["", f"## Paired comparison on the same 81 images: {B['label']} − {A['label']}", "",
               f"| Measure | Mean difference (95% CI) | {B['label']} higher / {A['label']} higher / tie | Sign-test p |", "|---|---|---|---|"]
        def ratio(R, im, k):
            e, c = float(R["per"][im][f"ev_{k}"]), float(R["per"][im][f"chance_{k}"])
            return e / c if (e == e and c == c and c > 0) else float("nan")
        for k, n in regions:
            pairs = [(ratio(A, im, k), ratio(B, im, k)) for im in imgs]
            pairs = np.array([p for p in pairs if p[0] == p[0] and p[1] == p[1]])
            d = pairs[:, 1] - pairs[:, 0]
            lo, hi = boot(lambda idx: d[idx].mean(), len(d))
            md.append(f"| {n} (ratio) | {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {(d>0).sum()} / {(d<0).sum()} / {(d==0).sum()} | {sign_test(d):.3g} |")
            if k == "lesions":
                lesion_pairs = pairs
        ha = np.array([int(A["per"][im]["point_hit"]) for im in imgs]); hb = np.array([int(B["per"][im]["point_hit"]) for im in imgs])
        md.append(f"| Pointing game hit | {hb.mean()-ha.mean():+.3f} | {((hb==1)&(ha==0)).sum()} / {((ha==1)&(hb==0)).sum()} / {(ha==hb).sum()} | {sign_test(list(hb-ha)):.3g} |")
        md += ["", f"Positive = {B['label']} puts more evidence there. For lesions higher is better; for the optic disc lower is better (less shortcut evidence). "
               "Mean difference CI: bootstrap over images; sign test: how many images go each way."]

        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        mx = np.nanmax(lesion_pairs) * 1.05
        ax.plot([0, mx], [0, mx], color=INK2, lw=1, ls="--")
        ax.axhline(1, color=GRID, lw=1); ax.axvline(1, color=GRID, lw=1)
        ax.scatter(lesion_pairs[:, 0], lesion_pairs[:, 1], s=22, color=B["color"], edgecolor="white", lw=0.8, zorder=3)
        nb = (lesion_pairs[:, 1] > lesion_pairs[:, 0]).sum()
        ax.text(0.03, 0.97, f"{B['label']} higher on {nb}/{len(lesion_pairs)} images", transform=ax.transAxes, va="top", fontsize=8, color=INK2)
        ax.set_xlabel(f"{A['label']}: evidence on lesions ÷ chance"); ax.set_ylabel(f"{B['label']}: evidence on lesions ÷ chance")
        ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_aspect("equal")
        fig.savefig(os.path.join(out, "paired_lesion_evidence.png")); plt.close(fig)

    # ---------------- 5. ratio vs lesion area
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for R in sc:
        x = np.array([float(R["per"][im]["chance_lesions"]) for im in imgs]) * 100
        yv = np.array([float(R["per"][im]["ev_lesions"]) for im in imgs]) / np.maximum(x / 100, 1e-9)
        ok = np.isfinite(yv)
        ax.scatter(x[ok], yv[ok], s=18, color=R["color"], edgecolor="white", lw=0.6, label=R["label"], alpha=0.9)
    ax.axhline(1, color=INK2, lw=1, ls="--"); ax.set_xscale("log")
    ax.set_xlabel("Lesion area (% of image, log scale)"); ax.set_ylabel("Evidence on lesions ÷ chance")
    ax.legend(frameon=False, fontsize=8); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.savefig(os.path.join(out, "ratio_vs_lesion_area.png")); plt.close(fig)

    # ---------------- 6. cases
    loc = {r["image"]: r for r in rcsv(os.path.join(a.csv, "localization.csv"))}
    def load_masks(im):
        r = loc[im]; m = {}
        for c in list(LES_COL) + ["OD"]:
            f = r[f"mask_{c}"]
            m[c] = (np.asarray(Image.open(os.path.join(a.data, "masks", f))) > 0) if f and f != "BROKEN" else None
        return m
    def rat(R, im):
        e, c = float(R["per"][im]["ev_lesions"]), float(R["per"][im]["chance_lesions"])
        return e / c if e == e and c > 0 else float("nan")
    picks = []
    def add(im, why):
        if im not in [p[0] for p in picks]:
            picks.append((im, why))
    if len(sc) >= 2:
        diff = {im: rat(sc[1], im) - rat(sc[0], im) for im in imgs if rat(sc[1], im) == rat(sc[1], im) and rat(sc[0], im) == rat(sc[0], im)}
        order = sorted(diff, key=diff.get)
        add(order[-1], f"largest gain for {sc[1]['label']}"); add(order[0], f"largest gain for {sc[0]['label']}")
        add(order[len(order) // 2], "typical (median difference)")
    best = max(imgs, key=lambda im: rat(sc[0], im) if rat(sc[0], im) == rat(sc[0], im) else -1); add(best, f"best localization ({sc[0]['label']})")
    ma_im = max(imgs, key=lambda im: float(sc[0]["per"][im]["chance_MA"] or 0) / max(float(sc[0]["per"][im]["chance_lesions"] or 1e-9), 1e-9))
    add(ma_im, "mostly microaneurysms (smallest lesions)")
    missed = [im for im in imgs if all(int(R["per"][im]["pred_pos"]) == 0 for R in sc)]
    if missed:
        add(missed[0], "missed by every model")
    picks = picks[:a.cases]

    ncol = 2 + len(sc)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(2.6 * ncol, 2.75 * len(picks)))
    axes = np.atleast_2d(axes)
    for i, (im, why) in enumerate(picks):
        img = np.asarray(Image.open(os.path.join(a.data, "images", loc[im]["filename"])).convert("RGB"))
        masks = load_masks(im); size = img.shape[0]
        axes[i, 0].imshow(img); axes[i, 0].set_title(f"{im}\n{why}", fontsize=7.5)
        overlay_lesions(axes[i, 1], img, masks, masks["OD"]); axes[i, 1].set_title(
            f"lesions: {100*float(sc[0]['per'][im]['chance_lesions']):.1f}% of image", fontsize=7.5)
        txt = [f"{im}: {why}", f"B-grade match: {loc[im]['b_match'] or 'none'} (DR {loc[im]['b_level'] or '?'})"]
        for j, R in enumerate(sc):
            D = R["maps"][im]; cell = size // D.shape[0]; vmax = np.abs(D).max()
            axes[i, 2 + j].imshow(img); axes[i, 2 + j].imshow(evidence_rgba(D, cell, vmax))
            p = R["per"][im]
            axes[i, 2 + j].set_title(f"{R['label']}: {'DR' if int(p['pred_pos']) else 'no DR'}, "
                                     f"{100*float(p['ev_lesions']):.0f}% on lesions\n"
                                     f"({rat(R, im):.1f}× chance), disc {100*float(p['ev_OD']):.0f}%", fontsize=7.5)
            txt.append(f"{R['label']}: predicted {'DR' if int(p['pred_pos']) else 'no DR'} (logit diff {float(p['logit_diff']):+.2f}); "
                       f"evidence on lesions {100*float(p['ev_lesions']):.1f}% vs area {100*float(p['chance_lesions']):.1f}% "
                       f"({rat(R, im):.1f}×); MA {100*float(p['ev_MA']):.1f}%, HE {100*float(p['ev_HE']):.1f}%, "
                       f"EX {100*float(p['ev_EX']):.1f}%; optic disc {100*float(p['ev_OD']):.1f}%; "
                       f"strongest cell on a lesion: {'yes' if int(p['point_hit']) else 'no'}.")
        for ax in axes[i]:
            ax.axis("off")
        # single-case figure
        f2, ax2 = plt.subplots(1, ncol, figsize=(2.9 * ncol, 3.1))
        ax2[0].imshow(img); overlay_lesions(ax2[1], img, masks, masks["OD"])
        for j, R in enumerate(sc):
            D = R["maps"][im]; cell = size // D.shape[0]; vmax = np.abs(D).max()
            ax2[2 + j].imshow(img); ax2[2 + j].imshow(evidence_rgba(D, cell, vmax)); ax2[2 + j].set_title(R["label"], fontsize=8)
        ax2[0].set_title(im, fontsize=8); ax2[1].set_title("lesion masks", fontsize=8)
        for x in ax2:
            x.axis("off")
        f2.savefig(os.path.join(out, "cases", f"{im}.png")); plt.close(f2)
        open(os.path.join(out, "cases", f"{im}.txt"), "w").write("\n".join(txt) + "\n")
    handles = [Line2D([], [], marker="s", ls="", color=LES_COL[c], mec="#888", label=LES_NAME[c]) for c in LES_COL]
    handles += [Line2D([], [], ls="--", color="#888", label="Optic disc"),
                Line2D([], [], marker="s", ls="", color="#d7301f", label="Evidence for DR"),
                Line2D([], [], marker="s", ls="", color="#2c5fbf", label="Evidence against DR")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.025, "Each evidence map uses its own colour scale (strongest cell = full colour); faint cells carry little evidence.",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(out, "cases_grid.png")); plt.close(fig)
    md += ["", "## Cases", ""] + [f"- `{im}`: {why} (see `cases/{im}.png`, `.txt`)" for im, why in picks]

    # ---------------- 7. microaneurysm zoom
    im = ma_im; masks = load_masks(im)
    img = np.asarray(Image.open(os.path.join(a.data, "images", loc[im]["filename"])).convert("RGB"))
    ma = masks["MA"]; ys, xs = np.nonzero(ma)
    if len(ys):
        # window (128 px) with the most microaneurysm pixels
        k = 128; cs = np.pad(ma.astype(int).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        best, by, bx = -1, 0, 0
        for y0 in range(0, img.shape[0] - k + 1, 16):
            for x0 in range(0, img.shape[1] - k + 1, 16):
                v = cs[y0 + k, x0 + k] - cs[y0, x0 + k] - cs[y0 + k, x0] + cs[y0, x0]
                if v > best:
                    best, by, bx = v, y0, x0
        crop = (slice(by, by + k), slice(bx, bx + k))
        fig, axs = plt.subplots(1, 1 + len(sc), figsize=(3.1 * (1 + len(sc)), 3.3))
        axs[0].imshow(img[crop], interpolation="nearest"); axs[0].contour(ma[crop], levels=[0.5], colors="#ff4fd8", linewidths=1.0); axs[0].set_title(f"{im}: microaneurysms (pink)\n{k}×{k} px crop of the 512 image", fontsize=8)
        for j, R in enumerate(sc):
            D = R["maps"][im]; cell = img.shape[0] // D.shape[0]; vmax = np.abs(D).max()
            axs[1 + j].imshow(img[crop], interpolation="nearest")
            axs[1 + j].imshow(evidence_rgba(D, cell, vmax)[crop], interpolation="nearest")
            for g in range(0, k + 1):
                if (by + g) % cell == 0:
                    axs[1 + j].axhline(g - 0.5, color="white", lw=0.6, alpha=0.7)
                if (bx + g) % cell == 0:
                    axs[1 + j].axvline(g - 0.5, color="white", lw=0.6, alpha=0.7)
            axs[1 + j].contour(ma[crop], levels=[0.5], colors="#ff4fd8", linewidths=0.8)
            axs[1 + j].set_title(f"{R['label']}: one evidence cell = {cell}×{cell} px", fontsize=8)
        for x in axs:
            x.axis("off")
        fig.savefig(os.path.join(out, "ma_zoom.png")); plt.close(fig)

    open(os.path.join(out, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()

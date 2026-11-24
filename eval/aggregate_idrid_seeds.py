"""Aggregate IDRiD runs across seeds (after eval_idrid_localization.py has run on each).

Usage: python aggregate_idrid_seeds.py --manifest manifest.csv --out <dir> [--analysis ~/analysis/idrid]
manifest.csv columns: label,seed,pooling,conv_cls,timestamp
Writes <out>/SEEDS_SUMMARY.md, seeds_table.csv, seeds_summary.png
"""
import argparse, csv, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948", "#8a8986", "#9c6b3c", "#127a8a", "#a64ca6"]; INK2, GRID = "#52514e", "#e4e3df"
FIXED = {"Mean SC": 0, "Top-k SC": 1, "Black-box": 2, "Multi SC": 3, "Swin SC": 4, "Swin Multi SC": 5,
         "S3 SC": 6, "Multi SC s3-topk": 7, "Top-k SC contrib": 2, "S3 SC topk": 8, "Multi SC both-topk": 9, "Multi SC s3-topk 1%": 10, "Multi SC s3-topk 10%": 11,
         "Mean SC lr 0.0003": 3, "Mean SC lr 0.003": 7, "Mean SC lr 0.01": 9}   # colour follows the model, not its position
TEST = ["acc", "auc", "sens", "spec"]
LOC = [("lesions", "All lesions"), ("MA", "Microaneurysms"), ("HE", "Haemorrhages"),
       ("EX", "Hard exudates"), ("SE", "Soft exudates"), ("OD", "Optic disc")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--analysis", default=os.path.expanduser("~/analysis/idrid"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.manifest)))
    data = defaultdict(dict)   # label -> seed -> metrics
    order = []
    for r in rows:
        p = os.path.join(a.analysis, r["timestamp"], "summary.json")
        if not os.path.exists(p):
            print("missing", p); continue
        S = json.load(open(p)); m = {k: S["test"][k] for k in TEST}
        if "all_partA" in S:
            for k, _ in LOC:
                m[k] = S["all_partA"][k]["ratio_to_chance"][0]
            m["pointing"] = S["all_partA"]["pointing"]["hit_rate"][0]
            m["faith"] = S["faithfulness_gap_max"]
        for st, v in (S.get("stages") or {}).items():
            m[f"s{st}_lesions"] = v["lesions"][0]; m[f"s{st}_MA"] = v["MA"][0]
        sp = os.path.join(a.analysis, r["timestamp"], "splits.json")   # from eval_idrid_splits.py
        if os.path.exists(sp):
            SP = json.load(open(sp))
            for split in ("train", "val"):
                for k in TEST:
                    m[f"{split}_{k}"] = SP[split][k]
        data[r["label"]][int(r["seed"])] = m
        if r["label"] not in order:
            order.append(r["label"])

    stage_cols = sorted({k for lab in data for d in data[lab].values() for k in d if k.startswith("s") and k[1].isdigit()})
    split_cols = [f"{sp}_{k}" for sp in ("train", "val") for k in TEST]
    cols = TEST + [k for k, _ in LOC] + ["pointing", "faith"] + stage_cols + split_cols
    with open(os.path.join(a.out, "seeds_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label", "seed"] + cols)
        for lab in order:
            for sd in sorted(data[lab]):
                w.writerow([lab, sd] + [data[lab][sd].get(c, "") for c in cols])

    def ms(lab, k):
        v = [d[k] for d in data[lab].values() if k in d]
        return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v)) if v else None

    md = ["# IDRiD across seeds", "", f"Seeds per model: " + ", ".join(f"{l} {sorted(data[l])}" for l in order), "",
          "Mean ± SD across seeds (test n = 103 per seed).", "",
          "| Model | Accuracy | AUC | Sensitivity | Specificity |", "|---|---|---|---|---|"]
    for lab in order:
        md.append(f"| {lab} | " + " | ".join(f"{ms(lab,k)[0]:.3f} ± {ms(lab,k)[1]:.3f}" for k in TEST) + " |")
    sp_labs = [l for l in order if any("val_acc" in d for d in data[l].values())]
    if sp_labs:
        md += ["", "Train / validation / test, mean ± SD across seeds (from eval_idrid_splits.py; no augmentation).",
               "Use validation to choose hyperparameters; train vs test shows overfitting.", "",
               "| Model | Seeds | Train acc | Val acc | Val AUC | Test acc | Test AUC |", "|---|---|---|---|---|---|---|"]
        for lab in sp_labs:
            f = lambda k: (lambda r: f"{r[0]:.3f} ± {r[1]:.3f}" if r else "—")(ms(lab, k))
            md.append(f"| {lab} | {ms(lab, 'val_acc')[2]} | {f('train_acc')} | {f('val_acc')} | {f('val_auc')} | {f('acc')} | {f('auc')} |")
    sc = [l for l in order if any("lesions" in d for d in data[l].values())]
    if sc:
        md += ["", "Localization on Part A (81 images), evidence ÷ chance, mean ± SD across seeds:", "",
               "| Region | " + " | ".join(sc) + " |", "|---|" + "---|" * len(sc)]
        for k, n in LOC + [("pointing", "Pointing game hit rate")]:
            md.append(f"| {n} | " + " | ".join(
                (f"{100*ms(l,k)[0]:.0f}% ± {100*ms(l,k)[1]:.0f}" if k == "pointing" else f"{ms(l,k)[0]:.2f} ± {ms(l,k)[1]:.2f}") for l in sc) + " |")
        md.append(f"| Max faithfulness gap | " + " | ".join(f"{max(d['faith'] for d in data[l].values()):.1e}" for l in sc) + " |")
        md += ["", "Pointing-game chance depends on cell size (~37% for 16×16 maps, ~22% for 32×32), so compare pointing only between models with the same map size."]
    for lab in sc:
        sts = sorted({k.split('_')[0] for d in data[lab].values() for k in d if k.startswith('s') and k[1].isdigit()})
        if sts:
            md += ["", f"Per stage for {lab} (each stage's weighted map at its own resolution), mean ± SD across seeds:", "",
                   "| Stage | Lesions ÷ chance | Microaneurysms ÷ chance |", "|---|---|---|"]
            for st in sts:
                md.append(f"| {st[1:]} | {ms(lab, st + '_lesions')[0]:.2f} ± {ms(lab, st + '_lesions')[1]:.2f} | "
                          f"{ms(lab, st + '_MA')[0]:.2f} ± {ms(lab, st + '_MA')[1]:.2f} |")
    A = sc[0] if sc else None
    for B in sc[1:]:
        seeds = sorted(set(data[A]) & set(data[B]))
        if not seeds:
            continue
        md += ["", f"Paired by seed ({B} − {A}, seeds {seeds}):", "",
               f"| Measure | Mean difference | Seeds where {B} is higher |", "|---|---|---|"]
        for k, n in [(t, t.upper() if t == "auc" else t.capitalize()) for t in TEST] + LOC + [("pointing", "Pointing game")]:
            d = [data[B][s][k] - data[A][s][k] for s in seeds]
            md.append(f"| {n} | {np.mean(d):+.3f} | {sum(x > 0 for x in d)}/{len(d)} |")
        md += ["", "For the optic disc, lower is better (less shortcut evidence)."]
    open(os.path.join(a.out, "SEEDS_SUMMARY.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # figure: small multiples, one panel per measure, dots = seeds, bar = mean
    panels = [("acc", "Test accuracy", order), ("auc", "Test AUC", order)]
    if sc:
        panels += [("lesions", "Lesion evidence ÷ chance", sc), ("MA", "Microaneurysm evidence ÷ chance", sc)]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0))
    for ax, (k, title, labs) in zip(np.atleast_1d(axes), panels):
        for i, lab in enumerate(labs):
            c = SLOT[FIXED.get(lab, order.index(lab)) % len(SLOT)]
            v = [d[k] for d in data[lab].values() if k in d]
            if k in ("lesions", "MA"):   # ratio panels start at 0, so bars are honest
                ax.bar(i, np.mean(v), width=0.6, color=c, alpha=0.25, edgecolor="none")
            else:                        # truncated axis: mean as a short line, not a bar
                ax.hlines(np.mean(v), i - 0.28, i + 0.28, color=c, lw=2)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=c, edgecolor="white", lw=0.8, zorder=3)
        ax.set_xticks(range(len(labs)), [l.replace(" ", "\n", 1) for l in labs], fontsize=8)
        ax.set_title(title, fontsize=9); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if k in ("lesions", "MA"):
            ax.axhline(1, color=INK2, lw=1, ls="--"); ax.set_ylim(bottom=0)
        else:
            lo = min(min(d[k] for d in data[l].values()) for l in labs); ax.set_ylim(max(0, lo - 0.08), 1.0)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "seeds_summary.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()

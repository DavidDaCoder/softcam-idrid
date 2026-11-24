# SoftCAM on IDRiD: self-explaining diabetic retinopathy screening

Code for my CSUSM final project: referable diabetic retinopathy (DR grade ≥ 2) vs. not on the
IDRiD fundus dataset, with SoftCAM models whose evidence map *is* the decision, evaluated on
both classification and how much of the evidence lands on doctor-traced lesions.

- **Model 1:** ResNet-50 SoftCAM, mean pooling (tuned: evidence stage, learning rate)
- **Model 2:** ResNet-50 SoftCAM, top-k pooling on the stage-3 map (tuned: stage, k)
- **Baseline:** black-box ResNet-50

Main result (10 seeds each, `results/SEEDS_SUMMARY.md`): same accuracy (test ≈ 0.84–0.86),
but stage-3 top-k puts 8.6× chance of its evidence on lesions vs 3.7× for standard SoftCAM
(microaneurysms 4.8× vs 3.0×).

## Built on
- **SoftCAM** — Djoumessi & Berens, MIDL 2026 (arXiv:2505.17748), code: https://github.com/kdjoumessi/SoftCAM (MIT).
  This repo does not copy that code; the scripts in `patches/` add features to a clone of it.
- **IDRiD** — Porwal et al., *Data* 3(3):25, 2018; data on IEEE DataPort, doi:10.21227/H25W98 (CC BY 4.0).
  Images are not included; download them from IEEE DataPort.

## Layout
| Folder | What's in it |
|---|---|
| `data_prep/` | `inspect_idrid.py` (overlap, duplicate and mask checks), `prepare_idrid.py` (crop to fundus, pad, 512 px, splits) |
| `splits/` | The exact train / val / test / localization splits used, plus `excluded.csv` (every removed image and why) |
| `patches/` | Install scripts that add top-k pooling (`add_topk.py`) and multi-stage evidence maps (`add_multilayer.py` + `multilayer_softcam.py`) to the SoftCAM repo. `add_topk.py` expects `add_vit.py` to have run first (ViT support; not used for IDRiD) |
| `configs/` | `default.yaml` (training config: ResNet-50, 70 epochs, SGD lr 0.001, λ = 0) and `paths_fundus.yaml` (the IDRiD dataset entry) |
| `run/` | `idrid_night.sh`: trains a list of models × seeds, then evaluates, scores train/val/test, makes figures and aggregates |
| `eval/` | Localization scoring, train/val/test scoring, cross-seed aggregation, figures |
| `results/` | Cross-seed summary tables for all runs |

## Reproduce
```bash
# 1. SoftCAM code + environment (see its README for requirements)
git clone https://github.com/kdjoumessi/SoftCAM ~/SoftCAM

# 2. Data: download IDRiD parts A and B, then
python data_prep/inspect_idrid.py /path/to/IDRiD reports/          # optional checks
python data_prep/prepare_idrid.py /path/to/IDRiD ~/data/IDRiD_512  # 512-px images, masks, split CSVs
#    The split CSVs should match splits/ (seed 0).
mkdir -p ~/SoftCAM/csv_files/IDRiD && cp splits/*.csv ~/SoftCAM/csv_files/IDRiD/
#    Paste configs/paths_fundus.yaml over the Fundus entry in ~/SoftCAM/configs/paths.yaml (set root),
#    and copy configs/default.yaml to ~/SoftCAM/configs/default.yaml

# 3. Add top-k and multi-stage SoftCAM to the repo
python patches/add_vit.py ~/SoftCAM && python patches/add_topk.py ~/SoftCAM && python patches/add_multilayer.py ~/SoftCAM

# 4. Train + evaluate (scripts expect the eval code in ~/analysis_code, data in ~/data/IDRiD)
mkdir -p ~/analysis_code && cp eval/* run/* ~/analysis_code/
DRY=1 MODELS="mean s3tk bb" SEEDS="0" bash ~/analysis_code/idrid_night.sh                # check config edits only
AUTOSTOP=0 MODELS="mean s3tk bb" SEEDS="0 1 2" bash ~/analysis_code/idrid_night.sh
```
Model keys: `mean` (Model 1), `s3tk` (Model 2), `bb` (black-box), `s3`, `multi`, `multitk3`,
`multitk34`, `multitk3_k1`, `multitk3_k10` (stage / k sweep), `mean_lr3e4`, `mean_lr3e3`, `mean_lr1e2`
(learning-rate sweep). `AUTOSTOP=0` turns off the JupyterHub server shutdown, which only works on
our cluster. Paths default to `~/SoftCAM`, `~/data/IDRiD`, `~/analysis/idrid`; most scripts take
`--repo`, `--data`, `--out` to change them.

`eval/fair_topk.sh` re-scores five specific top-k runs by their run IDs; it is kept for the record and
needs editing to use on new runs.

## AI use
Code written with Claude's help, with my instructor's permission.

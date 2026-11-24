"""
Install multi-layer SoftCAM into the SoftCAM repo. Safe to run more than once.
Run after add_vit.py and add_topk.py:   python ~/analysis_code/add_multilayer.py   (repo at ~/SoftCAM)

It (1) copies multilayer_softcam.py (from this script's folder) into modules/,
(2) patches modules/builder.py: when `multilayer` (under train:) is e.g. "3,4", a ResNet
    network is built as MultiLayerSoftCAM with evidence heads on those stages,
(3) adds `multilayer: none` and `stage_weights: equal` under train: in configs/default.yaml.
With `multilayer: none` everything behaves exactly as before.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "# multi-layer SoftCAM (added by add_multilayer.py)"
repo = Path(sys.argv[1] if len(sys.argv) > 1 else "~/SoftCAM").expanduser()
here = Path(__file__).resolve().parent
builder, cfgf = repo / "modules" / "builder.py", repo / "configs" / "default.yaml"
for f in (builder, cfgf):
    if not f.exists():
        sys.exit(f"Can't find {f}")

src = here / "multilayer_softcam.py"
if not src.exists():
    sys.exit(f"Put multilayer_softcam.py next to this script ({here}) and rerun.")
shutil.copy(src, repo / "modules" / "multilayer_softcam.py")
print("copied     multilayer_softcam.py -> modules/")

text = builder.read_text()
SP_OLD = "pooling=cfg.train.get('pooling', 'mean'), topk_frac=cfg.train.get('topk_frac', 0.05))\n"
SP_NEW = ("pooling=cfg.train.get('pooling', 'mean'), topk_frac=cfg.train.get('topk_frac', 0.05),\n"
          "                                 stage_pooling=cfg.train.get('stage_pooling', 'none'))\n")
if MARK in text and "stage_pooling" not in text[text.index(MARK):]:
    i = text.index(MARK); j = text.find(SP_OLD, i)
    if j < 0:
        sys.exit("builder.py: multi-layer block found but its return line differs; patch by hand.")
    builder.write_text(text[:j] + SP_NEW + text[j + len(SP_OLD):])
    print("updated    modules/builder.py (multi-layer route now passes stage_pooling)")
elif MARK in text:
    print("builder.py already patched")
else:
    anchor = "    pretrained = cfg.train.pretrained\n"
    if anchor not in text:
        sys.exit("builder.py: anchor 'pretrained = cfg.train.pretrained' not found; patch by hand.")
    shutil.copy(builder, builder.with_suffix(".py.pre_multilayer"))
    patch = (anchor
             + f"\n    {MARK}\n"
             + "    _ml = str(cfg.train.get('multilayer', 'none') or 'none').strip().lower()\n"
             + "    if _ml not in ('none', 'null', 'false', '0', ''):\n"
             + "        if not ('resnet' in network or 'resnext' in network):\n"
             + "            raise NotImplementedError('multilayer SoftCAM is implemented for ResNet/ResNeXt only')\n"
             + "        from .multilayer_softcam import MultiLayerSoftCAM\n"
             + "        base = BUILDER[network](weights=WEIGHTS[network])\n"
             + "        return MultiLayerSoftCAM(base, num_classes, stages=[int(s) for s in _ml.split(',')],\n"
             + "                                 stage_weights=cfg.train.get('stage_weights', 'equal'),\n"
             + "                                 pooling=cfg.train.get('pooling', 'mean'), topk_frac=cfg.train.get('topk_frac', 0.05),\n"
             + "                                 stage_pooling=cfg.train.get('stage_pooling', 'none'))\n")
    builder.write_text(text.replace(anchor, patch, 1))
    print("patched    modules/builder.py (backup builder.py.pre_multilayer)")

c = cfgf.read_text()
if re.search(r"^  multilayer:", c, re.M) and not re.search(r"^  stage_pooling:", c, re.M):
    m = re.search(r"^  stage_weights:.*\n", c, re.M) or re.search(r"^  multilayer:.*\n", c, re.M)
    cfgf.write_text(c[:m.end()] + "  stage_pooling: none\n" + c[m.end():])
    print("added      stage_pooling: none to configs/default.yaml")
elif re.search(r"^  multilayer:", c, re.M):
    print("default.yaml already has multilayer and stage_pooling")
else:
    shutil.copy(cfgf, cfgf.with_suffix(".yaml.pre_multilayer"))
    m = (re.search(r"^  topk_frac:.*\n", c, re.M) or re.search(r"^  pooling:.*\n", c, re.M)
         or re.search(r"^  lambda_l2:.*\n", c, re.M))
    if not m:
        sys.exit("default.yaml: no pooling/topk_frac/lambda_l2 line under train: to anchor on")
    c = c[:m.end()] + "  multilayer: none\n  stage_weights: equal\n  stage_pooling: none\n" + c[m.end():]
    cfgf.write_text(c)
    print("added      multilayer: none, stage_weights: equal, stage_pooling: none to configs/default.yaml")
print('done. Under train: `multilayer: 3,4` = stages 3 + 4; `multilayer: 3` = stage 3 only; '
      '`stage_pooling: topk,mean` = top-k on stage 3, mean on stage 4 (with pooling: mean).')

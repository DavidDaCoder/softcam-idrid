"""
Add top-k evidence pooling to the SoftCAM repo. Safe to run more than once.
Run AFTER add_vit.py:   python ~/add_topk.py   (repo at ~/SoftCAM)

With `pooling: topk` and `topk_frac: 0.05` under `train:` in configs/default.yaml,
each class logit is the mean of the strongest 5% of cells in that class's evidence
map (13 of 16x16 for ResNet/VGG, 51 of 32x32 for ViT/DeiT) instead of all cells.
Without those keys (or `pooling: mean`) everything behaves exactly as before.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "# top-k pooling (added by add_topk.py)"
repo = Path(sys.argv[1] if len(sys.argv) > 1 else "~/SoftCAM").expanduser()
models = repo / "modules" / "models.py"
builder = repo / "modules" / "builder.py"
for f in (models, builder):
    if not f.exists():
        sys.exit(f"Can't find {f}")

# 1. models.py: FullyConvModel (ResNet/VGG) only; file uses CRLF line endings
raw = models.read_bytes().decode()
nl = "\r\n" if "\r\n" in raw else "\n"
if MARK in raw:
    print("models.py already patched")
else:
    shutil.copy(models, models.with_suffix(".py.pre_topk"))
    m = re.search(r"class FullyConvModel\(nn\.Module\):", raw)
    if not m:
        sys.exit("FullyConvModel not found in models.py")
    head, body = raw[:m.start()], raw[m.start():]
    a = re.search(r"( +)network = cfg\.train\.network[^\r\n]*\r?\n", body)
    b = re.search(r"( +)out = out\.view\(out\.shape\[0\], -1\)[^\r\n]*\r?\n", body)
    if not a or not b or b.start() < a.end():
        sys.exit("models.py: expected lines not found; patch by hand.")
    ind = a.group(1)
    init_add = (f"{ind}{MARK}{nl}"
                f"{ind}self.pooling = cfg.train.get('pooling', 'mean'){nl}"
                f"{ind}self.topk_frac = cfg.train.get('topk_frac', 0.05){nl}")
    ind2 = b.group(1)
    fwd_add = (f"{ind2}if getattr(self, 'pooling', 'mean') == 'topk':  {MARK}{nl}"
               f"{ind2}    flat = activation.flatten(2){nl}"
               f"{ind2}    k = max(1, round(self.topk_frac * flat.shape[2])){nl}"
               f"{ind2}    out = flat.topk(k, dim=2).values.mean(dim=2){nl}")
    body = body[:b.end()] + fwd_add + body[b.end():]          # forward first (later offset)
    body = body[:a.end()] + init_add + body[a.end():]
    models.write_bytes((head + body).encode())
    print("patched    modules/models.py (FullyConvModel; backup models.py.pre_topk)")

# 2. builder.py: pass pooling settings to ViTSoftCAM
text = builder.read_text()
old = "return ViTSoftCAM(network, num_classes, img_size=cfg.data.input_size, pretrained=pretrained)"
new = ("return ViTSoftCAM(network, num_classes, img_size=cfg.data.input_size, pretrained=pretrained, "
       "pooling=cfg.train.get('pooling', 'mean'), topk_frac=cfg.train.get('topk_frac', 0.05))")
if new in text:
    print("builder.py already passes pooling")
elif old in text:
    builder.write_text(text.replace(old, new, 1))
    print("patched    modules/builder.py (ViT gets pooling settings)")
else:
    sys.exit("builder.py: ViT line not found. Run add_vit.py first.")

# 3. the installed ViT model must be the pooling-aware version
vit = repo / "modules" / "vit_softcam.py"
if not vit.exists() or "topk_pool" not in vit.read_text():
    sys.exit("modules/vit_softcam.py is the old version: upload the new vit_softcam.py to ~ and rerun add_vit.py, then this.")
print("vit_softcam.py supports pooling")
print("done. Add under train: `pooling: topk` and `topk_frac: 0.05` to use it.")

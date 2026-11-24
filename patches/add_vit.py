"""
Install ViT/DeiT SoftCAM into the SoftCAM repo. Safe to run more than once.

  python ~/add_vit.py              # repo at ~/SoftCAM
  python ~/add_vit.py /path/to/SoftCAM

It (1) copies vit_softcam.py (from this script's folder) into modules/,
(2) patches modules/builder.py so `network: vit_...` or `deit_...` builds ViTSoftCAM,
(3) lists the ViT/DeiT names in configs/networks.yaml.
"""
import shutil
import sys
from pathlib import Path

MARKER = "# ViT SoftCAM (added by add_vit.py)"
VIT_NAMES = [
    "vit_small_patch16_224", "vit_base_patch16_224",
    "deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224",
]
OLD_COND = "    if network.startswith('vit'):\n"
NEW_COND = "    if network.startswith(('vit', 'deit')):\n"

repo = Path(sys.argv[1] if len(sys.argv) > 1 else "~/SoftCAM").expanduser()
here = Path(__file__).resolve().parent
builder = repo / "modules" / "builder.py"
networks = repo / "configs" / "networks.yaml"
if not builder.exists():
    sys.exit(f"Can't find {builder}. Pass the repo path: python add_vit.py /path/to/SoftCAM")

# 1. model file
src = here / "vit_softcam.py"
if not src.exists():
    sys.exit(f"Put vit_softcam.py next to this script ({here}) and rerun.")
shutil.copy(src, repo / "modules" / "vit_softcam.py")
print(f"copied     {src} -> modules/vit_softcam.py")

# 2. builder patch
text = builder.read_text()
if MARKER in text and OLD_COND in text:
    builder.write_text(text.replace(OLD_COND, NEW_COND, 1))
    print("updated    modules/builder.py to also route deit_* networks")
elif MARKER in text:
    print("builder.py already patched")
else:
    anchor = "    pretrained = cfg.train.pretrained\n"
    if anchor not in text:
        sys.exit("builder.py: anchor line 'pretrained = cfg.train.pretrained' not found; patch by hand.")
    backup = builder.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copy(builder, backup)
    patch = (
        anchor
        + f"\n    {MARKER}\n"
        + NEW_COND
        + "        from .vit_softcam import ViTSoftCAM\n"
        + "        return ViTSoftCAM(network, num_classes, img_size=cfg.data.input_size, pretrained=pretrained)\n"
    )
    builder.write_text(text.replace(anchor, patch, 1))
    print(f"patched    modules/builder.py (original saved as {backup.name})")

# 3. network list
if networks.exists():
    ntext = networks.read_text()
    missing = [n for n in VIT_NAMES if n not in ntext]
    if missing:
        if not ntext.endswith("\n"):
            ntext += "\n"
        ntext += "".join(f"  - {n}\n" for n in missing)
        networks.write_text(ntext)
        print(f"added      {', '.join(missing)} to configs/networks.yaml")
    else:
        print("networks.yaml already lists the ViTs")

print("done. Set `network: deit_small_patch16_224` (or a vit_ name) in configs/default.yaml to use it.")

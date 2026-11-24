"""
ViT SoftCAM: SoftCAM's class-evidence layer on a Vision Transformer.

A standard ViT classifies from its CLS token, which hides where the evidence
came from. This model instead:
  1. runs the ViT and keeps the patch tokens from the last layer (CLS dropped),
  2. arranges them back into their 2D grid (512 input / 16 patch = 32x32),
  3. applies a 1x1 convolution (the class-evidence layer) -> one map per class,
  4. averages each map to get the logits.

The prediction is exactly the average of the evidence map, as in CNN SoftCAM.
forward() returns (logits, evidence_maps), the same interface as the repo's
FullyConvModel, so train.py, the L1/L2 penalty and evaluation work unchanged.

pooling='topk' (set via `pooling: topk` / `topk_frac:` in the config, see
add_topk.py) predicts from the mean of the strongest topk_frac of cells in each
class map instead of all cells. Still exactly faithful: each logit equals the
mean of the selected cells of that class's evidence map.
"""
import timm
import torch
import torch.nn as nn

# Large speedup on A100 for transformer matmuls (convolutions already use TF32).
torch.backends.cuda.matmul.allow_tf32 = True


def topk_pool(activation, frac):
    """Mean of the top `frac` share of cells in each class map: (bs, n_class, h, w) -> (bs, n_class)."""
    flat = activation.flatten(2)
    k = max(1, round(frac * flat.shape[2]))
    return flat.topk(k, dim=2).values.mean(dim=2)


class ViTSoftCAM(nn.Module):
    def __init__(self, vit_name, num_classes, img_size=512, pretrained=True,
                 pooling='mean', topk_frac=0.05):
        super().__init__()
        assert pooling in ('mean', 'topk'), pooling
        self.pooling, self.topk_frac = pooling, topk_frac
        # num_classes=0 removes timm's classification head; img_size makes timm
        # resample the pretrained position embeddings to the new patch grid.
        self.vit = timm.create_model(
            vit_name, pretrained=pretrained, num_classes=0, img_size=img_size
        )
        self.num_prefix = self.vit.num_prefix_tokens  # CLS (+ registers)
        self.grid = self.vit.patch_embed.grid_size            # e.g. (32, 32)
        self.classifier = nn.Conv2d(
            self.vit.embed_dim, num_classes, kernel_size=1, stride=1
        )

    def forward(self, x):
        tokens = self.vit.forward_features(x)        # (bs, prefix + h*w, d), final norm applied
        patches = tokens[:, self.num_prefix:]       # drop CLS: (bs, h*w, d)
        bs, n, d = patches.shape
        h, w = self.grid
        assert n == h * w, f"expected {h*w} patch tokens, got {n} (input size mismatch?)"
        feat = patches.transpose(1, 2).reshape(bs, d, h, w)  # (bs, d, h, w)
        activation = self.classifier(feat)                  # (bs, n_class, h, w)
        if getattr(self, 'pooling', 'mean') == 'topk':  # getattr: checkpoints pickled before pooling existed
            out = topk_pool(activation, getattr(self, 'topk_frac', 0.05))
        else:
            out = activation.mean(dim=(2, 3))                # (bs, n_class)
        return out, activation

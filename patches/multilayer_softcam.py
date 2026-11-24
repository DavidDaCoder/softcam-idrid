"""
Multi-layer SoftCAM for ResNet backbones.

Each chosen stage s gets its own 1x1-conv evidence head A_s (n_class maps). The class logit is
a weighted sum of per-stage pooled evidence; with mean pooling and weights summing to 1:

    logit = sum_s w_s * mean(A_s)

Explanation: upsample every A_s to the finest chosen grid by nearest neighbour (each cell copied
into an integer k x k block, which keeps its mean) and take the same weighted sum:

    F = sum_s w_s * up(A_s)        ->   mean(F) = logit exactly

So the single fused map F is as faithful as standard SoftCAM, at the finest stage's resolution
(ResNet-50 at 512 px: stage 3 = 32x32 cells of 16 px, stage 4 = 16x16 cells of 32 px).
With pooling='topk' the logit is the mean of the top-k cells of F (faithful to F by construction).

Per-stage pooling (stage_pooling="topk,mean", one entry per stage): a 'topk' stage contributes
w_s * (mean of its top-k cells). Its contribution map keeps only those k cells, scaled by N_s / k
(N_s = cells in that map), and zeros elsewhere, so the map's mean is exactly its contribution and
each cell's value is exactly what that cell adds to the logit. F = sum_s up(contribution_s) still
has mean(F) = logit exactly. stage_pooling=None (default) keeps the behaviour above.
A single stage (stages=(3,)) is plain SoftCAM on that stage; later stages are never run.

forward() returns (logits, F), the same interface as the repo's FullyConvModel, so train.py,
the L1/L2 penalties and the evaluation code work unchanged. Set `keep_stage_maps = True` to also
store the weighted per-stage contributions w_s * A_s (detached) in `self.stage_maps` for analysis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_pool(activation, frac):
    flat = activation.flatten(2)
    k = max(1, round(frac * flat.shape[2]))
    return flat.topk(k, dim=2).values.mean(dim=2)


class MultiLayerSoftCAM(nn.Module):
    def __init__(self, resnet, num_classes, stages=(3, 4), stage_weights="equal",
                 pooling="mean", topk_frac=0.05, stage_pooling=None):
        super().__init__()
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layers = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])
        self.stages = tuple(sorted(int(s) for s in stages))
        if not self.stages or any(s not in (1, 2, 3, 4) for s in self.stages):
            raise ValueError(f"stages must be from 1-4, got {stages}")
        heads = {}
        for s in self.stages:
            last = list(self.layers[s - 1].children())[-1]
            ch = last.conv3.out_channels if hasattr(last, "conv3") else last.conv2.out_channels
            heads[str(s)] = nn.Conv2d(ch, num_classes, kernel_size=1)
        self.heads = nn.ModuleDict(heads)
        if stage_weights in (None, "equal"):
            w = [1.0] * len(self.stages)
        else:
            w = [float(v) for v in (stage_weights.split(",") if isinstance(stage_weights, str) else stage_weights)]
            if len(w) != len(self.stages):
                raise ValueError("stage_weights must have one value per stage")
        w = torch.tensor(w) / sum(w)
        self.register_buffer("stage_w", w)
        self.pooling, self.topk_frac = pooling or "mean", float(topk_frac)
        sp = None if stage_pooling in (None, "", "none", "null") else (
            stage_pooling.split(",") if isinstance(stage_pooling, str) else list(stage_pooling))
        if sp is not None:
            sp = [x.strip().lower() for x in sp]
            if len(sp) != len(self.stages) or any(x not in ("mean", "topk") for x in sp):
                raise ValueError(f"stage_pooling needs one of mean/topk per stage {self.stages}, got {stage_pooling}")
            if self.pooling != "mean":
                raise ValueError("use pooling: mean together with stage_pooling (the fused map is then mean-pooled)")
        self.stage_pooling = sp
        self.keep_stage_maps = False
        self.stage_maps = None

    def forward(self, x):
        x = self.stem(x)
        maps = {}
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if i in self.stages:
                maps[i] = self.heads[str(i)](x)
            if i == self.stages[-1]:
                break
        h = max(m.shape[2] for m in maps.values()); w = max(m.shape[3] for m in maps.values())
        fused = 0
        contrib = {}
        sp = getattr(self, "stage_pooling", None)   # getattr: models pickled before stage_pooling existed
        for idx, (wi, s) in enumerate(zip(self.stage_w, self.stages)):
            a = maps[s]
            if h % a.shape[2] or w % a.shape[3]:
                raise RuntimeError(f"stage {s} map {tuple(a.shape[2:])} does not divide {(h, w)}")
            if sp is not None and sp[idx] == "topk":
                flat = a.flatten(2); n = flat.shape[2]; k = max(1, round(self.topk_frac * n))
                idx_k = flat.topk(k, dim=2).indices
                mask = torch.zeros_like(flat).scatter_(2, idx_k, 1.0).view_as(a)
                c = wi * a * mask * (n / k)          # mean(c) = wi * mean of the top-k cells
            else:
                c = wi * a
            if self.keep_stage_maps:
                contrib[s] = c.detach()
            fused = fused + F.interpolate(c, size=(h, w), mode="nearest")
        self.stage_maps = contrib if self.keep_stage_maps else None
        out = topk_pool(fused, self.topk_frac) if self.pooling == "topk" else fused.mean(dim=(2, 3))
        return out, fused

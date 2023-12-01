from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .edsr import make_edsr_baseline
from mmdet3d.models.builder import HEADS

__all__ = ["MapSegmentationHeadSRTransformer"]


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
        scale_factor: int = 4,
        finetune: bool = True,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor
        self.finetune = finetune

        self.upsampler = make_edsr_baseline(
            n_feats=512,
            scale=scale_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        # uniform grid
        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        # BEV range
        x = F.grid_sample(
            x,
            grid,
            mode="nearest",
            align_corners=False,
        )

        # super-resolution
        if self.finetune:
            x = self.upsampler(x.detach())

        return x


@HEADS.register_module()
class MapSegmentationHeadSRTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_transform: Dict[str, Any],
        classes: List[str],
        loss: str,
        num_heads: int,
        finetune: bool,
        num_sa : int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss

        self.finetune = finetune
        self.num_heads = num_heads
        self.scale = in_channels ** -0.5
        self.num_sa = num_sa

        grid_transform["finetune"] = finetune
        self.transform = BEVGridTransform(**grid_transform)

        qkv = []
        proj = []
        for _ in range(self.num_sa):
            qkv.append(nn.Linear(in_channels, in_channels*3, bias=False))
            proj.append(nn.Linear(in_channels, in_channels))

        self.qkv = nn.ModuleList(qkv)
        self.proj = nn.ModuleList(proj)

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1),
        )

        self.classifier_hr = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]

        if self.finetune:
            B, C, H, W = x.shape
            # self-attention
            for i in range(self.num_sa):
                q, k, v = (
                    self.qkv[i](x.flatten(2).permute(0,2,1))
                    .reshape(B, H*W, 3, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)

                res = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
                res = self.proj[i](res).reshape(B, H, W, -1).permute(0, 3, 1, 2)
                x = x + res
            x = self.transform(x) # SR networks
            x = self.classifier_hr(x) # decoding CNN

        else:
            x = self.transform(x)
            x = self.classifier(x)

        if self.training:
            losses = {}
            for index, name in enumerate(self.classes):
                if self.loss == "xent":
                    loss = sigmoid_xent_loss(x[:, index], target[:, index])
                elif self.loss == "focal":
                    loss = sigmoid_focal_loss(x[:, index], target[:, index])
                else:
                    raise ValueError(f"unsupported loss: {self.loss}")
                losses[f"{name}/{self.loss}"] = loss
            return losses
        else:
            return torch.sigmoid(x)

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .edsr import make_edsr_baseline

from mmdet3d.models.builder import HEADS

__all__ = ["DoubleBellTNN"]


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
        finetune: bool = False,
        inference: bool = False,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

        self.finetune = finetune
        self.inference = inference

        self.upsampler2 = make_edsr_baseline(
            n_feats=512,
            scale=scale_factor
        )

        self.upsampler = make_edsr_baseline(
            n_feats=512,
            scale=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.inference:
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

            x = F.grid_sample(
                x,
                grid,
                mode="nearest",
                align_corners=False,
            )
        
            if self.finetune:
                x = self.upsampler2(x.detach())

            else:
                x = self.upsampler(x.detach())

            return x

        else:
            coords = []
            for (imin, imax, _), (omin, omax, ostep) in zip(
                # self.input_scope, self.output_scope
                self.input_scope, [[-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5]]
            ):
                v = torch.arange(omin/2 + ostep / 2, omax/2, ostep)
                v = (v - imin) / (imax - imin) * 2 - 1
                coords.append(v.to(x.device))

            u, v = torch.meshgrid(coords, indexing="ij")
            grid = torch.stack([v, u], dim=-1)
            grid = torch.stack([grid] * x.shape[0], dim=0)
            x_ = F.grid_sample(
                x,
                grid,
                mode="nearest",
                align_corners=False,
            )
            lr = self.upsampler(x_.detach())

            coords = []
            for (imin, imax, _), (omin, omax, ostep) in zip(
                self.input_scope, self.output_scope
            ):
                v = torch.arange(omin/2 + ostep / 2, omax/2, ostep)
                v = (v - imin) / (imax - imin) * 2 - 1
                coords.append(v.to(x.device))

            u, v = torch.meshgrid(coords, indexing="ij")
            grid = torch.stack([v, u], dim=-1)
            grid = torch.stack([grid] * x.shape[0], dim=0)

            x = F.grid_sample(
                x,
                grid,
                mode="nearest",
                align_corners=False,
            )
            hr = self.upsampler2(x.detach())

            return hr, lr


@HEADS.register_module()
class DoubleBellTNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_transform: Dict[str, Any],
        classes: List[str],
        loss: str,
        num_sa: int,
        num_heads: int,
        finetune: bool,
        inference: bool = False,
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
        grid_transform["inference"] = inference
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

        self.classifier_hr2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1),
        )

        self.inference = inference

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]

        if self.inference:
            sr, lr = self.transform(x)
            B, C, H, W = sr.shape

            # Nearby region
            sr = torch.sigmoid(self.classifier_hr2(sr))

            # Distant region
            lr = torch.sigmoid(self.classifier_hr(lr))
            lr = F.interpolate(lr, size=(2*H, 2*W), mode="nearest")
            lr[:, :, 2*H//4:3*2*H//4, 2*H//4:3*2*H//4] = sr
            return lr

        if self.finetune:
            x = self.transform(x)
            x = self.classifier_hr2(x)

        else:
            x = self.transform(x)
            x = self.classifier_hr(x)

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

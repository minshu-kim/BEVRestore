from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .edsr import make_edsr_baseline

__all__ = ["BEVSR"]

class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        scale_factor: int = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.scale_factor = scale_factor

        self.upsampler = make_edsr_baseline(
            n_feats=512,
            scale=scale_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        x = self.upsampler(x)

        return x


class BEVSR(nn.Module):
    def __init__(
        self,
        grid_transform: Dict[str, Any],
    ):
        super().__init__()
        self.transform = BEVGridTransform(**grid_transform)

    def forward(self, x: torch.Tensor):
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = self.transform(x)

        return x

import torch
import torch.nn as nn

from .pointnet import CPN


class TripletNet(nn.Module):
    """
    Wrappea el backbone para tripletas (anchor, positive, negative).
    """

    def __init__(self, width: int = 64, dropout: float = 0.5, init_to_identity: bool = True):
        super().__init__()
        self.backbone = CPN(width=width, dropout=dropout, init_to_identity=init_to_identity)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        za = self.backbone(anchor)
        zp = self.backbone(positive)
        zn = self.backbone(negative)
        return za, zp, zn

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def triplet_loss_squared(z_a: torch.Tensor, z_p: torch.Tensor, z_n: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    d_ap = ((z_a - z_p) ** 2).sum(dim=1)
    d_an = ((z_a - z_n) ** 2).sum(dim=1)
    return torch.relu(d_ap - d_an + margin).mean()

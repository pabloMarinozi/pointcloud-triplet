import torch
import torch.nn as nn


def conv1d_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class InputTNet(nn.Module):
    """
    Aprende una matriz 3x3 para alinear coordenadas XYZ.
    Espera x con forma (B, 3, N).
    """

    def __init__(self, width: int = 64, init_to_identity: bool = True, k: int = 3):
        super().__init__()
        self.k = k
        c1, c2, c3 = width, 2 * width, 32 * width
        fc_dim = max(16, c3 // 8)

        self.conv1 = conv1d_block(self.k, c1)
        self.conv2 = conv1d_block(c1, c2)
        self.conv3 = conv1d_block(c2, c3)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(c3, fc_dim, bias=False),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(fc_dim, self.k * self.k, bias=True)

        if init_to_identity:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool(out).squeeze(-1)
        out = self.fc1(out)
        theta = self.fc2(out)
        A = theta.view(-1, self.k, self.k)

        I = torch.eye(self.k, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)
        return A + I

    @staticmethod
    def apply_transform(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A, x)


class FeatureTransformationNet(nn.Module):
    """
    Aprende una matriz kxk para alinear features intermedias.
    Espera x con forma (B, k, N). En este proyecto k = width.
    """

    def __init__(self, width: int = 64, k: int | None = None, init_to_identity: bool = True):
        super().__init__()
        self.k = width  # mantengo exactamente lo que hacía el Colab

        c1, c2, c3 = width, 2 * width, 32 * width
        fc_dim = max(16, c3 // 8)

        self.conv1 = conv1d_block(self.k, c1)
        self.conv2 = conv1d_block(c1, c2)
        self.conv3 = conv1d_block(c2, c3)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(c3, fc_dim, bias=False),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(fc_dim, self.k * self.k, bias=True)

        if init_to_identity:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool(out).squeeze(-1)
        out = self.fc1(out)
        theta = self.fc2(out)
        A = theta.view(-1, self.k, self.k)

        I = torch.eye(self.k, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)
        return A + I

    @staticmethod
    def apply_transform(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A, x)


class CPN(nn.Module):
    """
    Backbone estilo PointNet para producir un embedding por nube.

    input:  x (B, 3, N)
    output: z (B, emb_dim) con emb_dim = 64 * width
    """

    def __init__(self, width: int = 64, dropout: float = 0.5, init_to_identity: bool = True):
        super().__init__()
        self.emb_dim = 64 * width

        self.itn = InputTNet(width=width, init_to_identity=init_to_identity)

        self.fw1_conv1 = conv1d_block(3, width)
        self.fw1_conv2 = conv1d_block(width, width)

        self.ftn = FeatureTransformationNet(width=width, k=width, init_to_identity=init_to_identity)

        self.fw2_conv1 = conv1d_block(width, width)
        self.fw2_conv2 = conv1d_block(width, 2 * width)
        self.fw2_conv3 = conv1d_block(2 * width, 32 * width)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(32 * width, 8 * width),
            nn.BatchNorm1d(8 * width),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(8 * width, 4 * width),
            nn.BatchNorm1d(4 * width),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4 * width, self.emb_dim),
            nn.Sigmoid(),  # mantengo igual al Colab
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A3 = self.itn(x)
        x = self.itn.apply_transform(A3, x)

        x = self.fw1_conv1(x)
        x = self.fw1_conv2(x)

        A_k = self.ftn(x)
        x = self.ftn.apply_transform(A_k, x)

        x = self.fw2_conv1(x)
        x = self.fw2_conv2(x)
        x = self.fw2_conv3(x)

        x = self.pool(x).squeeze(-1)
        z = self.fc(x)
        return z

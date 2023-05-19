import torch
from torch import nn


def gen_pos_encoding(width: int, height: int) -> torch.Tensor:
    """
    Generates a simple 2D positional encoding from -1 to 1.
    The resulting shape is 1xWxH.
    """
    width_pos = torch.arange(0, width).repeat(height, 1).T / (width / 2.0) - 1.0
    height_pos = torch.arange(0, height).repeat(width, 1) / (height / 2.0) - 1.0
    pos = torch.stack([width_pos, height_pos]).unsqueeze(0)
    return pos


class GatingLayer(nn.Module):
    def __init__(self, elements, bias):
        super().__init__()
        self.bias = bias

        self.wr = nn.Linear(elements, elements)
        self.ur = nn.Linear(elements, elements)
        self.wz = nn.Linear(elements, elements)
        self.uz = nn.Linear(elements, elements)
        self.wg = nn.Linear(elements, elements)
        self.ug = nn.Linear(elements, elements)

    def forward(self, x, y):
        r = torch.sigmoid(self.wr(y) + self.ur(x))
        z = torch.sigmoid(self.wz(y) + self.uz(x) - self.bias)
        h = torch.tanh(self.wg(y) + self.ug(r * x))
        output = (1 - z) * x + z * h
        return output


class AttnBlock(nn.Module):
    """
    This is a special self attention block from the GTrXL paper.

    It's substantially more expressive, able to learn an identity function
    through the use of gating layers if required.
    """

    def __init__(self, emb_dim: int, input_size: int, num_heads: int, gate_bias=2):
        nn.Module.__init__(self)
        self.gate1 = GatingLayer(emb_dim, gate_bias)
        self.gate2 = GatingLayer(emb_dim, gate_bias)

        self.attention = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)

        self.norm1 = nn.BatchNorm1d(input_size)
        self.norm2 = nn.BatchNorm1d(input_size)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), nn.ReLU(), nn.Linear(emb_dim * 4, emb_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized)
        attended = self.relu(attended)
        x = self.gate1(x, attended)
        normalized = self.norm2(x)
        output = self.ff(normalized)
        output = self.relu(output)
        x = self.gate2(x, output)
        return x

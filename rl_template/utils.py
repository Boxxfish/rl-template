import importlib
from typing import Tuple

import torch
from torch import nn


def get_img_size(old_h: int, old_w: int, conv: torch.nn.Conv2d) -> Tuple[int, int]:
    """
    Returns the size of the image after the convolution is run on it.
    """
    new_h = (
        old_h
        + 2 * int(conv.padding[0])
        - conv.dilation[0] * (conv.kernel_size[0] - 1)
        - 1
    ) // conv.stride[0] + 1
    new_w = (
        old_w
        + 2 * int(conv.padding[1])
        - conv.dilation[1] * (conv.kernel_size[1] - 1)
        - 1
    ) // conv.stride[1] + 1
    return new_h, new_w


def copy_params(src: nn.Module, dest: nn.Module):
    """
    Copies params from one model to another.
    """
    with torch.no_grad():
        for dest_, src_ in zip(dest.parameters(), src.parameters()):
            dest_.data.copy_(src_.data)


def init_orthogonal(src: nn.Module):
    """
    Initializes model weights orthogonally. This has been shown to greatly
    improve training efficiency.
    """
    with torch.no_grad():
        for param in src.parameters():
            if len(param.size()) >= 2:
                param.copy_(torch.nn.init.orthogonal_(param.data))


def init_xavier(src: nn.Module):
    """
    Initializes model weights using the Xavier normal distribution.
    This has been shown to have good results for transformers.
    """
    with torch.no_grad():
        for param in src.parameters():
            if len(param.size()) >= 2:
                param.copy_(torch.nn.init.xavier_normal_(param.data))

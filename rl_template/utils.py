import importlib
from typing import Tuple
from pydantic import BaseModel

import torch
from torch import nn

T = TypeVar("T", bound=BaseModel)


def parse_args(cfg_t: type[T]) -> T:
    parser = ArgumentParser()
    for k, v in cfg_t.model_fields.items():
        flag_name = f"--{k.replace('_', '-')}"
        if v.annotation == bool:
            parser.add_argument(flag_name, default=v.default, action="store_true")
        elif get_origin(v.annotation) == Literal:
            choices = list(get_args(v.annotation))
            parser.add_argument(
                flag_name, default=v.default, choices=choices, type=type(choices[0])
            )
        else:
            assert v.annotation is not None
            parser.add_argument(flag_name, default=v.default, type=v.annotation)
    args = parser.parse_args()
    cfg = cfg_t(**args.__dict__)
    return cfg


def create_directory(out_dir_: str, meta: T) -> Path:
    for _ in range(100):
        if wandb.run.name not in ["" or None]:
            break
    if wandb.run.name not in ["" or None]:
        out_id = wandb.run.name
    else:
        out_id = "testing"

    out_dir = Path(out_dir_)
    exp_dir = out_dir / out_id
    try:
        os.mkdir(exp_dir)
    except OSError as e:
        print(e)
    with open(exp_dir / "meta.json", "w") as f:
        f.write(meta.model_dump_json(indent=2))

    chkpt_path = exp_dir / "checkpoints"
    try:
        os.mkdir(chkpt_path)
    except OSError as e:
        print(e)
    return chkpt_path

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


def polyak_avg(src: nn.Module, dest: nn.Module, p: float):
    """
    Smoothly copies params from one model to another.
    At `p` = 1, `src` overwrites `dest`, at `p` = 0, nothing happens.
    """
    for dest_, src_ in zip(dest.parameters(), src.parameters()):
        dest_.data.copy_(p * src_.data + (1.0 - p) * dest_.data)

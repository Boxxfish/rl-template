import torch
from torch import nn

from .replay_buffer import ReplayBuffer

INF = 10e8


def distrs_to_means(
    distrs: torch.Tensor,  # Shape: (batch_size, num_actions, num_supports)
    v_min: float,
    v_max: float,
) -> torch.Tensor:  # Shape: (batch_size, num_actions)
    """
    Converts action distributions to means.
    """
    num_supports = distrs.shape[-1]
    support_vals = (
        (v_min + torch.arange(num_supports, device=distrs.device) * v_max).unsqueeze(0).unsqueeze(0)
    )  # Shape: (1, 1, num_supports)
    return (distrs * support_vals).sum(-1)


def train_c51(
    q_net: nn.Module,
    q_net_target: nn.Module,
    q_opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
    v_min: float,
    v_max: float,
    num_supports: int,
) -> float:
    """
    Performs the C51 training loop.
    Returns the total Q loss.
    """
    total_q_loss = 0.0
    q_net.train()
    if device.type != "cpu":
        q_net.to(device)

    total_q_loss = 0.0
    d_z = (v_max - v_min) / num_supports
    support_vals = (
        v_min + torch.arange(num_supports, device=device) * d_z
    )  # Shape: (num_supports)
    for _ in range(train_iters):
        prev_states, states, actions, rewards, dones, _, next_masks = buffer.sample(
            train_batch_size
        )

        # Move batch to device if applicable
        prev_states = prev_states.to(device=device)
        states = states.to(device=device)
        actions = actions.to(device=device)
        rewards = rewards.to(device=device)
        dones = dones.to(device=device)
        next_masks = next_masks.to(device=device)

        # Train q network
        q_opt.zero_grad()
        with torch.no_grad():
            next_distrs_all = q_net_target(
                states
            ).detach()  # Shape: (batch_size, num_actions, num_supports)
            next_means = distrs_to_means(
                next_distrs_all, v_min, v_max
            )  # Shape: (batch_size, num_actions)
            next_actions = next_means.argmax(-1) # Shape: (batch_size)
            next_distrs = next_distrs_all[torch.arange(train_batch_size), next_actions] # Shape: (batch_size, num_supports)
            q_target = torch.zeros(next_distrs.shape, device=device) # Shape: (batch_size, num_supports)
            for j in range(num_supports):
                t_z_j = torch.clip(rewards + (1 - dones) * discount * support_vals[j], v_min, v_max - 1) # Shape: (batch_size)
                b_j = (t_z_j - v_min) / d_z # Shape: (batch_size)
                l = b_j.floor().int()
                u = b_j.ceil().int()
                q_target[torch.arange(train_batch_size), l] += next_distrs[torch.arange(train_batch_size), j] * (u - b_j)
                q_target[torch.arange(train_batch_size), u] += next_distrs[torch.arange(train_batch_size), j] * (b_j - l)
            q_target = q_target.detach()
        q_loss = (-(q_target * q_net(prev_states)[torch.arange(train_batch_size, device=device), actions].log()).sum(-1)).mean()
        q_loss.backward()
        q_opt.step()
        total_q_loss += q_loss.item()

    if device.type != "cpu":
        q_net.cpu()
    q_net.eval()
    return total_q_loss

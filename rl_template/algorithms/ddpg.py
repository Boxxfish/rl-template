from typing import Tuple

import torch
from torch import nn

from rl_template.utils import polyak_avg

from .replay_buffer import ReplayBuffer

INF = 10e8


def train_ddpg(
    q_net: nn.Module,
    q_net_target: nn.Module,
    q_opt: torch.optim.Optimizer,
    p_net: nn.Module,
    p_net_target: nn.Module,
    p_opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
    polyak: float,
) -> Tuple[float, float]:
    """
    Performs the DDPG training loop.
    Returns the total Q loss and policy loss.
    """
    q_net.train()
    p_net.train()
    if device.type != "cpu":
        q_net.to(device)
        p_net.to(device)

    total_q_loss = 0.0
    total_p_loss = 0.0
    for _ in range(train_iters):
        prev_states, states, actions, rewards, dones, _, _ = buffer.sample(
            train_batch_size
        )

        # Move batch to device if applicable
        prev_states = prev_states.to(device=device)
        states = states.to(device=device)
        actions = actions.to(device=device)
        rewards = rewards.to(device=device)
        dones = dones.to(device=device)

        # Perform gradient descent on Q network, reducing prediction error
        q_opt.zero_grad()
        targets = (
            rewards
            + discount
            * (1.0 - dones)
            * q_net_target(states, p_net_target(states).detach()).squeeze()
        ).detach()
        q_loss = torch.mean(
            (q_net(prev_states, actions).squeeze() - targets) ** 2, 0
        ).squeeze(0)
        q_loss.backward()
        q_opt.step()
        total_q_loss += q_loss.item()

        # Perform gradient ascent on policy network, increasing action return
        p_opt.zero_grad()
        p_loss = -torch.mean(q_net(prev_states, p_net(prev_states)), 0).squeeze(0)
        p_loss.backward()
        p_opt.step()
        total_p_loss += p_loss.item()

        # Update target networks
        polyak_avg(q_net, q_net_target, polyak)
        polyak_avg(p_net, p_net_target, polyak)

    if device.type != "cpu":
        q_net.cpu()
        p_net.cpu()
    q_net.eval()
    p_net.eval()
    return total_q_loss, total_p_loss

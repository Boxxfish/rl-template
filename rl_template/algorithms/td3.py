from typing import Tuple

import torch
from torch import nn

from rl_template.utils import polyak_avg

from .replay_buffer import ReplayBuffer

INF = 10e8


def train_td3(
    q_net_1: nn.Module,
    q_net_1_target: nn.Module,
    q_1_opt: torch.optim.Optimizer,
    q_net_2: nn.Module,
    q_net_2_target: nn.Module,
    q_2_opt: torch.optim.Optimizer,
    p_net: nn.Module,
    p_net_target: nn.Module,
    p_opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
    polyak: float,
    noise_clip: float,
    noise_scale: float,
    act_low: torch.Tensor,
    act_high: torch.Tensor,
    update_policy_every: int = 2,
) -> Tuple[float, float]:
    """
    Performs the TD3 training loop.
    Returns the total Q loss and policy loss.
    """
    q_net_1.train()
    q_net_2.train()
    p_net.train()
    if device.type != "cpu":
        q_net_1.to(device)
        q_net_2.to(device)
        p_net.to(device)
        act_low = act_low.to(device)
        act_high = act_high.to(device)

    total_q_loss = 0.0
    total_p_loss = 0.0
    for step in range(train_iters):
        prev_states, states, actions, rewards, dones, _, _ = buffer.sample(
            train_batch_size
        )

        # Move batch to device if applicable
        prev_states = prev_states.to(device=device)
        states = states.to(device=device)
        actions = actions.to(device=device)
        rewards = rewards.to(device=device)
        dones = dones.to(device=device)

        # Perform gradient descent on Q networks, reducing prediction error
        q_1_opt.zero_grad()
        q_2_opt.zero_grad()
        eps = torch.distributions.Normal(0, noise_scale).sample(torch.Size([train_batch_size, act_low.shape[0]])).to(device)
        target_acts = torch.clip(p_net_target(states).detach() + torch.clip(eps, -noise_clip, noise_clip), act_low, act_high)
        q_targets = torch.min(q_net_1_target(states, target_acts).squeeze(), q_net_2_target(states, target_acts).squeeze()).detach()
        targets = (
            rewards
            + discount
            * (1.0 - dones)
            * q_targets
        )
        q_1_loss = torch.mean(
            (q_net_1(prev_states, actions).squeeze() - targets) ** 2, 0
        ).squeeze(0)
        q_1_loss.backward()
        q_1_opt.step()
        total_q_loss += q_1_loss.item()

        q_2_loss = torch.mean(
            (q_net_2(prev_states, actions).squeeze() - targets) ** 2, 0
        ).squeeze(0)
        q_2_loss.backward()
        q_2_opt.step()
        total_q_loss += q_2_loss.item()

        if (step + 1) % update_policy_every == 0:
            # Perform gradient ascent on policy network, increasing action return
            p_opt.zero_grad()
            p_loss = -torch.mean(q_net_1(prev_states, p_net(prev_states)), 0).squeeze(0)
            p_loss.backward()
            p_opt.step()
            total_p_loss += p_loss.item()

            # Update P network
            polyak_avg(p_net, p_net_target, polyak)

        # Update Q networks
        polyak_avg(q_net_1, q_net_1_target, polyak)
        polyak_avg(q_net_2, q_net_2_target, polyak)

    if device.type != "cpu":
        q_net_1.cpu()
        q_net_2.cpu()
        p_net.cpu()
    q_net_1.eval()
    q_net_2.eval()
    p_net.eval()
    return total_q_loss, total_p_loss

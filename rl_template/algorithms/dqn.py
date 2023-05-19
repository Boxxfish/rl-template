from typing import Tuple
import torch
from torch import nn

from .replay_buffer import ReplayBuffer

INF = 10e8


def train_ppo(
    q_net: nn.Module,
    q_net_target: nn.Module,
    q_opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
) -> Tuple[float, float]:
    """
    Performs the DQN training loop.
    Returns the total Q loss.
    """
    total_q_loss = 0.0
    if buffer.filled:
        q_net.train()
        if device.type != "cpu":
            q_net.to(device)

        total_q_loss = 0.0
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
                next_actions = (
                    torch.where(next_masks == 1, -INF, q_net(states))
                    .argmax(1)
                    .squeeze(0)
                )
                q_target = rewards.unsqueeze(1) + discount * q_net_target(
                    states
                ).detach().gather(1, next_actions.unsqueeze(1)) * (
                    1.0 - dones.unsqueeze(1)
                )
            diff = q_net(prev_states).gather(1, actions.unsqueeze(1)) - q_target
            q_loss = (diff * diff).mean()
            q_loss.backward()
            q_opt.step()
            total_q_loss += q_loss.item()

        if device.type != "cpu":
            q_net.cpu()
        q_net.eval()

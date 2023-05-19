import copy
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from .rollout_buffer import RolloutBuffer


def train_ppo(
    p_net: nn.Module,
    v_net: nn.Module,
    p_opt: torch.optim.Optimizer,
    v_opt: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
    lambda_: float,
    epsilon: float,
    gradient_steps: int = 1,
    use_masks: bool = False,
) -> Tuple[float, float]:
    """
    Performs the PPO training loop. Returns a tuple of total policy loss and
    total value loss.

    Args:
        gradient_steps: Number of batches to step through before before
        adjusting weights.
        use_masks: If True, masks are passed to the model.
    """
    p_net.train()
    v_net_frozen = copy.deepcopy(v_net)
    v_net.train()
    if device.type != "cpu":
        p_net.to(device)
        v_net.to(device)

    total_v_loss = 0.0
    total_p_loss = 0.0

    p_opt.zero_grad()
    v_opt.zero_grad()

    for _ in tqdm(range(train_iters), position=1):
        batches = buffer.samples(train_batch_size, discount, lambda_, v_net_frozen)
        for (
            i,
            (prev_states, actions, action_probs, returns, advantages, action_masks),
        ) in enumerate(batches):
            # Move batch to device if applicable
            prev_states = prev_states.to(device=device)
            actions = actions.to(device=device)
            action_probs = action_probs.to(device=device)
            returns = returns.to(device=device)
            advantages = advantages.to(device=device)
            action_masks = action_masks.to(device=device)

            # Train policy network
            with torch.no_grad():
                old_act_probs = Categorical(logits=action_probs).log_prob(
                    actions.squeeze()
                )
            if use_masks:
                new_log_probs = p_net(prev_states, action_masks)
            else:
                new_log_probs = p_net(prev_states)
            new_act_probs = Categorical(logits=new_log_probs).log_prob(
                actions.squeeze()
            )
            term1 = (new_act_probs - old_act_probs).exp() * advantages
            term2 = (1.0 + epsilon * advantages.sign()) * advantages
            p_loss = -term1.min(term2).mean() / gradient_steps
            p_loss.backward()
            total_p_loss += p_loss.item()

            # Train value network
            diff = v_net(prev_states) - returns
            v_loss = (diff * diff).mean() / gradient_steps
            v_loss.backward()
            total_v_loss += v_loss.item()

        if (i + 1) % gradient_steps == 0:
            p_opt.step()
            v_opt.step()
            p_opt.zero_grad()
            v_opt.zero_grad()

    if device.type != "cpu":
        p_net.cpu()
        v_net.cpu()
    p_net.eval()
    v_net.eval()

    return (total_p_loss, total_v_loss)

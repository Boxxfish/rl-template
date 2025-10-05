import random
import torch
from torch import nn, Tensor


def train_alpha_zero(
    policy_value_net: nn.Module,
    opt: torch.optim.Optimizer,
    buffer: list[tuple[Tensor, Tensor, float]],
    device: torch.device,
) -> tuple[float, float]:
    # Construct batch
    random.shuffle(buffer)
    states_, action_probs_, mean_action_values_ = zip(*buffer)
    states = torch.stack(states_)  # Shape: (batch_size, ...obs_dims)
    action_probs = torch.stack(action_probs_)  # Shape: (batch_size, num_actions)
    mean_action_values = torch.tensor(mean_action_values_).unsqueeze(
        1
    )  # Shape: (batch_size, 1)

    # Train on batch
    opt.zero_grad()
    (
        pred_action_logits,  # Shape: (batch_size, num_actions)
        pred_mean_action_values,  # Shape: (batch_size, 1)
    ) = policy_value_net.forward(states)
    prob_ce = -(action_probs * torch.log_softmax(pred_action_logits, -1)).mean()
    val_mse: Tensor = ((mean_action_values - pred_mean_action_values) ** 2).mean()
    loss = prob_ce + val_mse
    loss.backward()
    opt.step()

    return prob_ce.item(), val_mse.item()

import random
import torch
from torch import nn, Tensor


def train_alpha_zero(
    policy_value_net: nn.Module,
    opt: torch.optim.Optimizer,
    buffer: list[tuple[Tensor, Tensor, float]],
    train_batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    if device.type != "cpu":
        policy_value_net.to(device=device)
    policy_value_net.train()

    # Construct batches
    random.shuffle(buffer)
    states_, action_probs_, mean_action_values_ = zip(*buffer)
    states = torch.stack(states_)  # Shape: (buffer_size, ...obs_dims)
    action_probs = torch.stack(action_probs_)  # Shape: (buffer_size, num_actions)
    mean_action_values = torch.tensor(mean_action_values_).unsqueeze(
        1
    )  # Shape: (buffer_size, 1)

    # Train on batches
    num_batches = len(buffer) // train_batch_size
    total_ce_loss, total_mse_loss = 0.0, 0.0
    for i in range(num_batches):
        start_idx = i * train_batch_size
        end_idx = (i + 1) * train_batch_size
        batch_states = states[start_idx:end_idx].to(device=device)
        batch_action_probs = action_probs[start_idx:end_idx].to(device=device)
        batch_mean_action_values = mean_action_values[start_idx:end_idx].to(device=device)
        assert len(batch_mean_action_values.shape) == 2
        opt.zero_grad()
        (
            pred_action_logits,  # Shape: (batch_size, num_actions)
            pred_mean_action_values,  # Shape: (batch_size, 1)
        ) = policy_value_net.forward(batch_states)
        prob_ce = -(batch_action_probs * torch.log_softmax(pred_action_logits, -1)).mean()
        val_mse: Tensor = ((batch_mean_action_values - pred_mean_action_values) ** 2).mean()
        loss = prob_ce + val_mse
        loss.backward()
        opt.step()

        total_ce_loss += prob_ce.cpu().item()
        total_mse_loss += val_mse.cpu().item()

    if device.type != "cpu":
        policy_value_net.to(device="cpu")
    policy_value_net.eval()

    return total_ce_loss / num_batches, total_mse_loss / num_batches

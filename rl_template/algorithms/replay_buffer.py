"""
A replay buffer for use with off policy algorithms.
"""
from typing import List, Optional, Tuple

import torch


class ReplayBuffer:
    """
    Stores transitions and generates mini batches.
    """

    def __init__(
        self,
        state_shape: torch.Size,
        action_masks_shape: torch.Size,
        capacity: int,
    ):
        k = torch.float
        state_shape = torch.Size([capacity] + list(state_shape))
        action_shape = torch.Size([capacity])
        action_masks_shape = torch.Size([capacity] + list(action_masks_shape))
        self.capacity = capacity
        self.next = 0
        d = torch.device("cpu")
        self.states = torch.zeros(state_shape, dtype=k, device=d, requires_grad=False)
        self.next_states = torch.zeros(
            state_shape, dtype=k, device=d, requires_grad=False
        )
        self.actions = torch.zeros(
            action_shape, dtype=torch.int64, device=d, requires_grad=False
        )
        self.rewards = torch.zeros([capacity], dtype=k, device=d, requires_grad=False)
        # Technically this is the "terminated" flag
        self.dones = torch.zeros([capacity], dtype=k, device=d, requires_grad=False)
        self.filled = False
        self.masks = torch.zeros(
            action_masks_shape, dtype=torch.int, device=d, requires_grad=False
        )
        self.next_masks = torch.zeros(
            action_masks_shape, dtype=torch.int, device=d, requires_grad=False
        )

    def insert_step(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: List[float],
        dones: List[bool],
        masks: Optional[torch.Tensor],
        next_masks: Optional[torch.Tensor],
    ):
        """
        Inserts a transition from each environment into the buffer. Make sure
        more data than steps aren't inserted.
        """
        batch_size = len(dones)
        d = torch.device("cpu")
        with torch.no_grad():
            indices = torch.arange(
                self.next,
                (self.next + batch_size),
            ).remainder(self.capacity)
            self.states.index_copy_(0, indices, states)
            self.next_states.index_copy_(0, indices, next_states)
            self.actions.index_copy_(0, indices, actions)
            self.rewards.index_copy_(
                0, indices, torch.tensor(rewards, dtype=torch.float, device=d)
            )
            self.dones.index_copy_(
                0, indices, torch.tensor(dones, dtype=torch.float, device=d)
            )
            if masks is not None:
                self.masks.index_copy_(0, indices, masks.to(torch.int))
            if next_masks is not None:
                self.next_masks.index_copy_(0, indices, next_masks.to(torch.int))
        self.next = (self.next + batch_size) % self.capacity
        if self.next == 0:
            self.filled = True

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generates minibatches of experience.
        """
        with torch.no_grad():
            indices = torch.randint(
                self.capacity,
                [batch_size],
                dtype=torch.int,
            )
            rand_states = self.states.index_select(0, indices)
            rand_next_states = self.next_states.index_select(0, indices)
            rand_actions = self.actions.index_select(0, indices)
            rand_rewards = self.rewards.index_select(0, indices)
            rand_dones = self.dones.index_select(0, indices)
            rand_masks = self.masks.index_select(0, indices)
            rand_next_masks = self.next_masks.index_select(0, indices)
            return (
                rand_states,
                rand_next_states,
                rand_actions,
                rand_rewards,
                rand_dones,
                rand_masks,
                rand_next_masks,
            )

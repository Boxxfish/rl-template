"""
Experiment for checking that AlphaZero works.

AlphaZero is a search-based RL algorithm that uses Monte Carlo Tree Search (MCTS) rollouts during both training and
inference to refine action values.
"""
import copy
import random
from functools import reduce
from typing import Any

import envpool  # type: ignore
import numpy as np  # type: ignore
import torch
import torch.nn as nn
import wandb
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from tqdm import tqdm
from pydantic import BaseModel

from rl_template.algorithms.dqn import train_dqn
from rl_template.algorithms.replay_buffer import ReplayBuffer
from rl_template.utils import init_orthogonal, parse_args

_: Any
INF = 10**8

class Args(BaseModel):
    train_steps: int = 512  # Number of steps to step through during sampling.
    iterations: int = 1_000  # Number of sample/train iterations.
    train_iters: int = 1  # Number of passes over the samples collected.
    train_batch_size: int = 512  # Minibatch size while training models.
    discount: float = 1.0  # Discount factor applied to rewards.
    eval_iters: int = 8  # Number of eval runs to average over.
    max_eval_steps: int = 300  # Max number of steps to take during each eval run.
    lr: float = 0.0001  # Learning rate.
    puct_train: float = 1.0 # PUCT constant during training. When 0, traversal becomes greedy.
    puct_eval: float = 1.0 # PUCT constant during eval. When 0, traversal becomes greedy.
    num_rollouts_train: int = 10 # Number of rollouts per training step.
    num_rollouts_eval: int = 50 # Number of rollouts per eval step.
    device: str = "cuda"

class PolicyValueNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        action_count: int,
    ):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.net = nn.Sequential(
            nn.Linear(flat_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.advantage = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_count)
        )
        self.value = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.action_count = action_count
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.net(input)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

def main():
    args = parse_args(Args)
    device = torch.device(args.device)

    config_dict = {"experiment": "alpha_zero"}
    config_dict.update(args.model_dump())
    wandb.init(
        project="tests",
        config=config_dict,
    )


    env = envpool.make("CartPole-v1", "gym", num_envs=num_envs)
    test_env = CartPoleEnv()

    # Initialize Q network
    obs_space = env.observation_space
    act_space = env.action_space
    q_net = QNet(obs_space.shape, int(act_space.n))
    q_net_target = copy.deepcopy(q_net)
    q_net_target.to(device)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=q_lr)

    # A replay buffer stores experience collected over all sampling runs
    buffer = ReplayBuffer(
        torch.Size(obs_space.shape),
        torch.Size((int(act_space.n),)),
        buffer_size,
    )

    obs = torch.Tensor(env.reset()[0])
    done = False
    for step in tqdm(range(iterations), position=0):
        percent_done = step / iterations

        # Collect experience
        with torch.no_grad():
            for _ in range(train_steps):
                if (
                    random.random() < q_epsilon * max(1.0 - percent_done, 0.05)
                    or step < warmup_steps
                ):
                    actions_list = [
                        random.randrange(0, int(act_space.n)) for _ in range(num_envs)
                    ]
                    actions_ = np.array(actions_list)
                else:
                    q_vals = q_net(obs)
                    actions_ = q_vals.argmax(1).numpy()
                obs_, rewards, dones, truncs, _ = env.step(actions_)
                next_obs = torch.from_numpy(obs_)
                buffer.insert_step(
                    obs,
                    next_obs,
                    torch.from_numpy(actions_).squeeze(0),
                    rewards,
                    dones,
                    None,
                    None,
                )
                obs = next_obs

        # Train
        if buffer.filled:
            total_q_loss = train_dqn(
                q_net,
                q_net_target,
                q_opt,
                buffer,
                device,
                train_iters,
                train_batch_size,
                discount,
            )

            # Evaluate the network's performance after this training iteration.
            eval_done = False
            with torch.no_grad():
                reward_total = 0
                pred_reward_total = 0
                obs_, info = test_env.reset()
                eval_obs = torch.from_numpy(np.array(obs_)).float()
                for _ in range(eval_iters):
                    steps_taken = 0
                    score = 0
                    for _ in range(max_eval_steps):
                        q_vals = q_net(eval_obs.unsqueeze(0)).squeeze()
                        action = q_vals.argmax(0).item()
                        pred_reward_total += (
                            q_net(eval_obs.unsqueeze(0)).squeeze().max(0).values.item()
                        )
                        obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
                        eval_obs = torch.from_numpy(np.array(obs_)).float()
                        steps_taken += 1
                        reward_total += reward
                        if eval_done or eval_trunc:
                            obs_, info = test_env.reset()
                            eval_obs = torch.from_numpy(np.array(obs_)).float()
                            break

            wandb.log(
                {
                    "avg_eval_episode_reward": reward_total / eval_iters,
                    "avg_eval_episode_predicted_reward": pred_reward_total / eval_iters,
                    "avg_q_loss": total_q_loss / train_iters,
                    "q_lr": q_opt.param_groups[-1]["lr"],
                }
            )

if __name__ == "__main__":
    main()
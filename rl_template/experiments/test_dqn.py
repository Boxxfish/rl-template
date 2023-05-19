"""
Experiment for checking that DQN works.

The Deep Q Network (DQN) algorithm is a popular offline deep reinforcement
learning algorithm. It's intuitive to understand, and it gets reliable results,
though it can take longer to run.
"""
import copy
from functools import reduce
import random
from typing import Any

import envpool  # type: ignore
import numpy as np  # type: ignore
import torch
import torch.nn as nn

# import wandb
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.spaces import Discrete
from torch.distributions import Categorical
from tqdm import tqdm
from rl_template.algorithms.dqn import train_dqn

from rl_template.algorithms.replay_buffer import ReplayBuffer
from rl_template.utils import init_orthogonal

_: Any
INF = 10**8

# Hyperparameters
num_envs = 32  # Number of environments to step through at once during sampling.
train_steps = 4  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 100000  # Number of sample/train iterations.
train_iters = 1  # Number of passes over the samples collected.
train_batch_size = 512  # Minibatch size while training models.
discount = 0.999  # Discount factor applied to rewards.
q_epsilon = 0.1  # Epsilon for epsilon greedy strategy. This gets annealed over time.
eval_steps = 8  # Number of eval runs to average over.
max_eval_steps = 300  # Max number of steps to take during each eval run.
q_lr = 0.0001  # Learning rate of the q net.
warmup_steps = 500  # For the first n number of steps, we will only sample randomly.
buffer_size = 10000  # Number of elements that can be stored in the buffer.
target_update = 500  # Number of iterations before updating Q target.
device = torch.device("cuda")

# Uncomment for logging
"""
wandb.init(
    project="tests",
    entity="ENTITY",
    config={
        "experiment": "ppo",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "q_epsilon": q_epsilon,
        "max_eval_steps": max_eval_steps,
        "q_lr": q_lr,
    },
)
"""


# The Q network takes in an observation and returns the predicted return for
# each action.
class QNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        action_count: int,
    ):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.net = nn.Sequential(
            nn.Linear(flat_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.advantage = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_count)
        )
        self.value = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.action_count = action_count
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.net(input)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)


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
        total_q_net = train_dqn(
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
            for _ in range(eval_steps):
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

        # Uncomment for logging
        """
        wandb.log(
            {
                "avg_eval_episode_reward": reward_total / eval_steps,
                "avg_eval_episode_predicted_reward": pred_reward_total / eval_steps,
                "avg_q_loss": total_q_loss / train_iters,
                "q_lr": q_opt.param_groups[-1]["lr"],
            }
        )
        """

        # Update Q target
        if (step + 1) % target_update == 0:
            q_net_target.load_state_dict(q_net.state_dict())

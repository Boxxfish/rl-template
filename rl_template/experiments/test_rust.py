"""
Experiment for checking that Rust env works.
This should be removed if the Rust environment changes.
"""
from functools import reduce
from typing import Any, Dict, Tuple

import gymnasium
import torch
import torch.nn as nn
import wandb
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import SyncVectorEnv
from torch.distributions import Categorical
from tqdm import tqdm

from rl_template.algorithms.ppo import train_ppo
from rl_template.algorithms.rollout_buffer import RolloutBuffer
from rl_template.conf import entity
from rl_template.utils import init_orthogonal
from rl_template_rust import CartpoleEnv  # type: ignore

_: Any

# Hyperparameters
num_envs = 256  # Number of environments to step through at once during sampling.
train_steps = 128  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
iterations = 1000  # Number of sample/train iterations.
train_iters = 2  # Number of passes over the samples collected.
train_batch_size = 512  # Minibatch size while training models.
discount = 0.98  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
max_eval_steps = 500  # Number of eval runs to average over.
eval_steps = 8  # Max number of steps to take during each eval run.
v_lr = 0.01  # Learning rate of the value net.
p_lr = 0.001  # Learning rate of the policy net.
device = torch.device("cuda")  # Device to use during training.

wandb.init(
    project="tests",
    entity=entity,
    config={
        "experiment": "rust",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "lambda": lambda_,
        "epsilon": epsilon,
        "max_eval_steps": max_eval_steps,
        "v_lr": v_lr,
        "p_lr": p_lr,
    },
)


# The value network takes in an observation and returns a single value, the
# predicted return
class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.v_layer1 = nn.Linear(flat_obs_dim, 64)
        self.v_layer2 = nn.Linear(64, 64)
        self.v_layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.v_layer1(input.flatten(1))
        x = self.relu(x)
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x


# The policy network takes in an observation and returns the log probability of
# taking each action
class PolicyNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.a_layer1 = nn.Linear(flat_obs_dim, 64)
        self.a_layer2 = nn.Linear(64, 64)
        self.a_layer3 = nn.Linear(64, action_count)
        self.relu = nn.ReLU()
        self.logits = nn.LogSoftmax(1)
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.a_layer1(input.flatten(1))
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.relu(x)
        x = self.a_layer3(x)
        x = self.logits(x)
        return x


# Wraps the Rust env in a Python class.
class PyCartpoleEnv(gymnasium.Env):
    def __init__(self):
        self.observation_space = Box(-1.0, 1.0, (4,))
        self.action_space = Discrete(2)
        self.env = CartpoleEnv()

    def step(self, action: int):
        obs, reward, terminated = self.env.step(action)
        return (obs, reward, terminated, False, {})

    def reset(self):
        obs = self.env.reset()
        return (obs, {})


env = SyncVectorEnv([lambda: PyCartpoleEnv() for _ in range(num_envs)])
test_env = PyCartpoleEnv()

# Initialize policy and value networks
obs_space = env.envs[0].observation_space
obs_size = torch.Size(obs_space.shape or [])
act_space = env.envs[0].action_space
if not isinstance(act_space, Discrete):
    print("Wrong action type.")
    quit()
v_net = ValueNet(obs_size)
p_net = PolicyNet(obs_size, int(act_space.n))
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

# A rollout buffer stores experience collected during a sampling run
buffer = RolloutBuffer(
    obs_size,
    torch.Size((1,)),
    torch.Size((int(act_space.n),)),
    torch.int,
    num_envs,
    train_steps,
)

obs = torch.Tensor(env.reset()[0])
done = False
for _ in tqdm(range(iterations), position=0):
    # Collect experience for a number of steps and store it in the buffer
    with torch.no_grad():
        for _ in tqdm(range(train_steps), position=1):
            action_probs = p_net(obs)
            actions = Categorical(logits=action_probs).sample().numpy()
            obs_, rewards, dones, truncs, _ = env.step(actions)
            buffer.insert_step(
                obs,
                torch.from_numpy(actions).unsqueeze(-1),
                action_probs,
                list(rewards),
                list(dones),
                list(truncs),
            )
            obs = torch.from_numpy(obs_)
            if done:
                obs = torch.Tensor(env.reset()[0])
                done = False
        buffer.insert_final_step(obs)

    # Train
    total_p_loss, total_v_loss = train_ppo(
        p_net,
        v_net,
        p_opt,
        v_opt,
        buffer,
        device,
        train_iters,
        train_batch_size,
        discount,
        lambda_,
        epsilon,
    )
    buffer.clear()

    # Evaluate the network's performance after this training iteration.
    eval_obs = torch.Tensor(test_env.reset()[0])
    eval_done = False
    with torch.no_grad():
        # Visualize
        reward_total = 0.0
        entropy_total = 0.0
        eval_obs = torch.Tensor(test_env.reset()[0])
        for _ in range(eval_steps):
            avg_entropy = 0.0
            steps_taken = 0
            for _ in range(max_eval_steps):
                distr = Categorical(logits=p_net(eval_obs.unsqueeze(0)).squeeze())
                action = distr.sample().item()
                obs_, reward, eval_done, _, _ = test_env.step(action)
                eval_obs = torch.Tensor(obs_)
                steps_taken += 1
                if eval_done:
                    eval_obs = torch.Tensor(test_env.reset()[0])
                    break
                reward_total += reward
                avg_entropy += distr.entropy()
            avg_entropy /= steps_taken
            entropy_total += avg_entropy

    wandb.log(
        {
            "avg_eval_episode_reward": reward_total / eval_steps,
            "avg_eval_entropy": entropy_total / eval_steps,
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
        }
    )

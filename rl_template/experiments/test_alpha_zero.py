"""
Experiment for checking that AlphaZero works.

AlphaZero is a search-based RL algorithm that uses Monte Carlo Tree Search (MCTS) rollouts during both training and
inference to refine action values.

Note that this isn't a TRUE version of AlphaZero -- AlphaZero was designed for 2 player, sparse reward environments,
while our version uses details from MuZero to extend this to the single player domain.

This also doesn't do the automatic minimum-maximum Q value rescaling done in MuZero, so make sure your `puct_c1` and
`puct_c2` constants are set accordingly, and/or you've normalized your reward function.
"""

import copy
import random
from typing import Any
from gymnasium import Env
import numpy as np  # type: ignore
import torch
import torch.nn as nn
from torch import Tensor
import wandb
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from tqdm import tqdm
from pydantic import BaseModel

from rl_template.algorithms.alpha_zero import train_alpha_zero
from rl_template.algorithms.mcts import (
    BasePolicyValuePredictor,
    BaseSavableEnv,
    MCTSNode,
)
from rl_template.utils import parse_args


class Args(BaseModel):
    train_steps: int = 2048  # Number of steps to step through during sampling.
    train_batch_size: int = 512  # Training batch size.
    iterations: int = 1_000  # Number of sample/train iterations.
    discount: float = 0.9  # Discount factor applied to rewards.
    lr: float = 10e-4  # Learning rate.
    ucb_c1: float = 1.25  # UCB constant. Higher values increase exploration.
    ucb_c2: float = 19652.0  # UCB constant. Higher values increase exploration.
    num_searches_train: int = 100  # Number of searches per training step.
    num_searches_eval: int = 500  # Number of searches per eval step.
    train_temperature: float = 1.0  # Action probability temperature during training.
    eval_temperature: float = 0.1  # Action probability temperature during evaluation.
    eval_every: int = 10  # How many iterations to train before evaluating.
    eval_iters: int = 8  # Number of eval runs to average over.
    max_eval_steps: int = 300  # Max number of steps to take during each eval run.
    device: str = "cuda"


class SaveableCartpole(Env, BaseSavableEnv[dict[str, Any]]):

    def __init__(self, base_env: CartPoleEnv):
        self.env = base_env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self, *args, seed=None, options=None):
        return self.env.reset(*args, seed=seed, options=options)

    def save_state(self):
        state = self.env.__dict__.copy()
        state.pop("screen", None)
        state.pop("surf", None)
        state.pop("clock", None)
        state.pop("_np_random", None)
        state.pop("render_mode", None)
        state.pop("screen_width", None)
        state.pop("screen_height", None)
        state.pop("isopen", None)
        state = copy.deepcopy(state)
        return state

    def load_state(self, save_state):
        for key, val in save_state.items():
            setattr(self.env, key, val)

    def render(self):
        return self.env.render()


class PolicyValueNet(nn.Module, BasePolicyValuePredictor[np.ndarray]):
    def __init__(
        self,
        obs_shape: torch.Size,
        action_count: int,
    ):
        nn.Module.__init__(self)
        obs_dim = obs_shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.policy = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_count)
        )
        self.value = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.action_count = action_count

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(input)
        policy_out = self.policy(x)
        value_out = self.value(x)
        return policy_out, value_out

    def predict(self, state: np.ndarray):
        with torch.no_grad():
            policy_logits, value = self.forward(torch.from_numpy(state).unsqueeze(0))
        return torch.softmax(policy_logits.squeeze(0), 0).tolist(), float(
            value.squeeze(0).item()
        )


def get_entropy(probs: list[float]) -> float:
    return -(np.array(probs) * np.log(np.array(probs))).sum()


def main():
    args = parse_args(Args)
    device = torch.device(args.device)
    assert args.train_steps % args.train_batch_size == 0

    config_dict = {"experiment": "alpha_zero"}
    config_dict.update(args.model_dump())
    wandb.init(
        project="tests",
        config=config_dict,
    )

    env = SaveableCartpole(CartPoleEnv())
    search_env = SaveableCartpole(CartPoleEnv())
    test_env = SaveableCartpole(CartPoleEnv())

    # Initialize network
    obs_space = env.observation_space
    act_space = env.action_space
    policy_value_net = PolicyValueNet(obs_space.shape, int(act_space.n))
    opt = torch.optim.AdamW(policy_value_net.parameters(), lr=args.lr)

    buffer = list[
        tuple[
            Tensor,  # State
            Tensor,  # MCTS action probs
            float,  # MCTS mean action value
        ]
    ]()

    obs = env.reset()[0]
    root = MCTSNode(1.0, args.ucb_c1, args.ucb_c2, args.discount)
    for step in tqdm(range(args.iterations), position=0):
        # Collect experience
        buffer.clear()
        total_train_act_entropy = 0.0
        total_train_prior_entropy = 0.0
        total_train_pred_return = 0.0
        with torch.no_grad():
            for _ in range(args.train_steps):
                # Perform MCTS
                save_state = env.save_state()
                while root.visit_count < args.num_searches_train:
                    search_env.load_state(save_state)
                    root.expand(
                        obs,
                        search_env,
                        policy_value_net,
                    )
                action_probs = root.get_probs(args.train_temperature)
                total_train_act_entropy += get_entropy(action_probs)
                total_train_prior_entropy += get_entropy(root.get_child_priors())
                action = random.choices(list(range(len(action_probs))), action_probs)[0]
                root = root.children[action]

                # Save step data
                mean_action_value = root.mean_action_value
                total_train_pred_return += mean_action_value
                buffer.append(
                    (
                        torch.from_numpy(obs),
                        torch.tensor(action_probs),
                        mean_action_value,
                    )
                )

                # Step through env
                obs, reward, done, trunc, _ = env.step(action)
                if done or trunc:
                    obs = env.reset()[0]
                    root = MCTSNode(1.0, args.ucb_c1, args.ucb_c2, args.discount)

        # Train
        ce_loss, mse_loss = train_alpha_zero(
            policy_value_net,
            opt,
            buffer,
            args.train_batch_size,
            device,
        )

        log_dict = {
            "prob_ce_loss": ce_loss,
            "val_mse_loss": mse_loss,
            "lr": opt.param_groups[-1]["lr"],
            "avg_train_act_entropy": total_train_act_entropy / args.train_steps,
            "avg_train_prior_entropy": total_train_prior_entropy / args.train_steps,
            "avg_train_step_predicted_return": total_train_pred_return
            / args.train_steps,
        }

        # Evaluate the network's performance
        if step % args.eval_every == 0:
            with torch.no_grad():
                reward_total = 0.0
                total_eval_pred_return = 0.0
                total_eval_steps = 0
                total_eval_act_entropy = 0.0
                eval_obs = test_env.reset()[0]
                for _ in range(args.eval_iters):
                    eval_root = MCTSNode(1.0, args.ucb_c1, args.ucb_c2, args.discount)
                    for _ in range(args.max_eval_steps):
                        # Perform MCTS
                        save_state = test_env.save_state()
                        while eval_root.visit_count < args.num_searches_eval:
                            search_env.load_state(save_state)
                            eval_root.expand(
                                eval_obs,
                                search_env,
                                policy_value_net,
                            )
                        action_probs = eval_root.get_probs(args.eval_temperature)
                        total_eval_act_entropy += get_entropy(action_probs)
                        action = random.choices(
                            list(range(len(action_probs))), action_probs
                        )[0]
                        eval_root = eval_root.children[action]

                        # Step through environment
                        total_eval_pred_return += root.mean_action_value
                        eval_obs, reward, done, trunc, _ = test_env.step(action)
                        reward_total += reward
                        total_eval_steps += 1
                        if done or trunc:
                            eval_obs = test_env.reset()[0]
                            break

            log_dict.update(
                {
                    "avg_eval_episode_reward": reward_total / args.eval_iters,
                    "avg_eval_step_predicted_return": total_eval_pred_return
                    / total_eval_steps,
                    "avg_eval_act_entropy": total_eval_act_entropy / total_eval_steps,
                }
            )

        wandb.log(log_dict)


if __name__ == "__main__":
    main()

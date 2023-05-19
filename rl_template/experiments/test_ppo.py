"""
Experiment for checking that PPO is working.

Proximal Policy Optimization (PPO) is a popular deep reinforcement learning
algorithm. At OpenAI and a lot of other places, it's used as a baseline, since
you can get pretty good performance without having to fiddle with the
hyperparameters too much.
"""
from functools import reduce
from typing import Any

import envpool  # type: ignore
import torch
import torch.nn as nn
import wandb
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from torch.distributions import Categorical
from tqdm import tqdm

from ..algorithms.rollout_buffer import RolloutBuffer
from ..utils import copy_params, init_orthogonal

_: Any

# Hyperparameters
num_envs = 128
train_steps = 500
iterations = 300
train_iters = 2
train_batch_size = 512
discount = 0.98
lambda_ = 0.95
epsilon = 0.2
max_eval_steps = 500
v_lr = 0.01
p_lr = 0.001
device = torch.device("cpu")

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
        self.v_layer1 = nn.Linear(flat_obs_dim, 256)
        self.v_layer2 = nn.Linear(256, 256)
        self.v_layer3 = nn.Linear(256, 1)
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
        self.a_layer1 = nn.Linear(flat_obs_dim, 256)
        self.a_layer2 = nn.Linear(256, 256)
        self.a_layer3 = nn.Linear(256, action_count)
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


env = envpool.make("CartPole-v1", "gym", num_envs=num_envs)
test_env = CartPoleEnv()

# Initialize policy and value networks
obs_space = env.observation_space
act_space = env.action_space
v_net = ValueNet(obs_space.shape)
p_net = PolicyNet(obs_space.shape, act_space.n)
p_net_old = PolicyNet(obs_space.shape, act_space.n)
p_net_old.eval()
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

# A rollout buffer stores experience collected during a sampling run
buffer = RolloutBuffer(
    obs_space.shape,
    torch.Size((1,)),
    torch.Size((4,)),
    torch.int,
    num_envs,
    train_steps,
    device,
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
                rewards,
                dones,
                truncs,
            )
            obs = torch.from_numpy(obs_)
            if done:
                obs = torch.Tensor(env.reset()[0])
                done = False
        buffer.insert_final_step(obs)

    # Train
    p_net.train()
    v_net.train()
    copy_params(p_net, p_net_old)

    total_v_loss = 0.0
    total_p_loss = 0.0
    for _ in range(train_iters):
        # The rollout buffer provides randomized minibatches of samples
        batches = buffer.samples(train_batch_size, discount, lambda_, v_net)
        for prev_states, actions, action_probs, returns, advantages, _ in batches:
            # Train policy network.
            #
            # First, we get the log probabilities of taking the actions we took
            # when we took them. You can store this in the replay buffer right
            # after taking the action and reuse it, but here I'm just using a
            # copy of the original policy network and passing the observation to
            # get the same thing.
            with torch.no_grad():
                old_log_probs = p_net_old(prev_states)
                old_act_probs = Categorical(logits=old_log_probs).log_prob(
                    actions.squeeze()
                )
            # Next, we get the log probabilities of taking the actions with our
            # current network. During the first iteration, when we sample our
            # first minibatch, this should give us the exact same probabilities
            # as the step above, since we didn't update the network yet.
            p_opt.zero_grad()
            new_log_probs = p_net(prev_states)
            new_act_probs = Categorical(logits=new_log_probs).log_prob(
                actions.squeeze()
            )
            # Then, we run PPO's loss function, which is sometimes called the
            # surrogate loss. Written out explicitly, it's
            # min(current_probs/prev_probs * advantages,
            # clamp(current_probs/prev_probs, 1 - epsilon, 1 + epsilon) *
            # advantages). The actual code written is just a more optimized way
            # of writing that.
            #
            # Basically, we only want to update our network if the probability
            # of taking the actions with our current net is slightly less or
            # slightly more than the probability of taking the actions under the
            # old net. If that ratio is too high or too low, then the clipping
            # kicks in, and the gradient goes to 0 since we're differentiating a
            # constant (1 - epsilon or 1 + epsilon).
            #
            # Note that the only major difference is the importance sampling
            # term (the ratio) and the clipping; the loss function for A2C is
            # current_log_probs * advantages, which is very similar. Also, we're
            # using a negative loss function because we're trying to maximize
            # this instead of minimizing it.
            term1: torch.Tensor = (new_act_probs - old_act_probs).exp() * advantages
            term2: torch.Tensor = (1.0 + epsilon * advantages.sign()) * advantages
            p_loss = -term1.min(term2).mean()
            p_loss.backward()
            p_opt.step()
            total_p_loss += p_loss.item()

            # Train value network. Hopefully, this part is much easier to
            # understand.
            v_opt.zero_grad()
            diff: torch.Tensor = v_net(prev_states) - returns.unsqueeze(1)
            v_loss = (diff * diff).mean()
            v_loss.backward()
            v_opt.step()
            total_v_loss += v_loss.item()

    p_net.eval()
    v_net.eval()
    buffer.clear()

    # Evaluate the network's performance after this training iteration. The
    # reward per episode and entropy are both recorded here. Entropy is useful
    # because as the agent learns, unless there really *is* a benefit to
    # learning a policy with randomness (and usually there isn't), the agent
    # should act more deterministically as time goes on. So, the entropy should
    # decrease.
    #
    # No, you don't need to understand the code here.
    obs = torch.Tensor(test_env.reset()[0])
    done = False
    with torch.no_grad():
        # Visualize
        reward_total = 0
        entropy_total = 0.0
        obs = torch.Tensor(test_env.reset()[0])
        eval_steps = 8
        for _ in range(eval_steps):
            avg_entropy = 0.0
            steps_taken = 0
            for _ in range(max_eval_steps):
                distr = Categorical(logits=p_net(obs.unsqueeze(0)).squeeze())
                action = distr.sample().item()
                obs_, reward, done, _, _ = test_env.step(action)
                obs = torch.Tensor(obs_)
                steps_taken += 1
                if done:
                    obs = torch.Tensor(test_env.reset()[0])
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

    obs = torch.Tensor(env.reset()[0])
    done = False

from abc import ABC, abstractmethod
import copy
import math
import random
from typing import Any, Generic, TypeVar
from typing_extensions import override
from gymnasium import Env
import numpy as np


SaveState = TypeVar("SaveState")


class BaseSavableEnv(ABC, Generic[SaveState]):
    """Base class for envrionments that can save and load their state."""

    @abstractmethod
    def save_state(self) -> SaveState:
        """Saves the state of the environment and returns it."""
        pass

    @abstractmethod
    def load_state(self, save_state: SaveState) -> None:
        """Loads the provided state into the environment."""
        pass


State = TypeVar("State")


class BasePolicyValuePredictor(ABC, Generic[State]):
    """Base class for generating policy and state value outputs."""

    @abstractmethod
    def predict(self, state: State) -> tuple[list[float], float]:
        """Given the state, returns a probability distribution over actions and the state's predicted value."""
        pass


class MCTSNode(Generic[State]):
    """A node in the MCTS search tree."""

    def __init__(
        self, prior_prob: float, c1: float, c2: float, discount: float
    ) -> None:
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.mean_action_value = 0.0
        self.children = list[MCTSNode]()

    def expand(
        self, state: State, env: Env, predictor: BasePolicyValuePredictor[State]
    ) -> float:
        """
        Performs the expansion step at this state.

        If this is the first visit, the state will be evaluated.

        On subsequent visits, children of this node will be expanded.
        """
        if len(self.children) == 0:
            # Expand this node
            prior_probs, value = predictor.predict(state)
            self.mean_action_value = value
            self.children = [
                MCTSNode(p, self.c1, self.c2, self.discount) for p in prior_probs
            ]
            self.visit_count = 1
            return self.mean_action_value
        else:
            # Select the best node and expand it
            child_priors = np.array([child.prior_prob for child in self.children])
            child_visits = np.array([child.visit_count for child in self.children])
            puct = self.c1 + math.log((child_visits.sum() + self.c2 + 1) / self.c2)
            u = (puct * child_priors * math.sqrt(child_visits.sum())) / (
                1 + child_visits
            )
            child_values = np.array(
                [child.mean_action_value for child in self.children]
            )
            a = int(np.argmax(child_values + u))
            (
                state,
                reward,
                done,
                _,
                _,
            ) = env.step(a)
            g = 0.0
            if done:
                self.children[a].mean_action_value = (
                    self.children[a].visit_count * self.children[a].mean_action_value
                    + float(reward)
                ) / (self.children[a].visit_count + 1)
                self.children[a].visit_count += 1
                g = self.children[a].mean_action_value
            else:
                g = self.children[a].expand(state, env, predictor)

            # Backup values
            new_return = float(reward) + self.discount * g
            self.mean_action_value = (
                self.visit_count * self.mean_action_value + new_return
            ) / (self.visit_count + 1)
            self.visit_count += 1
            return new_return

    def get_probs(self, temperature: float = 1.0) -> list[float]:
        """Returns action probabilities for this node."""
        denom = sum([child.visit_count ** (1 / temperature) for child in self.children])
        assert denom > 0, "At least one child must be visited first"
        return [
            (child.visit_count ** (1 / temperature)) / denom for child in self.children
        ]


if __name__ == "__main__":
    from gymnasium.envs.classic_control.cartpole import CartPoleEnv
    from gymnasium.spaces import Discrete
    from gymnasium import Env

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

    env = SaveableCartpole(CartPoleEnv(render_mode="human"))
    action_space = env.action_space
    assert isinstance(action_space, Discrete)
    num_actions = int(action_space.n)
    obs, _ = env.reset()

    class DummyPredictor(BasePolicyValuePredictor[np.ndarray]):

        def __init__(self) -> None:
            pass

        @override
        def predict(self, state) -> tuple[list[float], float]:
            return [1.0 / num_actions for _ in range(num_actions)], 0.0

    predictor = DummyPredictor()
    search_env = SaveableCartpole(CartPoleEnv())
    while True:
        # Run MCTS
        save_state = env.save_state()
        root = MCTSNode[np.ndarray](1.0, 15.0, 19652.0, 0.9)
        for _ in range(200):
            search_env.load_state(save_state)
            root.expand(obs, search_env, predictor)
        action = random.choices(list(range(num_actions)), root.get_probs())[0]
        print(action, root.mean_action_value, root.get_probs())

        # Step through state
        obs, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()

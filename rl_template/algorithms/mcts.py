from abc import ABC, abstractmethod
import math
from typing import Generic, TypeVar
from gymnasium import Env
import numpy as np

State = TypeVar("State")


class BasePolicyValuePredictor(ABC):
    """Base class for generating policy and state value outputs."""

    @abstractmethod
    def predict(self, state: State) -> tuple[list[float], float]:
        """Given the state, returns a probability distribution over actions and the state's predicted value."""


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
        else:
            # Select the best node and expand it
            child_priors = np.array([child.prior_prob for child in self.children])
            child_visits = np.array([child.visit_count for child in self.children])
            puct = self.c1 + math.log((child_visits.sum() + self.c2 + 1) / self.c2)
            u = (puct * child_priors * math.sqrt(child_visits.sum())) / (
                1 + child_visits
            )
            child_values = np.array(
                [child.get_mean_action_value() for child in self.children]
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
                    + reward
                ) / self.children[a].visit_count
                self.children[a].visit_count += 1
                g = self.children[a].mean_action_value
            else:
                g = self.children[a].expand(state, env, predictor)

            # Backup values
            new_return = reward + self.discount * g
            self.mean_action_value = (
                self.visit_count * self.mean_action_value + new_return
            ) / (self.visit_count + 1)
            self.visit_count += 1

    def get_probs(self, temperature: float = 1.0) -> list[float]:
        """Returns action probabilities for this node."""
        denom = sum([child.visit_count ** (1 / temperature) for child in self.children])
        assert denom > 0, "At least one child must be visited first"
        return [
            (child.visit_count ** (1 / temperature)) / denom for child in self.children
        ]


if __name__ == "__main__":
    from gymnasium.envs.classic_control.cartpole import CartPoleEnv

    env = CartPoleEnv()

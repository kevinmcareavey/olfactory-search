"""
    Reference: 
     - Paper: https://link.springer.com/article/10.1140/epje/s10189-023-00277-8
     - Code: https://github.com/auroreloisy/otto-benchmark
"""

from abc import abstractmethod

from .parameters import ParametersIsotropic, ParametersWindy

import numpy as np
import scipy

import gymnasium as gym
from gymnasium import spaces


ACTIONS_2D = {
    0: np.array([1, 0]),  # right
    1: np.array([0, 1]),  # up
    2: np.array([-1, 0]),  # left
    3: np.array([0, -1]),  # right
}

ACTIONS_3D = {
    0: np.array([1, 0, 0]),  # x plus 1
    1: np.array([-1, 0, 0]),  # x minus 1
    2: np.array([0, 1, 0]),  # y plus 1
    3: np.array([0, -1, 0]),  # y minus 1
    4: np.array([0, 0, 1]),  # z plus 1
    5: np.array([0, 0, -1]),  # z minus 1
}


class OttoBase(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_dimensions, parameters, render_mode=None):
        self.num_dimensions = num_dimensions
        self.parameters = parameters
        self.render_mode = render_mode

        assert (
            self.render_mode is None
            or self.render_mode in self.metadata["render_modes"]
        )

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=0,
                    high=self.parameters.grid_size - 1,
                    shape=(num_dimensions,),
                    dtype=np.int64,
                ),
                "hits": spaces.Discrete(self.parameters.h_max + 1),
            }
        )

        self._action_to_direction = ACTIONS_2D if num_dimensions == 2 else ACTIONS_3D
        self.action_space = spaces.Discrete(len(self._action_to_direction))
        self._state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        agent_location = self.np_random.integers(
            0, self.parameters.grid_size, size=2, dtype=np.int64
        )
        source_location = self.np_random.integers(
            0, self.parameters.grid_size, size=2, dtype=np.int64
        )

        if options is not None:
            if "agent_location" in options:
                assert self.num_dimensions == len(options["agent_location"])
                agent_location = np.array(options["agent_location"], dtype=np.int64)
            if "source_location" in options:
                assert self.num_dimensions == len(options["source_location"])
                source_location = np.array(options["source_location"], dtype=np.int64)

        self._state = {"agent": agent_location, "source": source_location}

        observation = self._observation()
        info = self._info()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._state["agent"] = np.clip(
            self._state["agent"] + direction, 0, self.parameters.grid_size - 1
        )

        # An episode is done iff the agent has reached the source
        terminated = np.array_equal(self._state["agent"], self._state["source"])
        truncated = False  # use gymnasium.make([...[, max_episode_steps=Parameters.T_max) to handle episode truncation
        observation = self._observation()
        reward = 0 if observation["hits"] == -1 else -1  # Binary sparse rewards
        info = self._info()

        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _observation(self):
        raise NotImplementedError

    @abstractmethod
    def _info(self):  # this should not be exposed to agent
        raise NotImplementedError


class Isotropic2D(OttoBase):
    def __init__(
        self, num_dimensions, parameters: ParametersIsotropic, render_mode=None
    ):
        super(Isotropic2D, self).__init__(num_dimensions, parameters, render_mode)

    def _observation(self):
        if np.array_equal(self._state["agent"], self._state["source"]):
            hits = -1
        else:
            # Euclidean distance
            r = np.linalg.norm(self._state["source"] - self._state["agent"])
            # eq: B5
            mu_r = (
                scipy.special.k0(r / self.parameters.lambda_over_delta_x)
                / scipy.special.k0(1)
            ) * self.parameters.mu0_Poisson
            # weights = ((mu_r ** self.parameters.h) * np.exp(-mu_r)) / scipy.special.factorial(self.parameters.h)  # should be same as scipy.stats.poisson.pmf
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):  # this should not be exposed to agent
        return {"source": self._state["source"]}


class Isotropic3D(OttoBase):
    def __init__(
        self, num_dimensions, parameters: ParametersIsotropic, render_mode=None
    ):
        super(Isotropic3D, self).__init__(num_dimensions, parameters, render_mode)

    def _observation(self):
        if np.array_equal(self._state["agent"], self._state["source"]):
            hits = -1
        else:
            # Euclidean distance
            r = np.linalg.norm(self._state["source"] - self._state["agent"])
            # eq: A1b with V=0
            mu_r = (
                (self.parameters.R_times_delta_t * self.parameters.delta_x_over_a)
                / (2 * r)
                * np.exp(-r / self.parameters.lambda_over_delta_x)
            )
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}


class Windy2D(OttoBase):
    def __init__(self, num_dimensions, parameters: ParametersWindy, render_mode=None):
        super(Windy2D, self).__init__(num_dimensions, parameters, render_mode)

    def _observation(self):
        # The wind blows in the positive x-direction from Section B2
        if np.array_equal(self._state["agent"], self._state["source"]):
            hits = -1
        else:
            r = np.linalg.norm(self._state["source"] - self._state["agent"])
            # x_position is vector(r) * e_x
            x_position = self._state["agent"][0] - self._state["source"][0]
            mu_r = (
                self.parameters.R_bar
                / r
                * np.exp(
                    self.parameters.V_bar * x_position / self.parameters.delta_x_over_a
                    - r / self.parameters.lambda_over_a
                )
            )
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}
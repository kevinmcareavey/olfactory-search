import dataclasses

import numpy as np

import gymnasium as gym
import scipy
from gymnasium import spaces


@dataclasses.dataclass
class Parameters:
    grid_size: int  # length of each dimension
    h_max: int  # maximum number of hits where h in [0, 1, ..., h_max]
    T_max: int  # maximum episode length
    lambda_over_delta_x: float  # dispersion length-scale of particles in medium (lambda) / cell size (delta_x)
    R_times_delta_t: float  # source emission rate (R) * sniff time (delta_t)
    delta_x_over_a: float  # cell size (delta_x) / agent radius (a)

    def __post_init__(self):
        assert self.grid_size > 0
        assert self.h_max > 0
        assert self.T_max > 0
        assert self.lambda_over_delta_x > 0
        assert self.R_times_delta_t > 0
        assert self.delta_x_over_a > 0
        self.lambda_over_a = self.lambda_over_delta_x * self.delta_x_over_a
        self.mu0_Poisson = (1 / np.log(self.lambda_over_a) * scipy.special.k0(1)) * self.R_times_delta_t
        self.h = np.arange(0, self.h_max + 1)


SMALLER_DOMAIN = Parameters(
    grid_size=19,
    h_max=2,
    T_max=642,
    lambda_over_delta_x=1.0,
    R_times_delta_t=1.0,
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
)

LARGER_DOMAIN = Parameters(
    grid_size=53,
    h_max=3,
    T_max=2188,
    lambda_over_delta_x=3.0,
    R_times_delta_t=2.0,
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
)


class Isotropic2D(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, parameters, render_mode=None):
        self.parameters = parameters
        self.render_mode = render_mode

        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.parameters.grid_size-1, shape=(2,), dtype=np.int64),
                "hits": spaces.Discrete(self.parameters.h_max+1),
            }
        )

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # right
        }

        self._state = None

    def _observation(self):
        if np.array_equal(self._state["agent"], self._state["source"]):
            hits = -1
        else:
            r = np.linalg.norm(self._state["source"] - self._state["agent"])
            mu_r = (scipy.special.k0(r / self.parameters.lambda_over_delta_x) / scipy.special.k0(1)) * self.parameters.mu0_Poisson
            # weights = ((mu_r ** self.parameters.h) * np.exp(-mu_r)) / scipy.special.factorial(self.parameters.h)  # should be same as scipy.stats.poisson.pmf
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        agent_location = self.np_random.integers(0, self.parameters.grid_size, size=2, dtype=np.int64)

        source_location = agent_location
        while np.array_equal(source_location, agent_location):
            source_location = self.np_random.integers(0, self.parameters.grid_size, size=2, dtype=np.int64)

        self._state = {"agent": agent_location, "source": source_location}

        observation = self._observation()
        info = self._info()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._state["agent"] = np.clip(self._state["agent"] + direction, 0, self.parameters.grid_size - 1)

        # An episode is done iff the agent has reached the source
        terminated = np.array_equal(self._state["agent"], self._state["source"])
        truncated = False  # use gymnasium.make([...[, max_episode_steps=Parameters.T_max) to handle episode truncation
        observation = self._observation()
        reward = 0 if observation["hits"] == -1 else -1  # Binary sparse rewards
        info = self._info()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

import numpy as np

import gymnasium as gym
import scipy
from gymnasium import spaces


class OTTOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size=5, h_max=2, render_mode=None):
        self.grid_size = grid_size  # The size of the square grid
        self.h_max = h_max
        self.render_mode = render_mode

        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        self._h = np.arange(0, self.h_max + 1)

        self._R = 1.0  # source emission rate
        self._tau = 1.0  # oder particle finite lifetime
        self._D = 1.0  # effective diffusivity
        # self._V = 0  # mean wind speed
        # self._e_x = 0  # wind direction
        self._delta_x = 1.0  # receptor diameter / cell size
        self._delta_t = 1.0  # sense duration

        self._lambda = np.sqrt(self._D * self._tau)

        # Observations are dictionaries with both agent and source locations
        # Each location is encoded as an element of {0, ..., grid_size}^2, i.e. MultiDiscrete([grid_size, grid_size])
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.grid_size-1, shape=(2,), dtype=np.int64),
                "hits": spaces.Discrete(self.h_max+1),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # right
        }

        self._state = None

    def _observation(self):
        r = self._state["source"] - self._state["agent"]
        if np.array_equal(r, np.array([0, 0])):
            hits = -1
        else:
            mu_r = ((self._R * self._delta_t) / np.log((2 * self._lambda) / self._delta_x)) * scipy.special.k0(np.linalg.norm(r) / self._lambda)
            weights = ((mu_r ** self._h) * np.exp(-mu_r)) / scipy.special.factorial(self._h)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self._h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        agent_location = self.np_random.integers(0, self.grid_size, size=2, dtype=np.int64)

        # We will sample the source's location randomly until it does not coincide with the agent's location
        source_location = agent_location
        while np.array_equal(source_location, agent_location):
            source_location = self.np_random.integers(
                0, self.grid_size, size=2, dtype=np.int64
            )

        self._state = {"agent": agent_location, "source": source_location}

        observation = self._observation()
        info = self._info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._state["agent"] = np.clip(
            self._state["agent"] + direction, 0, self.grid_size - 1
        )

        # An episode is done iff the agent has reached the source
        terminated = np.array_equal(self._state["agent"], self._state["source"])
        truncated = False
        observation = self._observation()
        reward = 0 if observation["hits"] == -1 else -1  # Binary sparse rewards
        info = self._info()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

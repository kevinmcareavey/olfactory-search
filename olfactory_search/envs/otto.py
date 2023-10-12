from typing import Any, SupportsFloat

from .parameters import Parameters

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


class Isotropic2D(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, parameters: Parameters, render_mode=None):
        self.parameters = parameters
        self.render_mode = render_mode

        assert (
            self.render_mode is None
            or self.render_mode in self.metadata["render_modes"]
        )

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    0, self.parameters.grid_size - 1, shape=(2,), dtype=np.int64
                ),
                "hits": spaces.Discrete(self.parameters.h_max + 1),
            }
        )

        self._action_to_direction = ACTIONS_2D
        self.action_space = spaces.Discrete(len(ACTIONS_2D))

        self._state = None

    def _observation(self):
        if np.array_equal(self._state["agent"], self._state["source"]):
            hits = -1
        else:
            r = np.linalg.norm(self._state["source"] - self._state["agent"])
            mu_r = (
                scipy.special.k0(r / self.parameters.lambda_over_delta_x)
                / scipy.special.k0(1)
            ) * self.parameters.mu0_Poisson
            # weights = ((mu_r ** self.parameters.h) * np.exp(-mu_r)) / scipy.special.factorial(self.parameters.h)  # should be same as scipy.stats.poisson.pmf
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "agent_location" in options:
            assert len(options["agent_location"]) == 2
            agent_location = np.array(options["agent_location"], dtype=np.int64)
        else:
            agent_location = self.np_random.integers(
                0, self.parameters.grid_size, size=2, dtype=np.int64
            )

        if options is not None and "source_location" in options:
            assert len(options["source_location"]) == 2
            source_location = np.array(options["source_location"], dtype=np.int64)
        else:
            source_location = agent_location

        while np.array_equal(source_location, agent_location):
            source_location = self.np_random.integers(
                0, self.parameters.grid_size, size=2, dtype=np.int64
            )

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

    def render(self):
        pass

    def close(self):
        pass


class Isotropic3D(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, parameters: Parameters, render_mode=None):
        self.parameters = parameters
        self.render_mode = render_mode

        assert (
            self.render_mode is None
            or self.render_mode in self.metadata["render_modes"]
        )

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    0, self.parameters.grid_size - 1, shape=(3,), dtype=np.int64
                ),
                "hits": spaces.Discrete(self.parameters.h_max + 1),
            }
        )

        self._action_to_direction = ACTIONS_3D
        self.action_space = spaces.Discrete(len(ACTIONS_3D))

        self._state = None

    def _observation(self):
        # 3D hit model
        if np.array_equal(self._state["agent"], self._state["source"]):
            hits = -1
        else:
            r = np.linalg.norm(self._state["source"] - self._state["agent"])
            mu_r = (
                self.parameters.lambda_over_delta_x
                / r
                * np.exp(-r / self.parameters.lambda_over_delta_x + 1)
            ) * self.parameters.mu0_Poisson
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "agent_location" in options:
            assert len(options["agent_location"]) == 3
            agent_location = np.array(options["agent_location"], dtype=np.int64)
        else:
            agent_location = self.np_random.integers(
                0, self.parameters.grid_size, size=3, dtype=np.int64
            )

        if options is not None and "source_location" in options:
            assert len(options["source_location"]) == 3
            source_location = np.array(options["source_location"], dtype=np.int64)
        else:
            source_location = agent_location

        while np.array_equal(source_location, agent_location):
            source_location = self.np_random.integers(
                0, self.parameters.grid_size, size=3, dtype=np.int64
            )

        self._state = {"agent": agent_location, "source": source_location}

        observation = self._observation()
        info = self._info()

        return observation, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        direction = self._action_to_direction[action]

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


class Windy2D(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, parameters: Parameters, render_mode=None):
        self.parameters = parameters
        self.render_mode = render_mode

        assert (
            self.render_mode is None
            or self.render_mode in self.metadata["render_modes"]
        )

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    0, self.parameters.grid_size - 1, shape=(2,), dtype=np.int64
                ),
                "hits": spaces.Discrete(self.parameters.h_max + 1),
            }
        )

        self._action_to_direction = ACTIONS_2D
        self.action_space = spaces.Discrete(len(ACTIONS_2D))

        self._state = None

    def _observation(self):
        # 2D windy model
        # The wind blows in the positive x-direction
        if np.array_equal(self._state["agent"], self._state["source"]):
            hits = -1
        else:
            r = np.linalg.norm(self._state["source"] - self._state["agent"])
            x_position = self._state["agent"][0] - self._state["source"][0]
            mu_r = (
                self.parameters.lambda_over_delta_x
                / r
                * np.exp(
                    0.5 * self.parameters.V_times_delta_t * x_position
                    - r / self.parameters.lambda_bar
                )
            )
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "agent_location" in options:
            assert len(options["agent_location"]) == 2
            agent_location = np.array(options["agent_location"], dtype=np.int64)
        else:
            agent_location = self.np_random.integers(
                0, self.parameters.grid_size, size=2, dtype=np.int64
            )

        if options is not None and "source_location" in options:
            assert len(options["source_location"]) == 2
            source_location = np.array(options["source_location"], dtype=np.int64)
        else:
            source_location = agent_location

        while np.array_equal(source_location, agent_location):
            source_location = self.np_random.integers(
                0, self.parameters.grid_size, size=2, dtype=np.int64
            )

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

    def render(self):
        pass

    def close(self):
        pass


class Windy3D(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, parameters: Parameters, render_mode=None):
        self.parameters = parameters
        self.render_mode = render_mode

        assert (
            self.render_mode is None
            or self.render_mode in self.metadata["render_modes"]
        )

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    0, self.parameters.grid_size - 1, shape=(3,), dtype=np.int64
                ),
                "hits": spaces.Discrete(self.parameters.h_max + 1),
            }
        )

        self._action_to_direction = ACTIONS_3D
        self.action_space = spaces.Discrete(len(ACTIONS_3D))

        self._state = None

    def _observation(self):
        # 3D windy model
        # The wind blows in the positive x-direction
        if np.array_equal(self._state["agent"], self._state["source"]):
            hits = -1
        else:
            r = np.linalg.norm(self._state["source"] - self._state["agent"])
            x_position = self._state["agent"][0] - self._state["source"][0]
            mu_r = (
                self.parameters.lambda_over_delta_x
                / r
                * np.exp(
                    0.5 * self.parameters.V_times_delta_t * x_position
                    - r / self.parameters.lambda_bar
                )
            )
            weights = scipy.stats.poisson.pmf(self.parameters.h, mu_r)
            probabilities = weights / np.sum(weights)
            hits = self.np_random.choice(self.parameters.h, p=probabilities)
        return {"agent": self._state["agent"], "hits": hits}

    def _info(self):
        return {"source": self._state["source"]}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "agent_location" in options:
            assert len(options["agent_location"]) == 3
            agent_location = np.array(options["agent_location"], dtype=np.int64)
        else:
            agent_location = self.np_random.integers(
                0, self.parameters.grid_size, size=2, dtype=np.int64
            )

        if options is not None and "source_location" in options:
            assert len(options["source_location"]) == 3
            source_location = np.array(options["source_location"], dtype=np.int64)
        else:
            source_location = agent_location

        while np.array_equal(source_location, agent_location):
            source_location = self.np_random.integers(
                0, self.parameters.grid_size, size=3, dtype=np.int64
            )

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

    def render(self):
        pass

    def close(self):
        pass

# olfactory-search

A simple implementation of the [olfactory search problem](https://github.com/C0PEP0D/otto) as a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment.

# Installation

```bash
git clone https://github.com/kevinmcareavey/olfactory-search.git
cd olfactory-search
python3 -m pip install .
```

# Getting started

```python
import gymnasium
from olfactory_search.envs import SMALLER_DOMAIN, LARGER_DOMAIN

seed = None  # if seed is not None then episodes will always start in the same initial state
parameters = SMALLER_DOMAIN
env = gymnasium.make("olfactory_search/Isotropic2D-v0", parameters=parameters, max_episode_steps=parameters.T_max)

observation, info = env.reset(seed=seed)
print(f"initial observation = {observation}, info = {info}")
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"action = {action}, observation = {observation}, reward = {reward}, done = {terminated or truncated}, info = {info}")

    if terminated or truncated:
        print(f"episode done")
        observation, info = env.reset(seed=seed)
        print(f"\ninitial observation = {observation}, info = {info}")

print("simulation done")
env.close()
```

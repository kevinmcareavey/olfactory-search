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
import olfactory_search

seed = 2
env = gymnasium.make("olfactory_search/OTTO-v0", grid_size=19, h_max=2)

observation, info = env.reset(seed=seed)
print(f"initial observation = {observation}, info = {info}")
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"action = {action}, observation = {observation}, reward = {reward}, done = {terminated or truncated}, info = {info}")

    if terminated or truncated:
        observation, info = env.reset(seed=seed)
        print(f"\ninitial observation = {observation}")

print("simulation complete")
env.close()
```

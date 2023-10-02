from gymnasium.envs.registration import register

register(
     id="olfactory_search/OTTO-v0",
     entry_point="olfactory_search.envs:OTTOEnv",
)

from gymnasium.envs.registration import register

register(
     id="olfactory_search/Isotropic2D-v0",
     entry_point="olfactory_search.envs:Isotropic2D",
)

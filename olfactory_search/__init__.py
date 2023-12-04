from gymnasium.envs.registration import register

register(
     id="olfactory_search/Isotropic2D-v0",
     entry_point="olfactory_search.envs:Isotropic2D",
)

register(
     id="olfactory_search/Isotropic3D-v0",
     entry_point="olfactory_search.envs:Isotropic3D",
)

register(
     id="olfactory_search/Windy2D-v0",
     entry_point="olfactory_search.envs:Windy2D",
)

register(
     id="olfactory_search/Windy3D-v0",
     entry_point="olfactory_search.envs:Windy3D",
)

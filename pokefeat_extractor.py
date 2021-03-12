"""
Since pokemon are largely encoded equally, we want to have a feature encoder
that processes all the pokemon in parallel, ideally extracting salient features
"""
import gym
import torch
import torch.nn as nn
from typing import List

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PokemonFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param team_size (int): the number of pokemon in a team tensor
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 500,
        team_size: int = 6,
        net_arch: List[int] = None,
    ):
        super(PokemonFeatureExtractor, self).__init__(observation_space, features_dim)
        # Just going stacks of dense here
        self.poke_feat = observation_space.shape[1] // team_size
        self.encoder = nn.Sequential(
            nn.Linear(self.poke_feat, self.poke_feat * 2),
            nn.ReLU(),
            nn.Linear(self.poke_feat * 2, self.poke_feat * 2),
            nn.ReLU(),
            nn.Linear(self.poke_feat * 2, self.poke_feat),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.encoder(
                torch.as_tensor(observation_space.sample()[None])
                .reshape(1, 2, 6, self.poke_feat)
                .float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.encoder(observations.reshape(-1, 2, 6, self.poke_feat)))

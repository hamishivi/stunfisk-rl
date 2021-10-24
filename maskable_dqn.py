
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule

# POLICIES
# only real difference is the 'set_action_mask' function, which allows setting a mask
# which then gets applied on the forward function
# this can just be used by dqn natively.

class MaskableQNetwork(QNetwork):
    def set_action_mask(self, action_mask):
        self.action_mask = action_mask

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.
        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        res = self.q_net(self.extract_features(obs))
        if self.action_mask:
            return res * self.action_mask
        return res

class MaskableDQNPolicy(DQNPolicy):
    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskableQNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # TODO add checks that youve got a maskablepolicyfeaturesextractor here
        mask = self.features_extractor.mask
        self.q_net.set_action_mask(mask)
        self.q_net_target.set_action_mask(mask)
        return self._predict(obs, deterministic=deterministic)

class MaskablePolicyFeatureExtractor(CombinedExtractor):
    def __init__(self, observation_space):
        super(MaskablePolicyFeatureExtractor, self).__init__(observation_space)
        self.action_mask = None

    def forward(self, observations):
        self.action_mask = observations['action_mask']
        return super().forward(observations['battle'])

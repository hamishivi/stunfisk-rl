"""
Since pokemon are largely encoded equally, we want to have a feature encoder
that processes all the pokemon in parallel, ideally extracting salient features
"""
from collections import defaultdict
import gym
import torch
import torch.nn as nn
from typing import List, Dict

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from battle_converter import BattleOptions


class PokemonFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param team_size (int): the number of pokemon in a team tensor
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        poke_feats: List[BattleOptions],
        move_feats: List[BattleOptions],
        features_dim: int = 500,
    ):
        super(PokemonFeatureExtractor, self).__init__(observation_space, features_dim)
        # we will have two components to a pokemon extractor:
        # first, move extractor, to convert move features to fixed embedding
        # second, pokemon extractor, to convert pokemon to fixed embedding.
        # the idea is to allow a network to construct an idea of pokemon/types/etc.
        # that can be used to process it all.
        # first, setup embedding layers
        type_dim = 20
        cat_dim = 4
        gen_dim = 4
        move_dim = 100
        self.type_encoder = nn.Embedding(19, type_dim)
        self.category_encoder = nn.Embedding(3, cat_dim)
        self.gender_encoder = nn.Embedding(3, gen_dim)
        # next, our basic move encoder
        # TODO: probably nicer if battle option notes its categorical.
        move_feat_size = sum(
            [
                m.shape
                for m in move_feats
                if "type" not in m.name and "cat" not in m.name
            ]
        )
        ## add type and cat
        move_feat_size += type_dim + cat_dim
        self.move_encoder = nn.Sequential(
            nn.Linear(move_feat_size, 200),
            nn.ReLU(),
            nn.Linear(200, move_dim),
            nn.ReLU(),
        )
        # next, pokemon encoder
        poke_feat_size = sum(
            [
                m.shape
                for m in poke_feats
                if "type" not in m.name and "gender" not in m.name
            ]
        )
        # add types and gender
        poke_feat_size += type_dim * 2 + gen_dim
        poke_feat_size += move_dim * 4  # add moves
        self.poke_encoder = nn.Sequential(
            nn.Linear(poke_feat_size, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(poke_feat_size * 2, features_dim), nn.ReLU()
        )

    def encode_pokemon(
        self, observations: Dict[str, torch.Tensor], idx: int
    ) -> torch.Tensor:
        # we construct a tensor by simply concatenating everything together
        # if was categorical we pass it through.
        ours_tensor = []
        enemy_tensor = []
        moves_ours = defaultdict(list)
        moves_enemy = defaultdict(list)
        for k, v in observations.items():
            if int(k.split(".")[1]) != idx:
                continue
            if "type" in k:
                v = self.type_encoder(v.long())
            elif "gender" in k:
                v = self.gender_encoder(v.long())
            elif "cat" in k:
                v = self.category_encoder(v.long())
            if len(v.shape) > 2:
                v = v.squeeze(1)
            # special handling for moves
            if "move" in k:
                if "ours" in k:
                    moves_ours[k.split(".")[3]].append(v)
                else:
                    moves_enemy[k.split(".")[3]].append(v)
            elif "ours" in k:
                ours_tensor.append(v)
            else:
                enemy_tensor.append(v)
        # go through generate move vecs
        for m in moves_ours:
            move_tensor = torch.cat(moves_ours[m], 1)
            move_val = self.move_encoder(move_tensor)
            ours_tensor.append(move_val)
        for m in moves_enemy:
            move_tensor = torch.cat(moves_enemy[m], 1)
            move_val = self.move_encoder(move_tensor)
            enemy_tensor.append(move_val)
        # finally add encode pokemon
        # ours_tensor = self.poke_encoder(torch.cat(ours_tensor, 1))
        # enemy_tensor = self.poke_encoder(torch.cat(enemy_tensor, 1))
        return torch.cat(ours_tensor, 1), torch.cat(enemy_tensor, 1)

    def forward(self, observations: Dict[str, torch.tensor]) -> torch.Tensor:
        ours, enemy = zip(*[self.encode_pokemon(observations, i) for i in range(1)])
        team = torch.cat([torch.cat(ours, 1), torch.cat(enemy, 1)], 1)
        return self.linear(team)

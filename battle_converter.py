"""
Utilities for converting pokemon and battle-related items to tensors.
"""
import numpy as np
from data import TYPES, MOVE_CATS, GENDERS
from gym import spaces


class BattleOptions:
    """
    Class for representing presence of a battle item.
    This helps us with configuring tensor creation via config
    """

    def __init__(self, name, lower_bound, upper_bound, shape, extract_func, active):
        self.name = name
        self.l = lower_bound
        self.u = upper_bound
        self.a = active
        self.s = shape
        # a bit funky, but oh well
        self.e = extract_func

    @property
    def shape(self):
        return self.s

    @property
    def lower_bound(self):
        return self.l

    @property
    def upper_bound(self):
        return self.u

    @property
    def active(self):
        return self.a

    def extract(self, *args, **kwargs):
        x = self.e(*args, **kwargs)
        # if cant extract, return lower bound
        return x if x is not None else self.l


class BattleConverter:
    def __init__(self, cfg):
        # config so we can easily test out stuff
        self.cfg = cfg
        # move features
        self.move_feats = [
            BattleOptions(
                "acc", 0, 100, 1, lambda x: x.accuracy, cfg.BATTLE.MOVE.ACCURACY
            ),
            BattleOptions(
                "bsp", 0, 200, 1, lambda x: x.base_power, cfg.BATTLE.MOVE.BASE_POWER
            ),
            BattleOptions("pp", 0, 50, 1, lambda x: x.current_pp, cfg.BATTLE.MOVE.PP),
            BattleOptions(
                "pri", 0, 14, 1, lambda x: x.priority + 7, cfg.BATTLE.MOVE.PRIORITY
            ),
            BattleOptions(
                "cat", 0, 1, 1, lambda x: MOVE_CATS[x.category], cfg.BATTLE.MOVE.CAT
            ),
            BattleOptions(
                "type", 0, 1, 1, lambda x: TYPES[x.type], cfg.BATTLE.MOVE.TYPE
            ),
            BattleOptions("mm", 0, 5, 1, lambda x, d: d(x), cfg.BATTLE.MOVE.MOVE_MULT),
        ]
        # pokemon features
        self.poke_feats = [
            BattleOptions(
                "active", 0, 1, 1, lambda x: int(x.active), cfg.BATTLE.POKEMON.ACTIVE
            ),
            BattleOptions(
                "atk",
                0,
                2000,
                1,
                lambda x: x.base_stats["atk"],
                cfg.BATTLE.POKEMON.ATTACK,
            ),
            BattleOptions(
                "def",
                0,
                2000,
                1,
                lambda x: x.base_stats["def"],
                cfg.BATTLE.POKEMON.DEFENCE,
            ),
            BattleOptions(
                "spa",
                0,
                2000,
                1,
                lambda x: x.base_stats["spa"],
                cfg.BATTLE.POKEMON.SPECIAL_ATTACK,
            ),
            BattleOptions(
                "spd",
                0,
                2000,
                1,
                lambda x: x.base_stats["spd"],
                cfg.BATTLE.POKEMON.SPECIAL_DEFENCE,
            ),
            BattleOptions(
                "spe",
                0,
                2000,
                1,
                lambda x: x.base_stats["spe"],
                cfg.BATTLE.POKEMON.SPEED,
            ),
            BattleOptions(
                "hp",
                0,
                2000,
                1,
                lambda x: x.current_hp if x.current_hp else 0,
                cfg.BATTLE.POKEMON.HP,
            ),
            BattleOptions(
                "hp_frac",
                0,
                1,
                1,
                lambda x: x.current_hp_fraction,
                cfg.BATTLE.POKEMON.HP_FRACTION,
            ),
            BattleOptions(
                "att", 0, 1, 1, lambda x: int(x.fainted), cfg.BATTLE.POKEMON.FAINTED
            ),
            BattleOptions(
                "gender",
                0,
                1,
                1,
                lambda x: GENDERS[x.gender],
                cfg.BATTLE.POKEMON.GENDER,
            ),
            BattleOptions(
                "type1", 0, 1, 1, lambda x: TYPES[x.types[0]], cfg.BATTLE.POKEMON.TYPE1
            ),
            BattleOptions(
                "type2",
                0,
                1,
                1,
                lambda x: TYPES[x.types[0] if len(x.types) > 1 else None],
                cfg.BATTLE.POKEMON.TYPE2,
            ),
        ]
        self.num_moves = (
            cfg.BATTLE.POKEMON.NUM_MOVES
        )  # this will always be 4, basically
        # team options
        self.num_poke = cfg.BATTLE.TEAM.SIZE  # this will always be 6, basically
        # field/battle options - currently none but maybe will add field types in future

    def get_observation_space(self):
        # overall battle observation space. Currently,
        # will just do two pokemon: active opp, active ours.
        # eventually add teams.
        # Nested dicts are not supported, so we use a nested key
        # structure instead!
        od = {}
        for i in range(1):
            for k, v in self.get_pokemon_obs_dict().items():
                od[f"ours.{i}.{k}"] = v
        for i in range(1):
            for k, v in self.get_pokemon_obs_dict().items():
                od[f"enemy.{i}.{k}"] = v
        return spaces.Dict(od)

    def get_pokemon_obs_dict(self):
        # dict representing a single pokemon
        od = {
            m.name: spaces.Box(low=m.lower_bound, high=m.upper_bound, shape=(m.shape,))
            for m in self.poke_feats
            if m.active
        }
        for i in range(self.num_moves):
            for k, v in self.get_move_obs_dict().items():
                od[f"moves.{i}.{k}"] = v
        return od

    def get_move_obs_dict(self):
        # dict representing a single move
        return {
            m.name: spaces.Box(low=m.lower_bound, high=m.upper_bound, shape=(m.shape,))
            for m in self.move_feats
            if m.active
        }

    def _extract_move(self, move_obj, enemy_types):
        d = {
            f.name: f.extract(move_obj)
            for f in self.move_feats
            if f.active and f.name != "mm"
        }
        if self.move_feats[-1].active:
            mult = move_obj.type.damage_multiplier(
                enemy_types[0], enemy_types[1] if len(enemy_types) > 1 else None
            )
            d["mm"] = mult
        return d

    # TODO: proper dummy values
    def _generate_dummy_move(self):
        return {f.name: f.upper_bound for f in self.move_feats if f.active}

    def _extract_poke(self, poke_obj, enemy_type):
        vals = {f.name: f.extract(poke_obj) for f in self.poke_feats if f.active}
        move_counter = 0
        for i, m in enumerate(poke_obj.moves):
            for k, v in self._extract_move(poke_obj.moves[m], enemy_type).items():
                vals[f"moves.{i}.{k}"] = v
            move_counter += 1
            if move_counter >= self.num_moves:
                break
        while move_counter < self.num_moves:
            for k, v in self._generate_dummy_move().items():
                vals[f"moves.{move_counter}.{k}"] = v
            move_counter += 1
        return vals

    # TODO: proper dummy values
    def _generate_dummy_poke(self):
        vals = {f.name: f.upper_bound for f in self.poke_feats if f.active}
        for i in range(self.num_moves):
            for k, v in self._generate_dummy_move().items():
                vals[f"moves.{i}.{k}"] = v
        return vals

    def battle_to_tensor(self, battle):
        od = {}
        counter = 0
        # this is sort of hacky...
        team = [battle.active_pokemon]
        for i, (pokemon) in enumerate(team):
            for k, v in self._extract_poke(
                pokemon, battle.opponent_active_pokemon.types
            ).items():
                od[f"ours.{i}.{k}"] = v
            counter += 1
        # while counter < 6:
        #    for k, v in self._generate_dummy_poke().items():
        #        od[f"ours.{counter}.{k}"] = v
        #    counter += 1
        counter = 0
        team = [battle.opponent_active_pokemon]
        for i, (pokemon) in enumerate(team):
            for k, v in self._extract_poke(
                battle.opponent_active_pokemon, battle.active_pokemon.types
            ).items():
                od[f"enemy.{i}.{k}"] = v
            counter += 1
        # while counter < 6:
        #    for k, v in self._generate_dummy_poke().items():
        #        od[f"enemy.{counter}.{k}"] = v
        #    counter += 1
        return od

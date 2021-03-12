"""
Utilities for converting pokemon and battle-related items to tensors.
"""
import numpy as np
from data import TYPES, MOVE_CATS, GENDERS


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
                "type_mult",
                0,
                5,
                lambda x, b: x.damage_multiplier(
                    b.opponent_active_pokemon.type_1, b.opponent_active_pokemon.type_1
                ),
                cfg.BATTLE.MOVE.MOVE_MULT,
            ),
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
                "cat",
                0,
                1,
                len(MOVE_CATS),
                lambda x: np.eye(len(MOVE_CATS))[MOVE_CATS[x.category]],
                cfg.BATTLE.MOVE.CAT,
            ),
            BattleOptions(
                "type",
                0,
                1,
                len(TYPES),
                lambda x: np.eye(len(TYPES))[TYPES[x.type]],
                cfg.BATTLE.MOVE.TYPE,
            ),
        ]
        # pokemon features
        self.poke_feats = [
            BattleOptions(
                "active", 0, 1, 1, lambda x: int(x.active), cfg.BATTLE.POKEMON.ACTIVE
            ),
            BattleOptions(
                "atk", 0, 2000, 1, lambda x: x.stats["atk"], cfg.BATTLE.POKEMON.ATTACK
            ),
            BattleOptions(
                "def", 0, 2000, 1, lambda x: x.stats["def"], cfg.BATTLE.POKEMON.DEFENCE
            ),
            BattleOptions(
                "spa",
                0,
                2000,
                1,
                lambda x: x.stats["spa"],
                cfg.BATTLE.POKEMON.SPECIAL_ATTACK,
            ),
            BattleOptions(
                "spd",
                0,
                2000,
                1,
                lambda x: x.stats["spd"],
                cfg.BATTLE.POKEMON.SPECIAL_DEFENCE,
            ),
            BattleOptions(
                "spe", 0, 2000, 1, lambda x: x.stats["spe"], cfg.BATTLE.POKEMON.SPEED
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
                len(GENDERS),
                lambda x: np.eye(len(GENDERS))[GENDERS[x.gender]],
                cfg.BATTLE.POKEMON.GENDER,
            ),
            BattleOptions(
                "type1",
                0,
                1,
                len(TYPES),
                lambda x: np.eye(len(TYPES))[TYPES[x.types[0]]],
                cfg.BATTLE.POKEMON.TYPE1,
            ),
            BattleOptions(
                "type1",
                0,
                1,
                len(TYPES),
                lambda x: np.eye(len(TYPES))[
                    TYPES[x.types[0] if len(x.types) > 1 else TYPES["NULL"]]
                ],
                cfg.BATTLE.POKEMON.TYPE2,
            ),
        ]
        self.num_moves = (
            cfg.BATTLE.POKEMON.NUM_MOVES
        )  # this will always be 4, basically
        # team options
        self.num_poke = cfg.BATTLE.TEAM.SIZE  # this will always be 6, basically
        # field/battle options - currently none but maybe will add field types in future

    def get_tensor_shape(self):
        # nice and simple for now: we concat everything, and so just add all shapes
        move_s = sum([x.shape for x in self.move_feats]) * self.num_moves
        poke_s = (sum([x.shape for x in self.poke_feats]) + move_s) * self.num_poke
        return (2, poke_s)

    def get_lower_bounds(self):
        # similar to above, but get lower bound for it all
        # note that must match tensor embed
        def poke_lb():
            lb = []
            for _ in range(self.num_poke):
                for f in self.poke_feats:
                    lb += [f.lower_bound for _ in range(f.shape)]
                for _ in range(self.num_moves):
                    for f in self.move_feats:
                        lb += [f.lower_bound for _ in range(f.shape)]
            return lb

        return np.array([poke_lb(), poke_lb()])

    def get_upper_bounds(self):
        def poke_ub():
            ub = []
            for _ in range(self.num_poke):
                for f in self.poke_feats:
                    ub += [f.upper_bound for _ in range(f.shape)]
                for _ in range(self.num_moves):
                    for f in self.move_feats:
                        ub += [f.upper_bound for _ in range(f.shape)]
            return ub

        return np.array([poke_ub(), poke_ub()])

    def _extract_move(self, move_obj, battle_obj):
        # damage mult is a special case
        vals = [self.move_feats[0].extract(move_obj, battle_obj)]
        for f in self.move_feats[1:]:
            x = f.extract(move_obj)
            # for now, output is either list of vals or one val
            if type(x) == np.ndarray:
                vals += x.tolist()
            else:
                vals.append(x)
        return vals

    def _generate_dummy_move(self):
        vals = []
        for f in self.move_feats:
            vals += [f.lower_bound for _ in range(f.shape)]
        return vals

    def _extract_poke(self, poke_obj, battle_obj):
        vals = []
        for f in self.poke_feats:
            x = f.extract(poke_obj)
            # for now, output is either list of vals or one val
            if type(x) == np.ndarray:
                vals += x.tolist()
            else:
                vals.append(x)
        move_counter = 0
        for m in poke_obj.moves:
            vals += self._extract_move(poke_obj.moves[m], battle_obj)
            move_counter += 1
            if move_counter >= self.num_moves:
                break
        while move_counter < self.num_moves:
            vals += self._generate_dummy_move()
            move_counter += 1
        return vals

    def _generate_dummy_poke(self):
        vals = []
        for f in self.poke_feats:
            vals += [f.lower_bound for x in range(f.shape)]
        for _ in range(self.num_moves):
            vals += self._generate_dummy_move()
        return vals

    def battle_to_tensor(self, battle):
        our_team = []
        for poke in battle.team:
            our_team += self._extract_poke(battle.team[poke], battle)
        p_count = 0
        their_team = []
        for poke in battle.opponent_team:
            p_count += 1
            their_team += self._extract_poke(battle.opponent_team[poke], battle)
        while p_count < self.num_poke:
            their_team += self._generate_dummy_poke()
            p_count += 1
        res = np.array([our_team, their_team])
        if res.shape != (2, 924):
            import pdb

            pdb.set_trace()
        return np.array([our_team, their_team])

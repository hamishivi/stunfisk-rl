import numpy as np
from gym import spaces
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player

from poke_tensor_utils import opposing_poke_to_tensor, poke_to_tensor, poke_shape

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        # init all the parent stuff
        super().__init__(*args, **kwargs)
        # define observation space, which is based on embed_battle
        poke_shapes = poke_shape()
        lower_bounds = np.repeat(np.array(poke_shapes['lower_bounds']).reshape(1, -1), 7, axis=0)
        upper_bounds = np.repeat(np.array(poke_shapes['upper_bounds']).reshape(1, -1), 7, axis=0)
        self.observation_space = spaces.Box(
            low=lower_bounds,
            high=upper_bounds,
            shape=(7, poke_shapes['shape'][0],)
        )
        self.action_box = spaces.Discrete(super().action_space[-1])
        self.cur_bat = None

    @property
    def action_space(self):
        return self.action_box
        
    # how to embed battle?
    # how to represent current pokemon?
    # one idea:
    # stats + 
    def embed_battle(self, battle):
        def damage_mult_func(move):
            return move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
        #print([poke_to_tensor(battle.team[p], damage_mult_func).shape for p in battle.team])
        poke_tensors = [poke_to_tensor(battle.team[p], damage_mult_func) for p in battle.team]
        def damage_mult_func_opp(move):
            return move.type.damage_multiplier(
                    battle.active_pokemon.type_1,
                    battle.active_pokemon.type_2,
                )
        poke_tensors.append(opposing_poke_to_tensor(battle.opponent_active_pokemon, damage_mult_func_opp))
        our_team = np.stack(poke_tensors, axis=0)

        #remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        #remaining_mon_opponent = (
        #    len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        #)
        return our_team

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2,
            hp_value=1,
            victory_value=30,
        )

# for playing against users
class EvaluatePlayer(Player):
    def __init__(self, player, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_player = player
        self.model = model

    def choose_move(self, battle):
        obs = self.env_player.embed_battle(battle)
        action, _ = self.model.predict(obs.reshape(1, -1), deterministic=True)
        return self.env_player._action_to_move(action[0], battle)


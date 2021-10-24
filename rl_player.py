from gym import spaces
from gym.spaces import flatten
import numpy as np
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.player import Player

from battle_converter import BattleConverter


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    """
    Class to handle the interaction between game and algo
    Main 'embedding' handled by BattleConverter class
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.bc = BattleConverter(cfg)
        self.observation_space = self.bc.get_observation_space()
        self.action_box = spaces.Discrete(super().action_space[-1] + 1)
        self.action_mask_in_obs = True

    @property
    def action_space(self):
        return self.action_box

    def index_to_move(self, action_index, battle):
        return self._action_to_move(action_index, battle)

    def embed_battle(self, battle):
        if self.action_mask_in_obs:
            return spaces.Dict({"battle": self.bc.battle_to_tensor(battle), "action_mask": self.action_masks()})
        return self.bc.battle_to_tensor(battle)

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=self.cfg.REWARD.FAINTED,
            hp_value=self.cfg.REWARD.HP,
            victory_value=self.cfg.REWARD.VICTORY,
        )

    # true if move is valid, false otherwise
    def isGen8ValidMove(self, action, battle):
        if action == -1:
            return True # forfeit always valid
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return True
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0
            <= action - 4
            < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
        ):
            return True
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return True
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return True
        elif 0 <= action - 16 < len(battle.available_switches):
            return True
        else:
            return False
    
    def action_masks(self) -> np.ndarray:
        # this is an expensive way to generate the mask but its conceptually simple: 
        # test each action index for validity. we could construct the mask with a
        # cleverer func faster but idm slowness rn
        # one edge case: force switch with no pokemon to switch to. mask will be all 0s
        # this is weird anyway, will leave as a bug (lol)
        num_moves = super().action_space[-1] + 1
        mask = np.ones(num_moves)
        for x in range(num_moves):
            if not self.isGen8ValidMove(x, self._current_battle):
                mask[x] = 0
        return mask



# for playing against users
class EvaluatePlayer(Player):
    def __init__(self, player, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_player = player
        self.model = model

    def choose_move(self, battle):
        # embed, and flatten the result
        embed_battle = self.env_player.observation(self.env_player.embed_battle(battle))
        action, _ = self.model.predict(embed_battle, deterministic=True)
        return self.env_player.index_to_move(action, battle)

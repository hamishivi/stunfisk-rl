from gym import spaces
from gym.spaces import flatten
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
        self.action_box = spaces.Discrete(super().action_space[-1])
        self.cur_bat = None

    @property
    def action_space(self):
        return self.action_box

    def index_to_move(self, action_index, battle):
        return self._action_to_move(action_index, battle)

    def embed_battle(self, battle):
        return self.bc.battle_to_tensor(battle)

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=self.cfg.REWARD.FAINTED,
            hp_value=self.cfg.REWARD.HP,
            victory_value=self.cfg.REWARD.VICTORY,
        )


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

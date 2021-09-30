import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from yacs.config import CfgNode
from config import cfg
from poke_env.player.random_player import RandomPlayer
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines.deepq.policies import FeedForwardPolicy
from max_player import MaxDamagePlayer
from rl_player import SimpleRLPlayer
from battle_env import train, test
import yaml
from gym.wrappers import FlattenObservation


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(
            *args,
            **kwargs,
            layers=[256, 128, 64],
            layer_norm=True,
            feature_extraction="mlp",
        )


def cfg_node_to_dict(cfg):
    raw_cfg = yaml.safe_load(cfg.dump())
    return raw_cfg


# flatten by making everything top level, and using '_' to join
def flatten_dict(raw_cfg):
    def expand(key, value):
        if isinstance(value, dict):
            return [(key + "-" + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in raw_cfg.items() for item in expand(k, v)]

    return dict(items)


# main driver for running stuff
# gen8anythinggoes
def train_and_test(
    cfg, battle_format, team_file=None, enemy_team_file=None, train_rand=True, verbose=1
):
    team, enemy_team = None, None
    if team_file:
        team = open(team_file, "r").read()
    if enemy_team_file:
        enemy_team = open(enemy_team_file, "r").read()

    env_player = FlattenObservation(SimpleRLPlayer(cfg, battle_format=battle_format))
    max_opponent = MaxDamagePlayer(battle_format=battle_format)
    rand_opponent = RandomPlayer(battle_format=battle_format)
    model = DQN(
        CustomDQNPolicy,
        env_player,
        learning_rate=cfg.DQN.LEARNING_RATE,
        buffer_size=cfg.DQN.BUFFER_SIZE,
        learning_starts=cfg.DQN.LEARNING_STARTS,
        gamma=cfg.DQN.GAMMA,
        verbose=verbose,
        tensorboard_log="./dqn_pokemon_tensorboard/",
    )
    # train against both?
    if train_rand:
        train(env_player, rand_opponent, model, timesteps=cfg.DQN.TRAIN_TIMESTEPS)
    else:
        train(env_player, max_opponent, model, timesteps=cfg.DQN.TRAIN_TIMESTEPS)
    print("evaluating...")
    # model.save("gen5")
    rand_won = test(env_player, rand_opponent, model)
    max_won = test(env_player, max_opponent, model)

    return rand_won, max_won, env_player, model


def run_exp(exp_name, team, enemy_team, battle_format="gen8anythinggoes"):
    print(f"Running {exp_name}")
    results = {}
    r, m, _, _ = train_and_test(cfg, battle_format, team, enemy_team)
    results["rand | rand"] = r
    results["rand | max"] = m
    print(results)
    r, m, _, _ = train_and_test(cfg, battle_format, team, enemy_team, train_rand=False)
    results["max | rand"] = r
    results["max | max"] = m
    return results


# some of this cfg code was used for sweeps, but i removed this.
raw_cfg = cfg_node_to_dict(cfg)
raw_cfg_flat = flatten_dict(raw_cfg)
cfg.merge_from_other_cfg(CfgNode(raw_cfg))
# uncomment to use raml
# cfg.merge_from_file("basic.yaml")
results = {}
# tests:
# grookey vs youngster jake rand and max
# results["Jake vs Grookey"] = run_exp(
#     "Jake vs Grookey", "teams/starting_grookey.txt", "teams/youngster_jake.txt"
# )
# red vs red rand and max
# results["Red vs Red  "] = run_exp(
#     "Red vs Red  ", "teams/red.txt", "teams/red.txt", battle_format="gen7anythinggoes"
# )
# full randoms rand and max
results["Rand vs Rand"] = run_exp(
    "Rand vs Rand", None, None, battle_format="gen8randombattle"
)
# do what you want with results!
print(results)

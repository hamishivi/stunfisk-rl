from yacs.config import CfgNode
from config import cfg
from poke_env.player.random_player import RandomPlayer
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy as DqnMlpPolicy
from max_player import MaxDamagePlayer
from rl_player import SimpleRLPlayer
from pokefeat_extractor import PokemonFeatureExtractor
from battle_env import train, test
import yaml


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

    env_player = SimpleRLPlayer(cfg, battle_format=battle_format, team=team)
    max_opponent = MaxDamagePlayer(battle_format=battle_format, team=enemy_team)
    rand_opponent = RandomPlayer(battle_format=battle_format, team=enemy_team)
    policy_kwargs = dict(
        features_extractor_class=PokemonFeatureExtractor,
        net_arch=[cfg.NETWORK.POKEMON_FEATURE_SIZE] + [1000, 500, 250, 100, 50],
        features_extractor_kwargs=dict(
            poke_feats=env_player.bc.poke_feats,
            move_feats=env_player.bc.move_feats,
            features_dim=cfg.NETWORK.POKEMON_FEATURE_SIZE,
        ),
    )
    model = DQN(
        DqnMlpPolicy,
        env_player,
        policy_kwargs=policy_kwargs,
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

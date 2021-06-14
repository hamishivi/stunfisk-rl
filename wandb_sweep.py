from yacs.config import CfgNode
from config import cfg
from poke_env.player.random_player import RandomPlayer
from stable_baselines3 import DQN, PPO
from stable_baselines3.dqn import MlpPolicy as DqnMlpPolicy
from stable_baselines3.ppo import MlpPolicy as PpoMlpPolicy
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


def unflatten_dict(raw_config):
    resultDict = dict()
    for key, value in raw_config.items():
        parts = key.split("-")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


# main driver for running stuff
# gen8anythinggoes
def train_and_test(cfg):
    env_player = SimpleRLPlayer(
        cfg, battle_format="gen8anythinggoes", team=open("teams/red.txt", "r").read()
    )
    max_opponent = MaxDamagePlayer(
        battle_format="gen8anythinggoes", team=open("teams/red.txt", "r").read()
    )
    rand_opponent = RandomPlayer(
        battle_format="gen8anythinggoes", team=open("teams/red.txt", "r").read()
    )
    policy_kwargs = dict(
        features_extractor_class=PokemonFeatureExtractor,
        net_arch=[cfg.NETWORK.POKEMON_FEATURE_SIZE]
        + [cfg.NETWORK.HIDDEN_LAYER_SIZE] * cfg.NETWORK.NUM_LAYERS,
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
        verbose=1,
        tensorboard_log="./dqn_pokemon_tensorboard/",
    )
    # model = PPO(
    #    PpoMlpPolicy,
    #    env_player,
    #    policy_kwargs=policy_kwargs,
    #    learning_rate=cfg.DQN.LEARNING_RATE,
    #    gamma=cfg.DQN.GAMMA,
    #    verbose=1,
    #    tensorboard_log="./dqn_pokemon_tensorboard/",
    # )
    # train against both?
    # train(env_player, rand_opponent, model, timesteps=cfg.DQN.TRAIN_TIMESTEPS)
    # print("evaluating random....")
    train(env_player, max_opponent, model, timesteps=cfg.DQN.TRAIN_TIMESTEPS)
    model.save("dqn_pokemon")
    rand_won = test(env_player, rand_opponent, model)
    max_won = test(env_player, max_opponent, model)
    return rand_won, max_won


raw_cfg = cfg_node_to_dict(cfg)
raw_cfg_flat = flatten_dict(raw_cfg)
# wandb.init(config=raw_cfg_flat, project="stunfisk-rl")
# load in wandb config, merge into cfg
# raw_cfg = unflatten_dict(dict(wandb.config))
cfg.merge_from_other_cfg(CfgNode(raw_cfg))
r, m = train_and_test(cfg)
print({"rand_won": r, "max_won": m, "avg_won": (r + m) / 2})
# wandb.log({"rand_won": r, "max_won": m, "avg_won": (r + m) / 2})

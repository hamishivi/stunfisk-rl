from yacs.config import CfgNode
from config import cfg
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy as DqnMlpPolicy
from rl_player import SimpleRLPlayer
from pokefeat_extractor import PokemonFeatureExtractor
import yaml
from data import TYPES
from poke_env.environment.pokemon_type import PokemonType
import gradio as gr
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
from torch.nn.functional import softmax


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


raw_cfg = cfg_node_to_dict(cfg)
cfg.merge_from_other_cfg(CfgNode(raw_cfg))
cfg.merge_from_file("basic.yaml")


def load_model(model_file, cfg, battle_format):
    env_player = SimpleRLPlayer(cfg, battle_format=battle_format)
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
        tensorboard_log="./dqn_pokemon_tensorboard/",
        device="cpu",
    )
    model.load(model_file, custom_objects={"policy_kwargs": policy_kwargs})
    return model


model = load_model(
    "/Users/hamishivison/Programming/stunfisk-data/gen5.zip", cfg, "gen5randombattle"
)


def predict(observation):
    q_values = model.predict(observation)
    print(q_values)
    return q_values.reshape(-1)


# glue for gradio interface
# yes, its monstrous due to the massive number of variables going into the model.
def predict(
    poke_move_1_bsp,
    poke_move_1_mm,
    poke_move_2_bsp,
    poke_move_2_mm,
    poke_move_3_bsp,
    poke_move_3_mm,
    poke_move_4_bsp,
    poke_move_4_mm,
    poke_atk,
    poke_def,
    poke_spa,
    poke_spd,
    poke_spe,
    poke_hp_frac,
    poke_type1,
    poke_type2,
    enemy_move_1_bsp,
    enemy_move_1_mm,
    enemy_move_2_bsp,
    enemy_move_2_mm,
    enemy_move_3_bsp,
    enemy_move_3_mm,
    enemy_move_4_bsp,
    enemy_move_4_mm,
    enemy_atk,
    enemy_def,
    enemy_spa,
    enemy_spd,
    enemy_spe,
    enemy_hp_frac,
    enemy_type1,
    enemy_type2,
):
    obs = {
        "ours.0.moves.0.bsp": poke_move_1_bsp,
        "ours.0.moves.0.mm": poke_move_1_mm,
        "ours.0.moves.1.bsp": poke_move_2_bsp,
        "ours.0.moves.1.mm": poke_move_2_mm,
        "ours.0.moves.2.bsp": poke_move_3_bsp,
        "ours.0.moves.2.mm": poke_move_3_mm,
        "ours.0.moves.3.bsp": poke_move_4_bsp,
        "ours.0.moves.3.mm": poke_move_4_mm,
        "ours.0.atk": poke_atk,
        "ours.0.def": poke_def,
        "ours.0.spa": poke_spa,
        "ours.0.spd": poke_spd,
        "ours.0.spe": poke_spe,
        "ours.0.hp_frac": poke_hp_frac,
        "ours.0.type1": TYPES[
            PokemonType.from_name(poke_type1) if poke_type1 != "NONE" else None
        ],
        "ours.0.type2": TYPES[
            PokemonType.from_name(poke_type2) if poke_type2 != "NONE" else None
        ],
        "enemy.0.moves.0.bsp": enemy_move_1_bsp,
        "enemy.0.moves.0.mm": enemy_move_1_mm,
        "enemy.0.moves.1.bsp": enemy_move_2_bsp,
        "enemy.0.moves.1.mm": enemy_move_2_mm,
        "enemy.0.moves.2.bsp": enemy_move_3_bsp,
        "enemy.0.moves.2.mm": enemy_move_3_mm,
        "enemy.0.moves.3.bsp": enemy_move_4_bsp,
        "enemy.0.moves.3.mm": enemy_move_4_mm,
        "enemy.0.atk": enemy_atk,
        "enemy.0.def": enemy_def,
        "enemy.0.spa": enemy_spa,
        "enemy.0.spd": enemy_spd,
        "enemy.0.spe": enemy_spe,
        "enemy.0.hp_frac": enemy_hp_frac,
        "enemy.0.type1": TYPES[
            PokemonType.from_name(enemy_type1) if enemy_type1 != "NONE" else None
        ],
        "enemy.0.type2": TYPES[
            PokemonType.from_name(enemy_type2) if enemy_type2 != "NONE" else None
        ],
    }
    # I want all values, so i have to manually do some of the work stable baselines does
    obs = {k: np.array(v).reshape(1, -1) for k, v in obs.items()}
    obs = obs_as_tensor(obs, "cpu")

    q_values = model.policy.q_net.forward(obs).reshape(-1)
    # map q value to action labels.
    return {
        "move 1": q_values[0].item(),
        "move 2": q_values[1].item(),
        "move 3": q_values[2].item(),
        "move 4": q_values[3].item(),
        # "switch to pokemon 1": q_values[4].item(),
        # "switch to pokemon 2": q_values[5].item(),
        # "switch to pokemon 3": q_values[6].item(),
        # "switch to pokemon 4": q_values[7].item(),
        # "switch to pokemon 5": q_values[8].item(),
    }


type_strings = [str(t).split()[0] for t in TYPES.keys()]

gr_inputs = [
    gr.inputs.Slider(0, 200, 1, default=70, label="Move 1 Base Power"),
    gr.inputs.Slider(
        0, 5, 0.5, default=1, label="Move 1 Type Effectiveness Multiplier"
    ),
    gr.inputs.Slider(0, 200, 1, default=80, label="Move 2 Base Power"),
    gr.inputs.Slider(
        0, 5, 0.5, default=2, label="Move 2 Type Effectiveness Multiplier"
    ),
    gr.inputs.Slider(0, 200, 1, default=100, label="Move 3 Base Power"),
    gr.inputs.Slider(
        0, 5, 0.5, default=1, label="Move 3 Type Effectiveness Multiplier"
    ),
    gr.inputs.Slider(0, 200, 1, default=70, label="Move 4 Base Power"),
    gr.inputs.Slider(
        0, 5, 0.5, default=1, label="Move 4 Type Effectiveness Multiplier"
    ),
    gr.inputs.Slider(0, 200, default=80, label="Attack"),
    gr.inputs.Slider(0, 250, default=100, label="Defence"),
    gr.inputs.Slider(0, 200, default=50, label="Special Attack"),
    gr.inputs.Slider(0, 250, default=80, label="Special Defence"),
    gr.inputs.Slider(0, 200, default=70, label="Speed"),
    gr.inputs.Slider(0, 1, default=1, label="HP %"),
    gr.inputs.Dropdown(type_strings, default="NORMAL", label="Type 1"),
    gr.inputs.Dropdown(type_strings, default="NONE", label="Type 2"),
    gr.inputs.Slider(0, 200, 1, default=70, label="Opponent Move 1 Base Power"),
    gr.inputs.Slider(
        0, 5, 0.5, default=1, label="Opponent Move 1 Type Effectiveness Multiplier"
    ),
    gr.inputs.Slider(0, 200, 1, default=80, label="Opponent Move 2 Base Power"),
    gr.inputs.Slider(
        0, 5, 0.5, default=1, label="Opponent Move 2 Type Effectiveness Multiplier"
    ),
    gr.inputs.Slider(0, 200, 1, default=120, label="Opponent Move 3 Base Power"),
    gr.inputs.Slider(
        0, 5, 0.5, default=2, label="Opponent Move 3 Type Effectiveness Multiplier"
    ),
    gr.inputs.Slider(0, 200, 1, default=70, label="Opponent Move 4 Base Power"),
    gr.inputs.Slider(
        0, 5, 0.5, default=1, label="Opponent Move 4 Type Effectiveness Multiplier"
    ),
    gr.inputs.Slider(0, 200, default=100, label="Enemy Attack"),
    gr.inputs.Slider(0, 250, default=50, label="Enemy Defence"),
    gr.inputs.Slider(0, 200, default=80, label="Enemy Special Attack"),
    gr.inputs.Slider(0, 250, default=70, label="Enemy Special Defence"),
    gr.inputs.Slider(0, 200, default=65, label="Enemy Speed"),
    gr.inputs.Slider(0, 1, default=1, label="Enemy HP %"),
    gr.inputs.Dropdown(type_strings, default="NORMAL", label="Enemy Type 1"),
    gr.inputs.Dropdown(type_strings, default="NONE", label="Enemy Type 2"),
]


gr_outputs = [gr.outputs.Label()]


iface = gr.Interface(fn=predict, inputs=gr_inputs, outputs=gr_outputs, live=True)

iface.launch()

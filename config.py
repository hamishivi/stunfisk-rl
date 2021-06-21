from yacs.config import CfgNode as CN

_C = CN()
## Battle tensor options
_C.BATTLE = CN()
# move options
_C.BATTLE.MOVE = CN()
_C.BATTLE.MOVE.ACCURACY = True
_C.BATTLE.MOVE.BASE_POWER = True
_C.BATTLE.MOVE.MOVE_MULT = True
_C.BATTLE.MOVE.PP = True
_C.BATTLE.MOVE.PRIORITY = True
_C.BATTLE.MOVE.CAT = True
_C.BATTLE.MOVE.TYPE = True
# pokemon options
_C.BATTLE.POKEMON = CN()
_C.BATTLE.POKEMON.ACTIVE = True
_C.BATTLE.POKEMON.ATTACK = True
_C.BATTLE.POKEMON.DEFENCE = True
_C.BATTLE.POKEMON.SPECIAL_ATTACK = True
_C.BATTLE.POKEMON.SPECIAL_DEFENCE = True
_C.BATTLE.POKEMON.SPEED = True
_C.BATTLE.POKEMON.HP = True
_C.BATTLE.POKEMON.HP_FRACTION = True
_C.BATTLE.POKEMON.FAINTED = True
_C.BATTLE.POKEMON.GENDER = True
_C.BATTLE.POKEMON.TYPE1 = True
_C.BATTLE.POKEMON.TYPE2 = True
_C.BATTLE.POKEMON.NUM_MOVES = 4
# general battle options
_C.BATTLE.TEAM = CN()
_C.BATTLE.TEAM.SIZE = 1

## Reward options
_C.REWARD = CN()
_C.REWARD.FAINTED = 50
_C.REWARD.HP = 10
_C.REWARD.VICTORY = 5000

## DQN params
_C.DQN = CN()
_C.DQN.LEARNING_RATE = 0.0001
_C.DQN.BUFFER_SIZE = 10000
_C.DQN.LEARNING_STARTS = 1000
_C.DQN.GAMMA = 0.9
_C.DQN.TRAIN_TIMESTEPS = 100000

## Network size details
_C.NETWORK = CN()
_C.NETWORK.POKEMON_FEATURE_SIZE = 6000
_C.NETWORK.HIDDEN_LAYER_SIZE = 500
_C.NETWORK.NUM_LAYERS = 10

# using global singleton pattern (should swap later)
cfg = _C

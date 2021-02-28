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
_C.BATTLE.TEAM.SIZE = 6

## Reward options
_C.REWARD = CN()
_C.REWARD.FAINTED = 2
_C.REWARD.HP = 1
_C.REWARD.VICTORY = 30


cfg = _C
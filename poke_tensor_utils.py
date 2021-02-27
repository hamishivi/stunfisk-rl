'''
Utilities for converting pokemon and battle-related items to tensors.
'''
import numpy as np
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.pokemon_gender import PokemonGender
from poke_env.environment.move_category import MoveCategory

types = {
    PokemonType.BUG: 0,
    PokemonType.DARK: 1,
    PokemonType.DRAGON: 2,
    PokemonType.ELECTRIC: 3,
    PokemonType.FAIRY: 4,
    PokemonType.FIGHTING: 5,
    PokemonType.FIRE: 6,
    PokemonType.FLYING: 7,
    PokemonType.GHOST: 8,
    PokemonType.GRASS: 9,
    PokemonType.GROUND: 10,
    PokemonType.ICE: 11,
    PokemonType.NORMAL: 12,
    PokemonType.POISON: 13,
    PokemonType.PSYCHIC: 14,
    PokemonType.ROCK: 15,
    PokemonType.STEEL: 16,
    PokemonType.WATER: 17,
    None: 18
}

move_cats = {
    MoveCategory.PHYSICAL: 0,
    MoveCategory.SPECIAL: 1,
    MoveCategory.STATUS: 2
}

genders = {
    PokemonGender.FEMALE: 0,
    PokemonGender.MALE: 1,
    PokemonGender.NEUTRAL: 2
}

def bool2int(b):
    return 1 if b else 0

def move_to_tensor(move, damage_func):
    '''
    Convert move into tensor.
    TODO: handle unknown values
    '''
    accuracy = move.accuracy
    base_power = move.base_power
    move_mult = damage_func(move)
    pp = move.current_pp
    priority = move.priority + 7 # lowest priority is -7
    category = move_cats[move.category]
    one_hot_cat = np.eye(len(move_cats))[category]
    move_type = types[move.type]
    move_type_one_hot = np.eye(len(types))[move_type]
    move_arr = [move_mult, accuracy, base_power, pp, priority]
    return np.concatenate([
        np.array(move_arr),
        one_hot_cat,
        move_type_one_hot
    ], axis=0)

def move_shape():
    '''
    for defining observation shape, shape of a move with bounds
    '''
    num_elements = 5 + len(move_cats) + len(types)
    return {
        'shape': (num_elements,),
        'lower_bounds': [0] * num_elements,
        'upper_bounds': [5, 1, 200, 50, 12] + [1]*len(move_cats) + [1]*len(types)
    }

def dummy_move_tensor():
    return np.zeros(move_shape()['shape'][0])


def poke_to_tensor(pokemon, damage_func):
    '''
    Convert pokemon object to tensor. Currently is mainly base stats and moves.
    TODO: handle unknown values (for opponent pokemon)
    '''
    opponent = 0
    active = bool2int(pokemon.active)
    stats = [pokemon.stats[k] for k in pokemon.stats]
    # sometimes current hp can be 'None'
    current_hp = pokemon.current_hp if pokemon.current_hp else 0
    current_hp_fraction = pokemon.current_hp_fraction
    fainted = bool2int(pokemon.fainted)
    gender = genders[pokemon.gender]
    ptypes = [types[t] for t in pokemon.types]
    if len(ptypes) == 1:
        ptypes.append(types['NULL'])
    one_hot_types = np.eye(len(types))[np.array(ptypes)].flatten()
    one_hot_gender = np.eye(len(genders))[np.array(gender)]
    pokemon_arr = [opponent, active] + stats + [current_hp, current_hp_fraction, fainted]
    # ignore gmax moves, these are added onto end of thing...
    # and sometimes it bugs out with zoarark...
    move_tensors = [move_to_tensor(pokemon.moves[m], damage_func) for m in pokemon.moves if 'gmax' not in m][:4]
    while len(move_tensors) < 4:
        move_tensors.append(dummy_move_tensor())
    return np.concatenate([np.array(pokemon_arr), one_hot_gender, one_hot_types] + move_tensors, axis=0)

def poke_shape():
    '''
    for defining observation shape, shape of a pokemon with ounds
    '''
    move_shapes = move_shape()
    num_elements = move_shapes['shape'][0]*4 + 9 + len(genders) + len(types)*2 + 1
    return {
        'shape': (num_elements,),
        'lower_bounds': [0]*num_elements,
        'upper_bounds': [1, 1, 2000, 2000, 2000, 2000, 2000, 2000, 1, 1] + [1]*len(genders) + [1]*len(types)*2 + 4*move_shapes['upper_bounds']
    }

def opposing_poke_to_tensor(pokemon, damage_func):
    opponent = 1
    active = 1
    stats = [pokemon.base_stats[k] for k in pokemon.base_stats if k != 'hp']
    # sometimes current hp can be 'None'
    current_hp = pokemon.base_stats['hp'] * pokemon.current_hp_fraction
    current_hp_fraction = pokemon.current_hp_fraction
    fainted = 0
    gender = genders[pokemon.gender]
    ptypes = [types[t] for t in pokemon.types]
    if len(ptypes) == 1:
        ptypes.append(types['NULL'])
    one_hot_types = np.eye(len(types))[np.array(ptypes)].flatten()
    one_hot_gender = np.eye(len(genders))[np.array(gender)]
    pokemon_arr = [opponent, active] + stats + [current_hp, current_hp_fraction, fainted]
    # ignore gmax moves, these are added onto end of thing...
    # and sometimes it bugs out with zoarark...
    move_tensors = [move_to_tensor(pokemon.moves[m], damage_func) for m in pokemon.moves if 'gmax' not in m][:4]
    while len(move_tensors) < 4:
        move_tensors.append(dummy_move_tensor())
    return np.concatenate([np.array(pokemon_arr), one_hot_gender, one_hot_types] + move_tensors, axis=0)
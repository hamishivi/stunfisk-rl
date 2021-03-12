"""
Various data things
"""
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.pokemon_gender import PokemonGender
from poke_env.environment.move_category import MoveCategory

TYPES = {
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
    None: 18,
}

MOVE_CATS = {MoveCategory.PHYSICAL: 0, MoveCategory.SPECIAL: 1, MoveCategory.STATUS: 2}

GENDERS = {PokemonGender.FEMALE: 0, PokemonGender.MALE: 1, PokemonGender.NEUTRAL: 2}

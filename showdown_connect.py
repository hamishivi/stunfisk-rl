# -*- coding: utf-8 -*-
import asyncio
import logging

from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
from gym.wrappers import FlattenObservation
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy

from config import cfg
from rl_player import SimpleRLPlayer, EvaluatePlayer
from max_player import MaxDamagePlayer


async def main():
    logging.getLogger().setLevel(logging.DEBUG)
    # Setup. In order to use stable-baselines native loading, I have to wrap the agent in
    # a special poke-env player to allow the connectivity to work.
    # player = FlattenObservation(SimpleRLPlayer(
    #    cfg,
    #    battle_format='gen8randombattle',
    # ))
    # model = DQN.load(
    #    "pokemon_trained.model",
    #    policy=MlpPolicy,
    #    env=player,
    #    learning_rate=cfg.DQN.LEARNING_RATE,
    #    buffer_size=cfg.DQN.BUFFER_SIZE,
    #    learning_starts=cfg.DQN.LEARNING_STARTS,
    #    gamma=cfg.DQN.GAMMA,
    # )
    # showdown_player = EvaluatePlayer(
    #    player,
    #    model,
    #    player_configuration=PlayerConfiguration("hambot97", 'password'),
    #    server_configuration=ShowdownServerConfiguration,
    #    start_timer_on_battle_start=True, # to avoid us dealing with unresponsive players
    #    battle_format='gen8randombattle'
    # )
    # other players to benchmark against
    # showdown_player = MaxDamagePlayer(
    #     battle_format='gen8randombattle',
    #     player_configuration=PlayerConfiguration("hambot98", 'password'),
    #     server_configuration=ShowdownServerConfiguration,
    #     start_timer_on_battle_start=True
    # )
    showdown_player = RandomPlayer(
        battle_format="gen8randombattle",
        player_configuration=PlayerConfiguration("hambot99", "password"),
        server_configuration=ShowdownServerConfiguration,
        start_timer_on_battle_start=True,
    )

    #### Below we can setup the way we want the agent to connect to showdown

    # Sending challenges to 'your_username'
    # await showdown_player.send_challenges("your_username", n_challenges=1)

    # Accepting one challenge from any user. replace none with string to accept only
    # from specific usernames
    # await showdown_player.accept_challenges(None, 1)

    # Playing 5 games on the ladder
    await showdown_player.ladder(100)

    # Print the rating of the player and its opponent after each battle
    for battle in showdown_player.battles.values():
        print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

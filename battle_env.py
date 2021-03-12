import asyncio

from rl_player import SimpleRLPlayer, EvaluatePlayer
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration

from max_player import MaxDamagePlayer


def train(env_player, opponent, model, timesteps=100000):
    def learn(player, model):
        model.learn(total_timesteps=timesteps)

    env_player.play_against(
        env_algorithm=learn, opponent=opponent, env_algorithm_kwargs={"model": model}
    )


def test(env_player, opponent, model, algo_name="DQN", eval_eps=100):
    def evaluate(player, model):
        player.reset_battles()
        evaluate_policy(model, player, n_eval_episodes=eval_eps)
        # print(f"{algo_name} Evaluation: {player.n_won_battles} victories out of {100} episodes")

    env_player.play_against(
        env_algorithm=evaluate, opponent=opponent, env_algorithm_kwargs={"model": model}
    )
    return env_player.n_won_battles / eval_eps


async def play_human(env_player, model):
    player = EvaluatePlayer(
        env_player,
        model,
        player_configuration=PlayerConfiguration("ROBOHAMISH", "robotmish"),
    )
    await player.accept_challenges(None, 1)


if __name__ == "__main__":
    from config import cfg

    env_player = SimpleRLPlayer(cfg, battle_format="gen8randombattle")
    opponent = MaxDamagePlayer(battle_format="gen8randombattle")
    model = DQN(
        MlpPolicy,
        env_player,
        learning_rate=0.00025,
        buffer_size=10000,
        learning_starts=1000,
        gamma=0.5,
        verbose=1,
    )
    print("training")
    train(env_player, opponent, model)
    print("testing")
    won = test(env_player, opponent, model)
    print(f"In testing we won {won*100}% of matches")
    # asyncio.get_event_loop().run_until_complete(play_human(env_player, model))

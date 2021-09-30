from rl_player import EvaluatePlayer
from stable_baselines.common.evaluation import evaluate_policy
from poke_env.player_configuration import PlayerConfiguration
from gym.wrappers import FlattenObservation


def train(env_player, opponent, model, timesteps=100000):
    def learn(player, model):
        model.learn(total_timesteps=timesteps)

    env_player.play_against(
        env_algorithm=learn, opponent=opponent, env_algorithm_kwargs={"model": model}
    )


def test(env_player, opponent, model, eval_eps=100):
    def evaluate(player, model):
        player.reset_battles()
        evaluate_policy(model, FlattenObservation(player), n_eval_episodes=eval_eps)
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

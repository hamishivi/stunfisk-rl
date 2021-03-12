"""
Testing the different algos
"""
import asyncio
from stable_baselines3 import A2C, a2c, DQN, dqn, HER, PPO, ppo

from battle_env import train, test, play_human
from max_player import MaxDamagePlayer
from rl_player import SimpleRLPlayer
from poke_env.player.random_player import RandomPlayer


def test_algo(algo, algo_name, env_player, opponent, play_against=False):
    print(f"Training {algo_name}")
    train(env_player, opponent, algo, timesteps=10000)
    print(f"Evaluating {algo_name}")
    win_perct = test(env_player, opponent, algo, algo_name=algo_name)
    if play_against:
        print(f"You can now play against the {algo_name} agent")
        asyncio.get_event_loop().run_until_complete(play_human(env_player, algo))
    return win_perct


env_player = SimpleRLPlayer(battle_format="gen8randombattle")
opponent = RandomPlayer(battle_format="gen8randombattle")

a2c_agent = A2C(a2c.MlpPolicy, env_player, learning_rate=0.00025, gamma=0.5)

dqn_agent = DQN(
    dqn.MlpPolicy,
    env_player,
    learning_rate=0.00025,
    buffer_size=10000,
    learning_starts=1000,
    gamma=0.5,
)

# requires special env stuff
# her_agent = HER('MlpPolicy',
#    env_player,
#    DQN,
#    n_sampled_goal=4,
#    goal_selection_strategy='future',
#    online_sampling=True,
#    max_episode_length=100) # unknown??

ppo_agent = PPO(ppo.MlpPolicy, env_player, learning_rate=0.00025, gamma=0.5)

agents = {
    "A2C": a2c_agent,
    "DQN": dqn_agent,
    #'HER': her_agent,
    "PPO": ppo_agent,
}

res = {}
for agent in agents:
    res[agent] = test_algo(agents[agent], agent, env_player, opponent)

print(res)

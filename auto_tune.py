import asyncio
from stable_baselines3 import A2C, a2c, DQN, dqn, HER, PPO, ppo
import optuna

from battle_env import train, test
from max_player import MaxDamagePlayer
from rl_player import SimpleRLPlayer

def objective(trial):
    # currently just test against this
    agent_type = 'DQN' #trial.suggest_categorical('agent', ['A2C', 'DQN', 'PPO'])

    env_player = SimpleRLPlayer(battle_format="gen8randombattle")
    opponent = MaxDamagePlayer(battle_format="gen8randombattle")
    
    if agent_type == 'A2C':
        agent = A2C(
            a2c.MlpPolicy,
            env_player,
            learning_rate=0.00025,
            gamma=0.5)
    elif agent_type == 'DQN':
        lr = trial.suggest_float('learning_rate', 0, 1)
        gamma = trial.suggest_float('gamma', 0, 0.99)
        agent = DQN(
            dqn.MlpPolicy,
            env_player,
            learning_rate=lr,
            buffer_size=10000,
            learning_starts=1000,
            gamma=gamma)
    else:
        agent = PPO(
            ppo.MlpPolicy,
            env_player,
            learning_rate=0.00025,
            gamma=0.5)

    train_steps = trial.suggest_int('train_steps', 0, 100000000)
    train(env_player, opponent, agent, timesteps=train_steps)

    win_perct = test(env_player, opponent, agent)

    return win_perct

# currently testing 3 key params: LR, g, train steps.
study = optuna.create_study()
study.optimize(objective, n_trials=100, show_progress_bar=True)

from cookie_rl.radius_environment import TwoPlayerCakeCuttingEnv, SelfPlayCakeCuttingEnv, SelfPlayCallback
from cookie_rl.models import MultiAgentActorCriticPolicy
from dataclasses import dataclass
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


@dataclass
class TrainingConfig:

    num_processes: int = 8
    n_players: int = 3
    angle_encoder_dim: int = 32,
    hidden_dim: int = 128
    learning_rate: float = 1.0e-4
    batch_size: int = 128
    num_steps: int = 10_000
    tensorboard_dir: str = "./tensorboard"

    def get_policy_kwargs(self):
        return {
            'n_players': self.n_players,
            'angle_encoder_dim': self.angle_encoder_dim,
            'hidden_dim': self.hidden_dim
        }




def train(config):
    num_processes = config.num_processes
    policy_kwargs = config.get_policy_kwargs()
    callback = SelfPlayCallback(update_freq=2048)

    env = DummyVecEnv([lambda: TwoPlayerCakeCuttingEnv() for _ in range(num_processes)])

    model = PPO(
        MultiAgentActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=config.learning_rate,
        n_steps=10,
        batch_size=config.batch_size,
        gamma=1.,
        tensorboard_log=config.tensorboard_dir
    )

    model.learn(total_timesteps=config.num_steps, callback=callback)

    return model


def evaluate(model, num_trials: int):

    env = TwoPlayerCakeCuttingEnv(policy=model, fixed_angle=0.0, noise_std=0.1)
    
    for episode in range(num_trials):
        angles = []
        
        # Reset environment
        env.angles = []
        env.current_player = 0
        env.done = False
        
        # Player 0: fixed
        env.angles.append(env.fixed_angle)
        env.current_player = 1
        
        # Player 1: use policy with proper observation
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        angles.append(env.fixed_angle)
        angles.append(float(action[0]))
        env.angles.append(float(action[0]))
        env.current_player = 2
        
        # Player 2: use policy with proper observation
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        angles.append(float(action[0]))
        
        # Compute results using the angles we collected
        env.angles = angles
        wedge_areas = env._compute_wedge_areas()
        player_areas = env._greedy_assignment(wedge_areas)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Angles: {[f'{a:.4f}' for a in angles]}")
        sorted_angles = sorted([(a + np.pi) % (2 * np.pi) for a in angles])
        print(f"  Sorted: {[f'{a:.4f}' for a in sorted_angles]}")
        print(f"  Wedge areas: {[f'{a:.4f}' for a in wedge_areas]}")
        print(f"  Player 0 (fixed, picks 1st):  {player_areas[0]:.4f}")
        print(f"  Player 1 (learned, picks 2nd): {player_areas[1]:.4f}")
        print(f"  Player 2 (learned, picks 3rd): {player_areas[2]:.4f}")


if __name__ == "__main__":

    ###########################
    num_processes= 8
    n_players = 3
    angle_encoder_dim: int = 64,
    hidden_dim: int = 256
    learning_rate: float = 1.0e-4
    batch_size: int = 8129
    num_steps: int = 500_000

    seed = 3
    tensorboard_dir="./ppo_radii_tensorboard/"
    num_trials = 50
    ###########################

    torch.manual_seed(seed)

    config = TrainingConfig(
        num_processes, 
        n_players,
        angle_encoder_dim, 
        hidden_dim, 
        learning_rate, 
        batch_size, 
        num_steps
    )

    model = train(config)
    model.save(f'radii_agents_two_player-{seed}.zip')

    evaluate(model, num_trials)

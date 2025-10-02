import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class SelfPlayCallback(BaseCallback):
    """
    Callback that updates the environment's policy pointer every N steps.
    """
    
    def __init__(self, update_freq=2048, verbose=0):
        super().__init__(verbose)
        self.update_freq = update_freq
    
    def _on_step(self) -> bool:
        # Update environment policy every update_freq steps
        if self.n_calls % self.update_freq == 0:
            # Update all environments in the vec env
            if hasattr(self.training_env, 'env_method'):
                self.training_env.env_method('set_policy', self.model)
            elif hasattr(self.training_env, 'set_policy'):
                self.training_env.set_policy(self.model)
        
        return True
    


class TwoPlayerCakeCuttingEnv(gym.Env):
    """
    Two-player cake cutting with self-play:
    - Player 0: Always cuts at 0 (with small noise)
    - Player 1: Cuts at some angle 
    - Player 2: Cuts at some angle 
    - Players pick in order 0, 1, 2 (but player 0 always gets same expected value)
    
    Alternates training player 1 and player 2.
    """
    
    def __init__(self, policy=None, fixed_angle=0.0, noise_std=0.1):
        super().__init__()
        
        self.n_players = 3
        self.radius = 1.0
        self.policy = policy
        self.fixed_angle = fixed_angle
        self.noise_std = noise_std
        
        self.action_space = spaces.Box(
            low=-np.pi, 
            high=np.pi, 
            shape=(1,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(self.n_players + self.n_players,),
            dtype=np.float32
        )
        
        self.episode_count = 0
        self.current_training_player = 1  # Only train players 1 and 2
        
        self.reset()
    
    def set_policy(self, policy):
        """Called by the training callback to update the policy."""
        self.policy = policy
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Alternate between training player 1 and player 2
        # Player 0 is always deterministic
        self.current_training_player = 1 + (self.episode_count % 2)
        self.episode_count += 1
        
        self.angles = []
        self.current_player = 0
        self.done = False
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        angles_padded = self.angles + [0.0] * (self.n_players - len(self.angles))
        
        player_one_hot = [0.0] * self.n_players
        if not self.done:
            player_one_hot[self.current_player] = 1.0
        
        obs = np.array(angles_padded + player_one_hot, dtype=np.float32)
        return obs
    
    def _compute_wedge_areas(self):
        sorted_angles = sorted([(angle + np.pi) % (2 * np.pi) for angle in self.angles])
        
        wedge_angles = []
        for i in range(len(sorted_angles)):
            next_angle = sorted_angles[(i + 1) % len(sorted_angles)]
            angle_diff = (next_angle - sorted_angles[i]) % (2 * np.pi)
            wedge_angles.append(angle_diff)
        
        wedge_areas = [angle / 2.0 for angle in wedge_angles]
        
        return sorted(wedge_areas, reverse=True)
    
    def _greedy_assignment(self, wedge_areas):
        player_areas = [0.0] * self.n_players
        
        for player in range(min(self.n_players, len(wedge_areas))):
            player_areas[player] = wedge_areas[player]
        
        return player_areas
    
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done, call reset()")
        
        # Determine the action for current player
        if self.current_player == 0:
            # Player 0: Always cuts at fixed_angle with small noise
            noise = np.random.normal(0, self.noise_std)
            angle = self.fixed_angle + noise
        elif self.current_player != self.current_training_player:
            # Not training player (1 or 2) - use current policy
            if self.policy is not None:
                obs = self._get_obs()
                action_pred, _ = self.policy.predict(obs, deterministic=False)
                angle = float(action_pred[0])
            else:
                # No policy yet (early training), use random
                angle = self.action_space.sample()[0]
        else:
            # Training player - use provided action
            angle = float(action[0])
        
        self.angles.append(angle)
        current_player = self.current_player
        self.current_player += 1
        
        # Continue until we reach the training player's turn or episode ends
        if self.current_player < self.n_players:
            if self.current_player == self.current_training_player:
                # Training player's turn next - return control
                return self._get_obs(), 0.0, False, False, {}
            else:
                # Not training player's turn - auto-step
                return self.step(self.action_space.sample())  # Dummy action, will be replaced
        
        # Episode is done
        self.done = True
        
        wedge_areas = self._compute_wedge_areas()
        player_areas = self._greedy_assignment(wedge_areas)
        
        # Return reward for the training player
        reward = player_areas[self.current_training_player]
        
        info = {
            'player_areas': player_areas,
            'wedge_areas': wedge_areas,
            'episode_done': True,
            'angles_world_frame': self.angles,
            'training_player': self.current_training_player
        }
        
        return self._get_obs(), reward, True, False, info



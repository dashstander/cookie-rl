from typing import Callable, Tuple
from gymnasium import spaces
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class AngleEncoder(nn.Module):
    """Encode angles using random Fourier features for periodicity."""
    
    def __init__(self, encoder_dim: int):
        super().__init__()
        self.register_buffer(
            'frequencies', 
            torch.randn(encoder_dim // 2)
        )
    
    def forward(self, angles):
        """
        Args:
            angles: shape (..., n_angles)
        Returns:
            encoded: shape (..., n_angles * encoder_dim)
        """
        # angles: (..., n_angles), frequencies: (encoder_dim // 2)
        features = angles.unsqueeze(-1) * self.frequencies
        
        # Concatenate cos and sin
        encoded = torch.cat([torch.cos(features), torch.sin(features)], dim=-1)
        
        # Flatten: (..., n_angles, encoder_dim) -> (..., n_angles * encoder_dim)
        return encoded.flatten(start_dim=-2)
    

class AngleDecoder(nn.Module):

    def __init__(self, feature_dim: int):
        super().__init__()
        self.decoder = nn.Linear(feature_dim, 1)

    def forward(self, tensor):
        return torch.pi * torch.tanh(self.decoder(tensor))


class MultiAgentNetwork(nn.Module):
    """
    Custom network with separate policy and value networks for each player.
    Each player gets: encoder -> hidden layers -> policy/value heads
    """

    def __init__(
        self,
        n_players: int = 3,
        angle_encoder_dim: int = 32,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.n_players = n_players
        self.latent_dim_pi = 1
        self.latent_dim_vf = 1
        
        # Shared angle encoder
        self.angle_encoder = AngleEncoder(angle_encoder_dim)
        
        # After encoding: n_players angles * encoder_dim + n_players (one-hot)
        encoded_dim = n_players * angle_encoder_dim + n_players
        
        # Separate networks for each player
        self.player_policy_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoded_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                AngleDecoder(hidden_dim)
            ) for _ in range(n_players)
        ])
        
        self.player_value_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoded_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                AngleDecoder(hidden_dim)
            ) for _ in range(n_players)
        ])

    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode angles and concatenate with player one-hot."""
        angles = observations[:, :self.n_players]
        player_one_hot = observations[:, self.n_players:]
        
        # Encode angles
        encoded_angles = self.angle_encoder(angles)
        
        # Concatenate with player ID
        return torch.cat([encoded_angles, player_one_hot], dim=-1)

    def _route_through_networks(
        self, 
        encoded_input: torch.Tensor, 
        player_one_hot: torch.Tensor,
        network_list: nn.ModuleList
    ) -> torch.Tensor:
        """Route each sample through the appropriate player's network."""
        batch_size = encoded_input.shape[0]
        output_dim = 1  # Get output dim from last layer
        output = torch.zeros(batch_size, output_dim, device=encoded_input.device)
        
        player_ids = player_one_hot.argmax(dim=1)
        
        for player_id in range(self.n_players):
            player_indices = (player_ids == player_id).nonzero(as_tuple=True)[0]
            
            if len(player_indices) > 0:
                player_inputs = encoded_input[player_indices]
                player_outputs = network_list[player_id](player_inputs)
                output[player_indices] = player_outputs
        
        return output

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value
        """
        return self.forward_actor(observations), self.forward_critic(observations)

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_observations(observations)
        player_one_hot = observations[:, self.n_players:]
        return self._route_through_networks(encoded, player_one_hot, self.player_policy_nets)

    def forward_critic(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_observations(observations)
        player_one_hot = observations[:, self.n_players:]
        return self._route_through_networks(encoded, player_one_hot, self.player_value_nets)


class MultiAgentActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        **kwargs,
    ):
        self.policy_kwargs = kwargs.get('policy_kwargs')
        super().__init__(
            observation_space,
            action_space,
            lr_schedule
        )

    def _build_mlp_extractor(self) -> None:
        if self.policy_kwargs:
            self.mlp_extractor = MultiAgentNetwork(
                **self.policy_kwargs
            )
        else:
            self.mlp_extractor = MultiAgentNetwork()


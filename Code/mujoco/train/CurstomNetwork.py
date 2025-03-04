import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Custom MLP with LayerNorm
class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=40):
        super().__init__(observation_space, features_dim)
        self.layers = nn.Sequential(
            nn.Linear(observation_space.shape[0], 50),
            nn.LayerNorm(50),
            nn.ReLU(),
            nn.Linear(50, features_dim),
        )

    def forward(self, x):
        return self.layers(x)
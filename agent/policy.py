import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    @torch.no_grad()
    def act(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.critic(obs).squeeze(-1)
        return action, logp, None, value

    def evaluate(self, obs, actions):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        actions = actions.long().view(-1)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(obs).squeeze(-1)
        
        return logp, value, entropy